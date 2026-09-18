"""디노이징 스텝 스윕 — 지연의 반대쪽 절반인 **성공률**을 잰다.

`bench_policy_latency.py`가 "num_steps를 줄이면 얼마나 빨라지나"를 답한다면, 이 스크립트는
"그래서 얼마나 잃나"를 답한다. **둘을 붙이지 않으면 아무 결론도 안 된다** —
이 레포군에서 INT8이 1.38배를 얻고 검출 절반을 잃었던 전례가 정확히 그 교훈이다.

각 num_steps마다 `lerobot-eval`을 돌리고 `eval_info.json`의 성공 여부를 모은다.
성공률은 스텝 수 기반이라 sim이 느려도 값 자체는 정확하다(wall-clock과 무관).

사용:
  python tools/sweep_denoise.py --num-steps 1,2,3,5,10 --episodes 5 --tasks all
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RENAME_MAP = ('{"observation.images.image":"observation.images.camera1",'
              '"observation.images.image2":"observation.images.camera2"}')


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson 구간 — 이항 비율의 신뢰구간. 에피소드 수가 적을 때 정규근사는 못 쓴다.

    이 레포군에서 이미 쓰던 방법(정확도 비교 시 Wilson 구간). n=50 같은 작은 표본에서
    "78% vs 74%"가 유의한지 아닌지를 눈으로 판단하지 않기 위해 함께 낸다.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def eval_bin() -> str:
    """venv의 lerobot-eval 을 절대경로로 잡는다.

    `venv/bin/python tools/...` 로 부르면 venv의 bin/ 이 PATH에 없어 FileNotFoundError 로 죽는다.
    venv를 활성화한 셸에서만 우연히 동작했고, 밤샘 체인(활성화 없이 $PY 직접 호출)에서 터졌다.
    """
    cand = Path(sys.executable).with_name("lerobot-eval")
    return str(cand) if cand.exists() else "lerobot-eval"


def run_one(policy: str, suite: str, task_ids: str, episodes: int, batch: int,
            num_steps: int, out_dir: Path, use_async: bool, empty_cameras: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        eval_bin(),
        "--policy.path=" + policy,
        "--policy.device=cuda",
        # 2026-08-19: 학습 조건은 0이다(train_config.json). 자세한 근거는 tools/run_eval.sh 주석.
        "--policy.empty_cameras={}".format(empty_cameras),
        "--policy.num_steps={}".format(num_steps),
        "--env.type=libero",
        "--env.task=" + suite,
        "--eval.batch_size={}".format(batch),
        "--eval.n_episodes={}".format(episodes),
        "--output_dir=" + str(out_dir),
        "--rename_map=" + RENAME_MAP,
    ]
    if task_ids and task_ids != "all":
        cmd.append("--env.task_ids=" + task_ids)
    if use_async:
        cmd.append("--eval.use_async_envs=true")

    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    dt = time.time() - t0

    info_path = out_dir / "eval_info.json"
    if not info_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-1200:]
        return {"num_steps": num_steps, "error": "eval_info.json 없음", "rc": proc.returncode,
                "seconds": round(dt, 1), "tail": tail}

    info = json.loads(info_path.read_text(encoding="utf-8"))
    successes = []
    per_task = []
    for entry in info.get("per_task", []):
        s = entry["metrics"]["successes"]
        successes.extend(bool(x) for x in s)
        per_task.append({"task_id": entry.get("task_id"),
                         "success": sum(bool(x) for x in s), "n": len(s)})
    k, n = sum(successes), len(successes)
    lo, hi = wilson(k, n)
    return {
        "num_steps": num_steps, "success": k, "n_episodes": n,
        "success_rate": round(100.0 * k / n, 1) if n else None,
        "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
        "per_task": per_task, "seconds": round(dt, 1), "rc": proc.returncode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="all", help='"all" 또는 "[0,1,2]"')
    ap.add_argument("--num-steps", default="1,2,3,5,10")
    ap.add_argument("--episodes", type=int, default=5, help="태스크당 에피소드")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--async-envs", action="store_true")
    ap.add_argument("--empty-cameras", type=int, default=0,
                    help="0 = 학습 조건. 2026-08-18 스윕은 1로 돌았다(당시 기록 유지)")
    ap.add_argument("--out", default="results/denoise_sweep.json")
    ap.add_argument("--work", default="results/denoise_runs")
    args = ap.parse_args()

    steps = [int(x) for x in args.num_steps.split(",") if x.strip()]
    print("정책={} suite={} tasks={} 에피소드/태스크={} num_steps={}".format(
        args.policy, args.suite, args.tasks, args.episodes, steps), flush=True)
    print()

    rows = []
    for ns in steps:
        print("--- num_steps={} 실행 중 ---".format(ns), flush=True)
        r = run_one(args.policy, args.suite, args.tasks, args.episodes, args.batch,
                    ns, Path(args.work) / "ns{}".format(ns), args.async_envs,
                    args.empty_cameras)
        rows.append(r)
        if "error" in r:
            print("   실패: {} (rc={}, {}초)".format(r["error"], r["rc"], r["seconds"]))
            print("   ...{}".format(r["tail"][-400:]))
        else:
            print("   성공률 {}% ({}/{})  Wilson95 [{}, {}]  · {}초".format(
                r["success_rate"], r["success"], r["n_episodes"],
                r["wilson95"][0], r["wilson95"][1], r["seconds"]), flush=True)

    ok = [r for r in rows if "error" not in r]
    print()
    print("=== 성공률 vs 디노이징 스텝 ===")
    print("{:>10} {:>10} {:>18} {:>10}".format("num_steps", "성공률%", "Wilson95", "에피소드"))
    for r in ok:
        print("{:>10} {:>10} {:>18} {:>10}".format(
            r["num_steps"], r["success_rate"],
            "[{}, {}]".format(r["wilson95"][0], r["wilson95"][1]), r["n_episodes"]))

    if len(ok) >= 2:
        base = max(ok, key=lambda r: r["num_steps"])
        print()
        print("기본값(num_steps={}) 대비:".format(base["num_steps"]))
        for r in ok:
            if r is base:
                continue
            # 구간이 겹치면 "떨어졌다"고 말하지 않는다.
            overlap = not (r["wilson95"][1] < base["wilson95"][0]
                           or base["wilson95"][1] < r["wilson95"][0])
            gap = r["success_rate"] - base["success_rate"]
            verdict = ("구간 겹침 — 차이 있다고 말하지 않음" if overlap
                       else ("**하락 {:.1f}%p**".format(-gap) if gap < 0
                             else "상승 {:.1f}%p".format(gap)))
            print("  num_steps={:2d}: {:+.1f}%p  → {}".format(r["num_steps"], gap, verdict))

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "meta": {"policy": args.policy, "suite": args.suite, "tasks": args.tasks,
                 "episodes_per_task": args.episodes, "batch": args.batch,
                 "async_envs": args.async_envs,
                 "empty_cameras": args.empty_cameras,
                 "note": "성공률은 스텝 기반이라 sim wall-clock과 무관하다."},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("저장: {}".format(args.out))


if __name__ == "__main__":
    main()
