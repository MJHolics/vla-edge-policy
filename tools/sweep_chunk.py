"""액션 청킹(개루프 실행 길이) 스윕 — PLAN.md P3의 마지막 손잡이, 아직 안 돌렸다.

`num_steps`(디노이징)는 §2에서, `empty_cameras`는 §3.5에서, 물리/시각 도메인 랜덤화는 §5·§5.1에서
이미 쟀다. 남은 건 PLAN.md ④가 처음부터 "진짜 질문"이라고 짚은 것 하나 — **50스텝을 개루프로
밀어도 되는가, 몇에서 무너지는가**다. `n_action_steps`는 한 번 추론한 청크(`chunk_size=50`) 중
몇 스텝을 재추론 없이 그대로 실행하는지를 정한다. 작을수록 자주 재추론해(반응성↑) 강건하고,
클수록 추론 호출이 줄어 효율적이지만 개루프 드리프트에 취약해진다 — 정확도-재추론빈도 트레이드오프.

num_steps=10(기본값, §2에서 확정한 표준)로 고정하고 n_action_steps만 바꾼다. 태스크·시드는
고정하지 않지만(각 실행이 독립 시드) 같은 구조로 짝짓지는 않는다 — sweep_denoise.py와 동일한
비짝지음 스윕 패턴(여러 조건을 순서대로 비교). 표본 크기(에피소드/태스크)는 호출부에서 --episodes로
조절한다.

사용: python tools/sweep_chunk.py --n-action-steps 1,5,10,25,50 --episodes 15 --tasks all
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
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def eval_bin() -> str:
    cand = Path(sys.executable).with_name("lerobot-eval")
    return str(cand) if cand.exists() else "lerobot-eval"


def run_one(policy: str, suite: str, task_ids: str, episodes: int, batch: int,
            num_steps: int, n_action_steps: int, out_dir: Path, use_async: bool,
            empty_cameras: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        eval_bin(),
        "--policy.path=" + policy,
        "--policy.device=cuda",
        "--policy.empty_cameras={}".format(empty_cameras),
        "--policy.num_steps={}".format(num_steps),
        "--policy.n_action_steps={}".format(n_action_steps),
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
        return {"n_action_steps": n_action_steps, "error": "eval_info.json 없음",
                "rc": proc.returncode, "seconds": round(dt, 1), "tail": tail}

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
        "n_action_steps": n_action_steps, "success": k, "n_episodes": n,
        "success_rate": round(100.0 * k / n, 1) if n else None,
        "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
        "per_task": per_task, "seconds": round(dt, 1), "rc": proc.returncode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="all", help='"all" 또는 "[0,1,2]"')
    ap.add_argument("--n-action-steps", default="1,5,10,25,50")
    ap.add_argument("--num-steps", type=int, default=10, help="디노이징 스텝(고정, §2 표준값)")
    ap.add_argument("--episodes", type=int, default=15, help="태스크당 에피소드")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--async-envs", action="store_true")
    ap.add_argument("--empty-cameras", type=int, default=0, help="0 = 학습 조건")
    ap.add_argument("--out", default="results/chunk_sweep.json")
    ap.add_argument("--work", default="results/chunk_runs")
    args = ap.parse_args()

    values = [int(x) for x in args.n_action_steps.split(",") if x.strip()]
    print("정책={} suite={} tasks={} 에피소드/태스크={} num_steps={} n_action_steps={}".format(
        args.policy, args.suite, args.tasks, args.episodes, args.num_steps, values), flush=True)
    print()

    rows = []
    for nas in values:
        print("--- n_action_steps={} 실행 중 ---".format(nas), flush=True)
        r = run_one(args.policy, args.suite, args.tasks, args.episodes, args.batch,
                    args.num_steps, nas, Path(args.work) / "nas{}".format(nas),
                    args.async_envs, args.empty_cameras)
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
    print("=== 성공률 vs 개루프 실행 길이(n_action_steps) ===")
    print("{:>14} {:>10} {:>18} {:>10}".format("n_action_steps", "성공률%", "Wilson95", "에피소드"))
    for r in ok:
        print("{:>14} {:>10} {:>18} {:>10}".format(
            r["n_action_steps"], r["success_rate"],
            "[{}, {}]".format(r["wilson95"][0], r["wilson95"][1]), r["n_episodes"]))

    if len(ok) >= 2:
        base = min(ok, key=lambda r: r["n_action_steps"])
        print()
        print("가장 짧은 개루프(n_action_steps={}, 가장 반응성 높음) 대비:".format(base["n_action_steps"]))
        for r in ok:
            if r is base:
                continue
            overlap = not (r["wilson95"][1] < base["wilson95"][0]
                           or base["wilson95"][1] < r["wilson95"][0])
            gap = r["success_rate"] - base["success_rate"]
            verdict = ("구간 겹침 — 차이 있다고 말하지 않음" if overlap
                       else ("**하락 {:.1f}%p**".format(-gap) if gap < 0
                             else "상승 {:.1f}%p".format(gap)))
            print("  n_action_steps={:2d}: {:+.1f}%p  → {}".format(r["n_action_steps"], gap, verdict))

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "meta": {"policy": args.policy, "suite": args.suite, "tasks": args.tasks,
                 "episodes_per_task": args.episodes, "batch": args.batch,
                 "async_envs": args.async_envs, "num_steps": args.num_steps,
                 "empty_cameras": args.empty_cameras,
                 "note": "n_action_steps는 개루프 실행 길이(chunk_size=50 중 재추론 없이 쓰는 스텝 수)."},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("저장: {}".format(args.out))


if __name__ == "__main__":
    main()
