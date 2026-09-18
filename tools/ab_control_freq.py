"""제어 주파수 A/B — 학습 데모의 기록 주기와 평가 env의 제어 주기가 다르다.

배경(2026-08-19):
  60% vs 공개 90%대의 잔여 격차를 좇다가 **fps 값이 세 곳에서 서로 다르다**는 것을 찾았다.

  | 어디 | fps | 무엇을 뜻하나 |
  |---|---|---|
  | 학습 데이터셋 `lerobot/libero` `meta/info.json` | **10.0** | 데모가 기록된 주기 |
  | 체크포인트 `train_config.json` 의 env 스탠자 | 30 | 학습 때는 env를 안 쓰므로 사실상 사문 |
  | 평가 `LiberoEnv.fps` (기본값) | **20** | `gym_kwargs["control_freq"]` 로 robosuite에 그대로 들어간다 |

  즉 **10Hz로 기록된 행동을 20Hz로 실행**하고 있을 수 있다. 이 프로젝트의 지연 예산
  (1스텝 = 50ms)도 20Hz 가정 위에 서 있으므로, 이게 틀리면 예산이 100ms가 되고
  "20Hz 폐루프에 1.3배 모자란다"는 결론 자체가 뒤집힌다.

  추측으로 끝내지 않고 잰다. 손잡이는 `--env.fps` 하나만 바꾼다.

사용: python tools/ab_control_freq.py --fps 20,10 --episodes 5
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


def run_arm(policy: str, suite: str, episodes: int, batch: int, num_steps: int,
            empty_cameras: int, fps: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        eval_bin(),
        "--policy.path=" + policy,
        "--policy.device=cuda",
        "--policy.empty_cameras={}".format(empty_cameras),
        "--policy.num_steps={}".format(num_steps),
        "--env.type=libero",
        "--env.task=" + suite,
        "--env.fps={}".format(fps),
        "--eval.batch_size={}".format(batch),
        "--eval.n_episodes={}".format(episodes),
        "--output_dir=" + str(out_dir),
        "--rename_map=" + RENAME_MAP,
    ]
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    dt = time.time() - t0

    info_path = out_dir / "eval_info.json"
    if not info_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        return {"fps": fps, "error": "eval_info.json 없음", "rc": proc.returncode,
                "seconds": round(dt, 1), "tail": tail}

    info = json.loads(info_path.read_text(encoding="utf-8"))
    successes, per_task = [], {}
    for entry in info.get("per_task", []):
        s = [bool(x) for x in entry["metrics"]["successes"]]
        successes.extend(s)
        per_task[int(entry.get("task_id"))] = {"success": sum(s), "n": len(s)}
    k, n = sum(successes), len(successes)
    lo, hi = wilson(k, n)
    return {"fps": fps, "num_steps": num_steps, "empty_cameras": empty_cameras,
            "success": k, "n_episodes": n,
            "success_rate": round(100.0 * k / n, 1) if n else None,
            "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
            "per_task": per_task, "seconds": round(dt, 1), "rc": proc.returncode}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--fps", default="20,10")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--empty-cameras", type=int, default=0,
                    help="기본 0 = 학습 조건. A/B에서 확정된 값을 쓴다")
    ap.add_argument("--out", default="results/ab_control_freq.json")
    ap.add_argument("--work", default="results/fps_runs")
    args = ap.parse_args()

    fpss = [int(x) for x in args.fps.split(",") if x.strip()]
    print("A/B: env.fps {} · empty_cameras={} · num_steps={} · {}".format(
        fpss, args.empty_cameras, args.num_steps, args.suite), flush=True)
    print()

    rows = []
    for f in fpss:
        print("--- env.fps={} 실행 중 ---".format(f), flush=True)
        r = run_arm(args.policy, args.suite, args.episodes, args.batch, args.num_steps,
                    args.empty_cameras, f, Path(args.work) / "fps{}".format(f))
        rows.append(r)
        if "error" in r:
            print("   실패: {} (rc={})".format(r["error"], r["rc"]))
            print("   ...{}".format(r["tail"][-500:]), flush=True)
        else:
            print("   성공률 {}% ({}/{})  Wilson95 [{}, {}] · {}초".format(
                r["success_rate"], r["success"], r["n_episodes"],
                r["wilson95"][0], r["wilson95"][1], r["seconds"]), flush=True)

    ok = [r for r in rows if "error" not in r]
    if len(ok) >= 2:
        base = next(r for r in ok if r["fps"] == fpss[0])
        print()
        print("{:>8} {:>10} {:>18}".format("env.fps", "성공률%", "Wilson95"))
        for r in ok:
            print("{:>8} {:>10} {:>18}".format(
                r["fps"], r["success_rate"],
                "[{}, {}]".format(r["wilson95"][0], r["wilson95"][1])))
        print()
        for r in ok:
            if r is base:
                continue
            overlap = not (r["wilson95"][1] < base["wilson95"][0]
                           or base["wilson95"][1] < r["wilson95"][0])
            print("  fps {} → {}: {:+.1f}%p  → {}".format(
                base["fps"], r["fps"], r["success_rate"] - base["success_rate"],
                "구간 겹침 — 차이 있다고 말하지 않음" if overlap
                else "**구간 분리 — 제어 주기가 성공률을 지배한다**"))

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "meta": {"policy": args.policy, "suite": args.suite,
                 "episodes_per_task": args.episodes, "batch": args.batch,
                 "num_steps": args.num_steps, "empty_cameras": args.empty_cameras,
                 "note": "env.fps 는 robosuite control_freq 로 그대로 들어간다(configs.py gym_kwargs)."},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("저장: {}".format(args.out))


if __name__ == "__main__":
    main()
