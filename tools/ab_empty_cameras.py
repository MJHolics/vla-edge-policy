"""빈 카메라 패딩 A/B — 우리 평가 하네스가 학습 조건을 재현하고 있는지 검증한다.

배경(2026-08-18):
  성공률 스윕에서 60%가 나왔는데 공개 수치는 90%대다. 1순위 가설은
  "정책이 카메라 3대로 학습됐는데 평가에서 2대만 준다"였고, **가설은 틀렸다.**

  학습 데이터셋 `lerobot/libero`의 카메라는 **2대**(image, image2)뿐이다.
  정책 config가 camera1/2/3을 선언하지만 train_config의 `empty_cameras=0`이므로
  `modeling_smolvla.prepare_images`는 없는 camera3을 **패딩하지 않고 건너뛴다**
  (`if num_empty_cameras >= self.config.empty_cameras: break`).
  → **학습은 이미지 2장으로 이뤄졌다.**

  그런데 우리 평가는 `--policy.empty_cameras=1`로 돌렸다. 즉 학습에서 존재한 적 없는
  **-1 패딩 이미지 1장 + mask 0**을 매 스텝 밀어 넣고 있었다. 학습/평가 불일치다.

이 스크립트는 그 한 손잡이만 바꿔 같은 조건으로 두 번 돌린다. 태스크·에피소드·시드가
같으므로 태스크별 짝지음 비교가 된다. 구간이 겹치면 차이가 있다고 말하지 않는다.

사용: python tools/ab_empty_cameras.py --episodes 5 --num-steps 10
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
    """venv의 lerobot-eval 을 절대경로로 잡는다.

    `venv/bin/python tools/...` 로 부르면 venv의 bin/ 이 PATH에 없어
    `FileNotFoundError: lerobot-eval` 로 죽는다(활성화했을 때만 우연히 동작했다).
    """
    cand = Path(sys.executable).with_name("lerobot-eval")
    return str(cand) if cand.exists() else "lerobot-eval"


def run_arm(policy: str, suite: str, episodes: int, batch: int, num_steps: int,
            empty_cameras: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        eval_bin(),
        "--policy.path=" + policy,
        "--policy.device=cuda",
        "--policy.empty_cameras={}".format(empty_cameras),
        "--policy.num_steps={}".format(num_steps),
        "--env.type=libero",
        "--env.task=" + suite,
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
        tail = (proc.stderr or proc.stdout or "")[-1200:]
        return {"empty_cameras": empty_cameras, "error": "eval_info.json 없음",
                "rc": proc.returncode, "seconds": round(dt, 1), "tail": tail}

    info = json.loads(info_path.read_text(encoding="utf-8"))
    successes, per_task = [], {}
    for entry in info.get("per_task", []):
        s = [bool(x) for x in entry["metrics"]["successes"]]
        successes.extend(s)
        per_task[int(entry.get("task_id"))] = {"success": sum(s), "n": len(s)}
    k, n = sum(successes), len(successes)
    lo, hi = wilson(k, n)
    return {
        "empty_cameras": empty_cameras, "num_steps": num_steps,
        "success": k, "n_episodes": n,
        "success_rate": round(100.0 * k / n, 1) if n else None,
        "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
        "per_task": per_task, "seconds": round(dt, 1), "rc": proc.returncode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--episodes", type=int, default=5, help="태스크당 에피소드")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=10, help="학습 기본값과 같게 둔다")
    ap.add_argument("--out", default="results/ab_empty_cameras.json")
    ap.add_argument("--work", default="results/ab_runs")
    args = ap.parse_args()

    print("A/B: empty_cameras 0(학습 조건) vs 1(현행 하네스) · num_steps={} · {}"
          .format(args.num_steps, args.suite), flush=True)
    print()

    arms = {}
    for ec in (0, 1):
        print("--- empty_cameras={} 실행 중 ---".format(ec), flush=True)
        r = run_arm(args.policy, args.suite, args.episodes, args.batch,
                    args.num_steps, ec, Path(args.work) / "ec{}".format(ec))
        arms[ec] = r
        if "error" in r:
            print("   실패: {} (rc={})".format(r["error"], r["rc"]))
            print("   ...{}".format(r["tail"][-400:]), flush=True)
        else:
            print("   성공률 {}% ({}/{})  Wilson95 [{}, {}] · {}초".format(
                r["success_rate"], r["success"], r["n_episodes"],
                r["wilson95"][0], r["wilson95"][1], r["seconds"]), flush=True)

    ok = [arms[e] for e in (0, 1) if "error" not in arms[e]]
    if len(ok) == 2:
        a, b = arms[0], arms[1]
        print()
        print("=== 결과 ===")
        print("{:>16} {:>10} {:>18}".format("empty_cameras", "성공률%", "Wilson95"))
        for r in (a, b):
            print("{:>16} {:>10} {:>18}".format(
                r["empty_cameras"], r["success_rate"],
                "[{}, {}]".format(r["wilson95"][0], r["wilson95"][1])))
        overlap = not (a["wilson95"][1] < b["wilson95"][0]
                       or b["wilson95"][1] < a["wilson95"][0])
        gap = a["success_rate"] - b["success_rate"]
        print()
        print("현행(1) 대비 학습 조건(0): {:+.1f}%p → {}".format(
            gap, "구간 겹침 — 차이 있다고 말하지 않음" if overlap
                 else "**구간이 분리됐다 — 하네스 결함이 성공률을 깎고 있었다**"))

        print()
        print("태스크별 짝지음 (0 vs 1):")
        for t in sorted(set(a["per_task"]) | set(b["per_task"])):
            pa = a["per_task"].get(t, {})
            pb = b["per_task"].get(t, {})
            print("   task {:2d}: {}/{}  vs  {}/{}".format(
                t, pa.get("success"), pa.get("n"), pb.get("success"), pb.get("n")))

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "meta": {"policy": args.policy, "suite": args.suite,
                 "episodes_per_task": args.episodes, "batch": args.batch,
                 "num_steps": args.num_steps,
                 "note": "empty_cameras=0 이 학습 조건(train_config.json)과 같다."},
        "arms": {str(k): v for k, v in arms.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("저장: {}".format(args.out))


if __name__ == "__main__":
    main()
