"""도메인 랜덤화(마찰·질량) 강건성 A/B — Sim2Real 이슈에 실측으로 답한다.

배경: 이 프로젝트는 지금까지 "정책이 LIBERO 표준 물리에서 몇 %를 성공하나"만 쟀다.
Sim2Real 갭을 실제로 다루려면 "물리가 조금만 흔들려도 성공률이 얼마나 깎이나"가 먼저다.
`tools/dr_sitecustomize.py`가 매 에피소드 reset()마다 마찰·질량을 균등분포로 스케일링한다
(hard_reset=True라 매번 XML 기본값에서 다시 스케일 — 누적 드리프트 없음).

같은 suite·task_ids·episodes·batch로 두 팔(DR off / DR on)을 돌리고, 결정론적 리셋 순서
(이 레포군이 이미 확인한 사실)를 이용해 같은 에피소드 슬롯끼리 짝짓는다 → McNemar 정확검정.

사용: python tools/dr_robustness.py --episodes 15 --suite libero_spatial
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from math import comb
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_denoise import wilson, eval_bin  # noqa: E402

RENAME_MAP = ('{"observation.images.image":"observation.images.camera1",'
              '"observation.images.image2":"observation.images.camera2"}')


def mcnemar_exact_p(b: int, c: int) -> float:
    """짝지음 불일치 쌍(b: A만 성공, c: B만 성공)에 대한 양측 정확검정.

    이 레포군이 num_steps 비교에서 이미 쓴 방법과 같다(RESULTS.md §2.5, p=0.018 사례).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return min(1.0, p)


def run_arm(policy: str, suite: str, episodes: int, batch: int, num_steps: int,
            dr_enable: bool, seed_base: int, out_dir: Path, log_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        eval_bin(),
        "--policy.path=" + policy,
        "--policy.device=cuda",
        "--policy.empty_cameras=0",
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
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = tools_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["VLA_DR_ENABLE"] = "1" if dr_enable else "0"
    env["VLA_DR_SEED_BASE"] = str(seed_base)
    env["VLA_DR_LOG"] = str(log_path)

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    dt = time.time() - t0

    info_path = out_dir / "eval_info.json"
    if not info_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        return {"dr": dr_enable, "error": "eval_info.json 없음", "rc": proc.returncode,
                "seconds": round(dt, 1), "tail": tail}

    info = json.loads(info_path.read_text(encoding="utf-8"))
    per_task_seq = {}
    all_seq = []
    for entry in info.get("per_task", []):
        s = [bool(x) for x in entry["metrics"]["successes"]]
        per_task_seq[int(entry.get("task_id"))] = s
        all_seq.append((int(entry.get("task_id")), s))
    k = sum(sum(s) for _, s in all_seq)
    n = sum(len(s) for _, s in all_seq)
    lo, hi = wilson(k, n)
    return {
        "dr": dr_enable, "success": k, "n_episodes": n,
        "success_rate": round(100.0 * k / n, 1) if n else None,
        "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
        "per_task_successes": {str(t): s for t, s in per_task_seq.items()},
        "seconds": round(dt, 1), "rc": proc.returncode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--episodes", type=int, default=15, help="태스크당 에피소드")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--friction-range", default="0.5,1.5")
    ap.add_argument("--mass-range", default="0.7,1.3")
    ap.add_argument("--seed-base", type=int, default=20260903)
    ap.add_argument("--out", default="results/dr_robustness.json")
    ap.add_argument("--work", default="results/dr_runs")
    args = ap.parse_args()

    os.environ["VLA_DR_FRICTION_RANGE"] = args.friction_range
    os.environ["VLA_DR_MASS_RANGE"] = args.mass_range

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    log_off = work / "dr_off.log"
    log_on = work / "dr_on.log"

    print("도메인 랜덤화 A/B: friction={} mass={} · suite={} · 태스크당 {}에피소드".format(
        args.friction_range, args.mass_range, args.suite, args.episodes), flush=True)
    print()

    arms = {}
    for dr in (False, True):
        tag = "on" if dr else "off"
        print("--- DR={} 실행 중 ---".format(tag), flush=True)
        r = run_arm(args.policy, args.suite, args.episodes, args.batch, args.num_steps,
                    dr, args.seed_base, work / "dr_{}".format(tag), log_on if dr else log_off)
        arms[tag] = r
        if "error" in r:
            print("   실패: {} (rc={})".format(r["error"], r["rc"]))
            print("   ...{}".format(r["tail"][-500:]), flush=True)
        else:
            print("   성공률 {}% ({}/{})  Wilson95 [{}, {}] · {}초".format(
                r["success_rate"], r["success"], r["n_episodes"],
                r["wilson95"][0], r["wilson95"][1], r["seconds"]), flush=True)

    result = {
        "meta": {"policy": args.policy, "suite": args.suite, "episodes_per_task": args.episodes,
                  "batch": args.batch, "num_steps": args.num_steps,
                  "friction_range": args.friction_range, "mass_range": args.mass_range,
                  "seed_base": args.seed_base},
        "arms": arms,
    }

    if "error" not in arms["off"] and "error" not in arms["on"]:
        off, on = arms["off"], arms["on"]
        print()
        print("=== 결과 ===")
        print("{:>10} {:>10} {:>18}".format("DR", "성공률%", "Wilson95"))
        for tag, r in (("off", off), ("on", on)):
            print("{:>10} {:>10} {:>18}".format(
                tag, r["success_rate"], "[{}, {}]".format(*r["wilson95"])))

        # 짝지음 — task_id별로 같은 인덱스(=같은 init_state_id·같은 랜덤화 시드 슬롯)를 매칭
        both_00, off1_on0, off0_on1, both_11 = 0, 0, 0, 0
        mismatches = []
        for t in sorted(set(off["per_task_successes"]) & set(on["per_task_successes"])):
            so = off["per_task_successes"][t]
            sn = on["per_task_successes"][t]
            if len(so) != len(sn):
                mismatches.append(t)
                continue
            for a, b in zip(so, sn):
                if a and b:
                    both_11 += 1
                elif a and not b:
                    off1_on0 += 1
                elif not a and b:
                    off0_on1 += 1
                else:
                    both_00 += 1
        if mismatches:
            print("⚠ 태스크 {} 은 두 팔의 에피소드 수가 달라 짝짓기에서 제외".format(mismatches))

        p = mcnemar_exact_p(off1_on0, off0_on1)
        n_pairs = both_11 + off1_on0 + off0_on1 + both_00
        print()
        print("짝지음 표(off\\on): 성공/성공={} 성공/실패={} 실패/성공={} 실패/실패={} (n={})".format(
            both_11, off1_on0, off0_on1, both_00, n_pairs))
        print("McNemar 정확검정 p={:.4f}  (불일치 {}쌍: off만성공 {} : on만성공 {})".format(
            p, off1_on0 + off0_on1, off1_on0, off0_on1))
        gap = on["success_rate"] - off["success_rate"]
        sig = "유의함(p<0.05)" if p < 0.05 else "유의하지 않음"
        print("DR on 대비 off: {:+.1f}%p → {}".format(gap, sig))

        result["mcnemar"] = {
            "both_success": both_11, "off_only": off1_on0, "on_only": off0_on1,
            "both_fail": both_00, "n_pairs": n_pairs, "p_value": round(p, 5),
            "gap_pp": round(gap, 1), "significant_p05": bool(p < 0.05),
        }

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("저장:", args.out)


if __name__ == "__main__":
    main()
