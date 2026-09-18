"""도메인 랜덤화(시각: geom 색상·조명 밝기) 강건성 A/B — §6 "시각은 안 건드렸다"를 닫는다.

배경: `dr_robustness.py`(2026-09-03)는 물리 파라미터(마찰·질량)만 흔들었고, RESULTS.md §6은
"시각(조명·색·카메라)은 건드리지 않았다"를 명시적 한계로 남겼다. 이 스크립트는 같은 방법론
(sitecustomize 몽키패치 + 짝지음 McNemar)을 시각 축으로 그대로 옮긴다 — geom_rgba(물체 색상)와
light_diffuse(조명 밝기)를 매 에피소드 리셋마다 균등분포로 스케일링한다. 형상·카메라 위치·질감은
건드리지 않는다.

사용: python tools/dr_visual_robustness.py --episodes 15 --suite libero_spatial
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dr_robustness import mcnemar_exact_p, run_arm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--episodes", type=int, default=15, help="태스크당 에피소드")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--rgba-range", default="0.6,1.4")
    ap.add_argument("--light-range", default="0.5,1.5")
    ap.add_argument("--seed-base", type=int, default=20260909)
    ap.add_argument("--out", default="results/dr_visual_robustness.json")
    ap.add_argument("--work", default="results/dr_visual_runs")
    args = ap.parse_args()

    os.environ["VLA_DR_RGBA_RANGE"] = args.rgba_range
    os.environ["VLA_DR_LIGHT_RANGE"] = args.light_range

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    log_off = work / "visual_off.log"
    log_on = work / "visual_on.log"

    print("시각 도메인 랜덤화 A/B: rgba={} light={} · suite={} · 태스크당 {}에피소드".format(
        args.rgba_range, args.light_range, args.suite, args.episodes), flush=True)
    print()

    arms = {}
    for dr in (False, True):
        tag = "on" if dr else "off"
        print("--- VISUAL_DR={} 실행 중 ---".format(tag), flush=True)
        env_flag = os.environ.copy()
        os.environ["VLA_DR_ENABLE"] = "0"
        os.environ["VLA_DR_VISUAL_ENABLE"] = "1" if dr else "0"
        r = run_arm(args.policy, args.suite, args.episodes, args.batch, args.num_steps,
                    False, args.seed_base, work / "visual_{}".format(tag), log_on if dr else log_off)
        os.environ.clear()
        os.environ.update(env_flag)
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
                  "rgba_range": args.rgba_range, "light_range": args.light_range,
                  "seed_base": args.seed_base},
        "arms": arms,
    }

    if "error" not in arms["off"] and "error" not in arms["on"]:
        off, on = arms["off"], arms["on"]
        print()
        print("=== 결과 ===")
        print("{:>10} {:>10} {:>18}".format("VISUAL_DR", "성공률%", "Wilson95"))
        for tag, r in (("off", off), ("on", on)):
            print("{:>10} {:>10} {:>18}".format(
                tag, r["success_rate"], "[{}, {}]".format(*r["wilson95"])))

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
        print("VISUAL_DR on 대비 off: {:+.1f}%p → {}".format(gap, sig))

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
