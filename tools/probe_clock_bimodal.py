"""이봉 지연 분포가 GPU 클럭 램핑 때문인지 직접 검증한다 — **클럭 고정 없이**.

RESULTS.md §3.5 "남은 측정(미실행)"이 시킨 두 가지: ① clocks.sm·power.draw 동시 샘플링
② `nvidia-smi -lgc`로 클럭 고정 후 재현. **②는 이 WSL2 환경에서 불가능하다** —
`nvidia-smi -lgc 3105,3105`(지원 목록에 있는 값)를 시도해도 "Unable to set GPU locked
clocks ... Unknown Error"만 난다. WSL2의 GPU 가상화 계층이 클럭 잠금 NVML 호출을
패스스루하지 않는 것으로 보인다(쿼리는 되는데 설정만 막힘). `-rgc`도 같은 에러라
애초에 걸린 적도 없다 — 안전하게 되돌릴 것도 없다.

그래서 ①만으로 가설을 검증한다: 매 호출 **직후**(타이밍 구간 밖에서) clocks.sm·power.draw를
질의해 지연과 같은 줄에 남긴다. 이러면 "낮은 지연 다발 vs 높은 지연 다발"이 클럭 상태로
갈리는지 직접 볼 수 있다 — 램핑 가설이 맞다면 저지연 호출은 낮은 클럭 직후(방금 깨어남)가
아니라 **오히려 다음 호출들이 고클럭에 안착한 뒤**에 몰려야 한다(클럭이 오르면서 지연이 준다).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_policy_latency import build_policy_and_batch  # noqa: E402


def _query_clock() -> tuple[float, float]:
    """nvidia-smi 1회 질의. 타이밍 구간 밖에서만 부른다(오버헤드가 지연 측정을 오염시키지 않게)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    sm_str, pw_str = out.split(",")
    return float(sm_str), float(pw_str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-steps", type=int, default=1,
                    help="가장 빠르고 고정비가 도드라지는 설정(기본 1)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threshold-ms", type=float, default=75.0,
                    help="RESULTS.md §3.5의 저/고 지연 모드 경계")
    ap.add_argument("--out", default="results/clock_bimodal_probe.json")
    args = ap.parse_args()

    import torch

    print("=== 정책 로드: {} ===".format(args.policy), flush=True)
    policy, batch, chunk_shape = build_policy_and_batch(
        args.policy, args.suite, args.task_id, args.device, empty_cameras=0)
    policy.config.num_steps = args.num_steps

    def call():
        with torch.no_grad():
            policy.predict_action_chunk(batch)

    print("워밍업 {}회...".format(args.warmup), flush=True)
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()

    print("측정 {}회 (호출 직후 클럭 질의)...".format(args.n), flush=True)
    rows = []
    for i in range(args.n):
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        sm_mhz, power_w = _query_clock()
        rows.append({"i": i, "latency_ms": round(lat_ms, 3),
                     "clocks_sm_mhz": sm_mhz, "power_w": power_w})
        if (i + 1) % 20 == 0:
            print("  {}/{}  최근 지연 {:.1f}ms  클럭 {:.0f}MHz  전력 {:.1f}W".format(
                i + 1, args.n, lat_ms, sm_mhz, power_w), flush=True)

    lo = [r for r in rows if r["latency_ms"] < args.threshold_ms]
    hi = [r for r in rows if r["latency_ms"] >= args.threshold_ms]

    def summarize(bucket, label):
        if not bucket:
            print("  {}: 표본 0개".format(label))
            return None
        lat = [r["latency_ms"] for r in bucket]
        sm = [r["clocks_sm_mhz"] for r in bucket]
        pw = [r["power_w"] for r in bucket]
        s = {
            "label": label, "n": len(bucket),
            "latency_ms_median": round(statistics.median(lat), 1),
            "clocks_sm_mhz_mean": round(statistics.mean(sm), 1),
            "clocks_sm_mhz_median": round(statistics.median(sm), 1),
            "power_w_mean": round(statistics.mean(pw), 1),
        }
        print("  {}: n={} 지연중앙값={:.1f}ms 클럭평균={:.0f}MHz(중앙값{:.0f}) 전력평균={:.1f}W".format(
            label, s["n"], s["latency_ms_median"], s["clocks_sm_mhz_mean"],
            s["clocks_sm_mhz_median"], s["power_w_mean"]))
        return s

    print()
    print("=== 버킷별 클럭·전력 (경계 {:.0f}ms) ===".format(args.threshold_ms))
    lo_summary = summarize(lo, "저지연(<{:.0f}ms)".format(args.threshold_ms))
    hi_summary = summarize(hi, "고지연(>={:.0f}ms)".format(args.threshold_ms))

    verdict = None
    if lo_summary and hi_summary:
        gap = hi_summary["clocks_sm_mhz_mean"] - lo_summary["clocks_sm_mhz_mean"]
        print()
        if gap < -200:
            verdict = ("고지연 버킷의 클럭이 저지연 버킷보다 {:.0f}MHz 낮다 — "
                       "클럭 램핑 가설과 일치(클럭이 낮을 때 느리다).".format(-gap))
        elif gap > 200:
            verdict = ("고지연 버킷의 클럭이 저지연 버킷보다 {:.0f}MHz *높다* — "
                       "클럭 램핑 가설과 반대. 다른 원인(스케줄링·메모리 대역폭 경합 등)을 봐야 한다.".format(gap))
        else:
            verdict = ("두 버킷의 클럭 차이가 {:.0f}MHz로 작다 — "
                       "이 데이터에서는 클럭 램핑이 이봉 분포의 1차 설명이 아닐 수 있다.".format(gap))
        print("판정: " + verdict)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "policy": args.policy, "num_steps": args.num_steps, "n": args.n,
                "warmup": args.warmup, "threshold_ms": args.threshold_ms,
                "note": "nvidia-smi -lgc가 이 WSL2 환경에서 실패해(Unknown Error, "
                        "지원 목록의 값 3105MHz로도 재현) 클럭 고정 대신 호출별 "
                        "클럭·전력을 직접 상관 분석했다.",
                "verdict": verdict,
            },
            "rows": rows,
            "lo_bucket": lo_summary, "hi_bucket": hi_summary,
        }, f, ensure_ascii=False, indent=2)
    print("저장: {}".format(args.out))


if __name__ == "__main__":
    main()
