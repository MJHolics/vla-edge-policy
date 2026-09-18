"""이봉 지연의 원인을 CUDA 제출 경로까지 좁힌다 — **NVTX로 호출 경계를 표시**.

`probe_clock_bimodal.py`(클럭 상관, 지지 안 됨) → `nsys profile`(cudaLaunchKernel/ioctl이
중앙값 대비 최댓값 1,000~4,000배인 두터운 꼬리, 하지만 이건 모델 로드까지 섞인 **전체 구간 집계**였다)
다음 단계. 이 스크립트는 각 추론 호출을 NVTX 레인지로 감싸, nsys 트레이스에서 **호출별로** 정확히
그 구간 안의 cudaLaunchKernel·ioctl 시간을 뽑아 지연과 대조할 수 있게 한다.

사용(nsys로 감싸서 실행해야 의미가 있다):
  nsys profile -t cuda,osrt,nvtx -o results/nsys_launch --force-overwrite=true \\
      /root/vla-venv/bin/python tools/probe_launch_overhead.py --n 40 --warmup 15
  (분석은 tools/analyze_nsys_launch.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_policy_latency import build_policy_and_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-steps", type=int, default=1)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--out", default="results/launch_overhead_probe.json")
    args = ap.parse_args()

    import torch
    import torch.cuda.nvtx as nvtx

    print("=== 정책 로드: {} ===".format(args.policy), flush=True)
    policy, batch, chunk_shape = build_policy_and_batch(
        args.policy, args.suite, args.task_id, args.device, empty_cameras=0)
    policy.config.num_steps = args.num_steps

    def call():
        with torch.no_grad():
            policy.predict_action_chunk(batch)

    print("워밍업 {}회 (NVTX: warmup_i)...".format(args.warmup), flush=True)
    for i in range(args.warmup):
        nvtx.range_push("warmup_{}".format(i))
        call()
        torch.cuda.synchronize()
        nvtx.range_pop()

    print("측정 {}회 (NVTX: call_i)...".format(args.n), flush=True)
    rows = []
    for i in range(args.n):
        nvtx.range_push("call_{}".format(i))
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        nvtx.range_pop()
        rows.append({"i": i, "latency_ms": round(lat_ms, 3)})
        if (i + 1) % 20 == 0:
            print("  {}/{}  최근 지연 {:.1f}ms".format(i + 1, args.n, lat_ms), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"policy": args.policy, "num_steps": args.num_steps,
                            "n": args.n, "warmup": args.warmup,
                            "note": "NVTX call_i/warmup_i 레인지로 감쌈. nsys -t cuda,osrt,nvtx로 "
                                    "감싸 실행해야 의미 있다."},
                  "rows": rows}, f, ensure_ascii=False, indent=2)
    print("저장: {}".format(args.out))


if __name__ == "__main__":
    main()
