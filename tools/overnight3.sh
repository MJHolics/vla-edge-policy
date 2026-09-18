#!/usr/bin/env bash
# 밤샘 체인 재실행 — 2026-08-19 00:55
#
# 1차 체인은 2·3단이 `FileNotFoundError: lerobot-eval` 로 즉사했다.
# 원인: `venv/bin/python tools/x.py` 로 부르면 venv 의 bin/ 이 PATH 에 없다.
# ab_empty_cameras.py 에서 이미 겪고 고쳤는데 **sweep_denoise.py 에는 전파하지 않았다.**
# (같은 결함을 같은 밤에 두 번 물렸다 — 고칠 땐 같은 패턴을 쓰는 파일을 grep 해야 한다.)
#
# 순서는 가치와 소요시간을 같이 본다:
#   1) 지연 짝지음 (ec0/ec1 교차 2회, ~20분) — 1차 재측정이 255→334ms 로 **되레 느려졌다.**
#      카메라를 줄였는데 느려질 수는 없으므로 머신 상태 드리프트를 의심한다.
#      같은 세션에서 교차로 재면 드리프트와 카메라 효과가 분리된다.
#   2) 표준 프로토콜 n=500 (~200분) — 이 프로젝트 최대 가치. Wilson ±13%p → ±4%p
#   3) 재현성 3회 반복 (~30분) — 같은 설정 재실행 흩어짐의 바닥
set -u
cd /root/vla
export MUJOCO_GL=egl
export HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/root/vla-venv/bin/python
LOG=/root/vla/results/overnight.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 재실행 체인 시작 ==="

say "1/3 지연 짝지음 (ec0 → ec1 → ec0 → ec1)"
for round in 1 2; do
  for ec in 0 1; do
    say "   round=$round empty_cameras=$ec"
    $PY tools/bench_policy_latency.py --empty-cameras $ec --repeat 2 \
        --out "results/latency_r${round}_ec${ec}.json" >> "$LOG" 2>&1
  done
done
say "1/3 완료"

say "2/3 표준 프로토콜 (태스크당 50에피소드 = n=500 · num_steps 10,1)"
$PY tools/sweep_denoise.py --num-steps 10,1 --episodes 50 --batch 5 \
    --tasks all --empty-cameras 0 \
    --out results/full_protocol.json --work results/full_runs >> "$LOG" 2>&1
say "2/3 완료 (rc=$?)"

say "3/3 재현성 3회 반복 (같은 설정 · 태스크당 5에피소드)"
for i in 1 2 3; do
  say "   반복 $i/3"
  $PY tools/sweep_denoise.py --num-steps 10 --episodes 5 --batch 5 --tasks all \
      --empty-cameras 0 \
      --out "results/repeat_run${i}.json" --work "results/repeat_runs/r${i}" >> "$LOG" 2>&1
done
say "3/3 완료"

say "=== 재실행 체인 종료 ==="
