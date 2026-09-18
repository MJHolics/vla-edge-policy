#!/usr/bin/env bash
# 밤샘 측정 체인 — 2026-08-19
#
# 순서가 중요하다. 앞의 측정이 끝나기 전에 다음을 걸면 CPU/GPU가 겹쳐 지연 수치가 오염된다.
#   0) 지금 돌고 있는 A/B(ab_control_freq)가 끝날 때까지 기다린다
#   1) 지연 재측정 — empty_cameras=0(학습 조건). 기존 헤드라인은 1로 강제된 값이라 부풀어 있다
#   2) 표준 프로토콜 성공률 — 태스크당 50 init state(=suite당 500에피소드).
#      지금까지는 5/50만 써서 Wilson 폭이 ±13%p였다. n=500이면 ±4%p로 좁아져
#      "num_steps 10 vs 1"에 실제로 답할 수 있다.
set -u
cd /root/vla
export MUJOCO_GL=egl
export HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/root/vla-venv/bin/python
LOG=/root/vla/results/overnight.log
mkdir -p /root/vla/results

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 밤샘 체인 시작 ==="

while pgrep -f "ab_control_freq.py" >/dev/null 2>&1; do
  sleep 30
done
say "선행 A/B 종료 확인"

say "1/2 지연 재측정 (empty_cameras=0)"
$PY tools/bench_policy_latency.py --empty-cameras 0 \
    --out results/policy_latency_ec0.json >> "$LOG" 2>&1
say "1/2 완료 (rc=$?)"

say "2/2 표준 프로토콜 성공률 (태스크당 50에피소드 · num_steps 10,1)"
$PY tools/sweep_denoise.py --num-steps 10,1 --episodes 50 --batch 5 \
    --tasks all --empty-cameras 0 \
    --out results/full_protocol.json --work results/full_runs >> "$LOG" 2>&1
say "2/2 완료 (rc=$?)"

say "=== 밤샘 체인 종료 ==="
