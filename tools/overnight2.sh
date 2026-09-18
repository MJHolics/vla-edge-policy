#!/usr/bin/env bash
# 밤샘 체인 2단 — 재현성 바닥 측정 (2026-08-19)
#
# 왜 필요한가: **같은 설정으로 두 번 돌렸는데 64.0% vs 60.0%가 나왔다**
#   (ab_empty_cameras 의 ec0 팔 · ab_control_freq 의 fps20 팔 — 정책·suite·에피소드·
#    num_steps·empty_cameras 가 전부 같다).
#   즉 우리가 비교하던 효과(+4.0%p)와 **재실행 노이즈가 같은 크기**다.
#   그 바닥을 모르면 어떤 A/B도 해석할 수 없으므로, 같은 설정을 3회 반복해 흩어짐을 잰다.
#
# lerobot-eval 은 seed 기본 1000 으로 set_seed 를 부르지만 env 는 start_seed=None 이면
# 수동 시드를 안 준다(lerobot_eval.py:528). GPU 비결정성까지 섞이므로 **추정하지 말고 잰다.**
set -u
cd /root/vla
export MUJOCO_GL=egl
export HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/root/vla-venv/bin/python
LOG=/root/vla/results/overnight.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

while pgrep -f "overnight.sh" >/dev/null 2>&1; do sleep 60; done
say "=== 2단: 재현성 3회 반복 (같은 설정) 시작 ==="

for i in 1 2 3; do
  say "반복 $i/3"
  $PY tools/sweep_denoise.py --num-steps 10 --episodes 5 --batch 5 --tasks all \
      --empty-cameras 0 \
      --out "results/repeat_run${i}.json" --work "results/repeat_runs/r${i}" >> "$LOG" 2>&1
done

say "=== 2단 종료 ==="
grep -h '"success_rate"' /root/vla/results/repeat_run*.json | tee -a "$LOG"
