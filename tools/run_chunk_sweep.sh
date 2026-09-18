#!/usr/bin/env bash
# 액션 청킹(n_action_steps) 스윕 백그라운드 실행기.
# PLAN.md P3의 마지막 손잡이 — num_steps=10 고정, n_action_steps={1,5,10,25,50} 스윕.
set -u
cd /root/vla
export MUJOCO_GL=egl
export HF_HUB_DISABLE_PROGRESS_BARS=1
LOG=/root/vla/results/chunk_sweep.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
say "=== 청킹 스윕 시작 (n_action_steps=1,5,10,25,50 · num_steps=10 · 태스크당 15에피소드 = n150/값) ==="
/root/vla-venv/bin/python tools/sweep_chunk.py \
  --n-action-steps 1,5,10,25,50 --episodes 15 --tasks all --batch 5 \
  --out results/chunk_sweep.json --work results/chunk_runs \
  >> "$LOG" 2>&1
say "=== 청킹 스윕 종료 (rc=$?) ==="
touch /root/vla/results/_CHUNK_SWEEP_DONE
