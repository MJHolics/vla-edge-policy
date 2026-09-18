#!/usr/bin/env bash
# 측정 체인 4차 — 2026-08-20 낮 기동
#
# 3차(overnight4.sh)도 1단 중간에 죽었다. 이번엔 절전 설정이 원인이 아니다:
#   - AC 대기 시간은 0(끔)으로 이미 바꿔 뒀고, Windows는 재부팅되지 않았다(부팅 08-12).
#   - 그런데 시스템 이벤트에 **01:38:49 ID 187 "user-mode process가 SetSuspendState 호출"**,
#     이어서 42(절전 진입) → 107(복귀)가 찍혀 있다. 즉 **유휴 절전이 아니라 명시적 절전**이었다.
#     powercfg 타임아웃은 명시적 절전을 막지 못한다.
#   - WSL2 VM은 이 4초짜리 절전에도 같이 죽었다(다음 날 uptime 0분).
#     → **잠깐의 서스펜드도 이 체인을 전부 날린다.**
#
# 이번 변경: **1단을 num_steps별로 쪼갠다.**
#   sweep_denoise.py는 json을 **마지막에 한 번** 쓴다(rows 전부 모은 뒤). 그래서
#   ns10을 80분 돌리고 ns1 도중에 죽으면 **ns10까지 통째로 사라진다** — 어제 정확히 그랬다
#   (01:34까지 돌던 ns10 결과가 0바이트로 남았다). 호출을 둘로 나누면 노출 시간이 절반이 되고,
#   먼저 끝난 쪽은 디스크에 남는다.
#
# 이미 있는 결과 파일은 건너뛴다(또 죽어도 이어서).
set -u
cd /root/vla
export MUJOCO_GL=egl
export HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/root/vla-venv/bin/python
LOG=/root/vla/results/overnight.log

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
have() { [ -s "$1" ]; }

say "=== 4차 체인 시작 (1단 분할: ns10 / ns1 → 재현성 → 지연ABBA) ==="

# ---------------------------------------------------------------- 1/4, 2/4
# 표준 프로토콜: 태스크당 50 init state = n=500. Wilson 95% 폭 ±13%p → ±4%p.
# num_steps를 따로 호출해 각각의 json을 남긴다.
for ns in 10 1; do
  out="results/full_protocol_ns${ns}.json"
  if have "$out"; then
    say "표준 프로토콜 ns=${ns} — 이미 있음, 건너뜀"
    continue
  fi
  say "표준 프로토콜 ns=${ns} · n=500 — 약 90분"
  $PY tools/sweep_denoise.py --num-steps "$ns" --episodes 50 --batch 5 \
      --tasks all --empty-cameras 0 \
      --out "$out" --work "results/full_runs/ns${ns}" >> "$LOG" 2>&1
  say "표준 프로토콜 ns=${ns} 완료 (rc=$?)"
done

# ---------------------------------------------------------------- 3/4
# 재현성: 같은 설정 3회. 같은 설정 두 실행이 64.0% vs 60.0%로 갈렸었다.
# 노이즈 바닥을 모르면 어떤 A/B도 해석할 수 없다.
say "재현성 3회 반복 (같은 설정 · 태스크당 5에피소드)"
for i in 1 2 3; do
  if have "results/repeat_run${i}.json"; then
    say "   반복 $i/3 — 이미 있음, 건너뜀"
    continue
  fi
  say "   반복 $i/3"
  $PY tools/sweep_denoise.py --num-steps 10 --episodes 5 --batch 5 --tasks all \
      --empty-cameras 0 \
      --out "results/repeat_run${i}.json" --work "results/repeat_runs/r${i}" >> "$LOG" 2>&1
done
say "재현성 완료"

# ---------------------------------------------------------------- 4/4
# 지연 짝지음 ABBA. 3차의 ABAB 결과는 머신 드리프트와 교란돼 해석 불가였다.
# 파일명에 순서 위치(p1..p4)를 박아 둔다 — 시각순 복원이 사후에도 가능해야 한다.
say "지연 짝지음 ABBA (ec0 → ec1 → ec1 → ec0)"
i=0
for ec in 0 1 1 0; do
  i=$((i+1))
  out="results/latency_abba_p${i}_ec${ec}.json"
  if have "$out"; then say "   p$i ec=$ec — 이미 있음, 건너뜀"; continue; fi
  say "   위치 p$i  empty_cameras=$ec"
  $PY tools/bench_policy_latency.py --empty-cameras $ec --repeat 2 \
      --out "$out" >> "$LOG" 2>&1
done
say "지연 완료"

say "=== 4차 체인 종료 ==="
touch results/_CHAIN_DONE
