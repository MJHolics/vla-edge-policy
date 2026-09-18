#!/usr/bin/env bash
# 밤샘 체인 3차 재실행 — 2026-08-20
#
# 2차 체인(overnight3.sh)은 1/3만 끝내고 죽었다. 로그에 에러가 없다 —
# 프로세스가 예외로 죽은 게 아니라 **WSL VM 자체가 내려갔다**(재부팅 후 uptime 0분).
# 원인은 Windows 절전: AC 대기 시간이 30분(0x708)이었다. 세션을 닫고 자리를 뜨면
# 30분 뒤 PC가 자고, WSL2 VM이 같이 죽는다. → 절전을 끄고(standby-timeout-ac 0) 다시 건다.
#
# 이번 변경 2가지:
#   (1) **순서를 가치순으로 바꿨다.** 매번 죽는 자리가 2단이었는데, 거기가 제일 비싸고
#       제일 값진 측정이다. 앞으로 뺀다. 싸고 짧은 건 뒤로 간다.
#   (2) **지연 벤치를 ABBA로 바꿨다.** 2차 체인의 ABAB(ec0,ec1,ec0,ec1)는 ec1이 항상
#       뒤에 오는 순서다. 결과를 보면 p50이 실행 시각순으로 단조 증가했고
#       (ns=10: 286.8 → 315.2 → 321.7 → 357.3ms), 같은 설정 재실행끼리도 +12~46% 벌어졌다.
#       드리프트가 시간축에 단조라면 ABAB에서 "뒤쪽 조건"은 무조건 손해를 본다 —
#       카메라 효과와 드리프트가 분리되지 않는다. ABBA면 두 조건의 평균 시각이 같아진다.
#
# 이미 있는 결과 파일은 건너뛴다(중간에 또 죽으면 이어서 돌리기 위함).
set -u
cd /root/vla
export MUJOCO_GL=egl
export HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/root/vla-venv/bin/python
LOG=/root/vla/results/overnight.log

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
have() { [ -s "$1" ]; }

say "=== 3차 체인 시작 (순서: n=500 → 재현성 → 지연ABBA) ==="

# ---------------------------------------------------------------- 1/3
# 표준 프로토콜: 태스크당 50 init state = n=500. Wilson 95% 폭 ±13%p → ±4%p.
# 이게 있어야 "num_steps 10 vs 1"에 처음으로 답할 수 있다.
if have results/full_protocol.json; then
  say "1/3 표준 프로토콜 — 이미 있음, 건너뜀"
else
  say "1/3 표준 프로토콜 n=500 (num_steps 10,1) — 약 200분"
  $PY tools/sweep_denoise.py --num-steps 10,1 --episodes 50 --batch 5 \
      --tasks all --empty-cameras 0 \
      --out results/full_protocol.json --work results/full_runs >> "$LOG" 2>&1
  say "1/3 완료 (rc=$?)"
fi

# ---------------------------------------------------------------- 2/3
# 재현성: 같은 설정 3회. 같은 설정 두 실행이 64.0% vs 60.0%로 갈렸었다.
# 노이즈 바닥을 모르면 어떤 A/B도 해석할 수 없다.
say "2/3 재현성 3회 반복 (같은 설정 · 태스크당 5에피소드)"
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
say "2/3 완료"

# ---------------------------------------------------------------- 3/3
# 지연 짝지음 ABBA. 2차의 ABAB 결과는 드리프트와 교란돼 해석 불가였다.
# 파일명에 순서 위치(p1..p4)를 박아 둔다 — 시각순 복원이 사후에도 가능해야 한다.
say "3/3 지연 짝지음 ABBA (ec0 → ec1 → ec1 → ec0)"
i=0
for ec in 0 1 1 0; do
  i=$((i+1))
  out="results/latency_abba_p${i}_ec${ec}.json"
  if have "$out"; then say "   p$i ec=$ec — 이미 있음, 건너뜀"; continue; fi
  say "   위치 p$i  empty_cameras=$ec"
  $PY tools/bench_policy_latency.py --empty-cameras $ec --repeat 2 \
      --out "$out" >> "$LOG" 2>&1
done
say "3/3 완료"

say "=== 3차 체인 종료 ==="
touch results/_CHAIN_DONE
