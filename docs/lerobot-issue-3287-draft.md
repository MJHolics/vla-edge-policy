# lerobot issue #3287 답변 — **초안 (미게시)**

> 대상: `huggingface/lerobot` issue **#3287** — *Inquiry about Training Configurations
> (Batch Size, LR, Commit Hash) for Replicating SmolVLA on LIBERO Benchmark* (Yoonkyo, 2026-04-05, **답변 0**)
> 초안 작성 2026-08-20 · **게시하지 않았다.** 외부 공개 행동이라 본인이 직접 올린다.
> 12개월 설계 트랙 C·Q1 「lerobot #3287 답변 (9월 첫 주)」 — "가장 값싼 외부 검증"

---

## 왜 우리가 답할 수 있나 — 그리고 **무엇은 답 못 하나**

| 질문자가 물은 것 | 우리가 가진 것 | 답 가능? |
|---|---|---|
| ① reported numbers가 frozen-VLM+expert-only 레시피인가 | — | ❌ 학습을 안 했다 |
| ② 40태스크 멀티태스크 단일 정책인가, suite별 정책인가 | **공개 체크포인트의 `train_config.json`을 직접 열어 봤다** | ✅ **부분 답변** |
| ③ 정확한 재현 커맨드·LR·batch·GPU 수·**평가 세부**·commit hash | 학습 쪽은 ❌. **평가 쪽은 직접 재고 하네스 결함 2건을 찾았다** | ✅ **평가 세부만** |

> **선을 지킨다.** 우리는 **재학습을 하지 않았다.** 공개 체크포인트 `lerobot/smolvla_libero`를
> 평가만 했다. 그러니 "당신 학습 설정이 틀렸다"는 말은 하지 않는다.
> 대신 **"평가 쪽에도 격차의 후보가 있다"**는 독립 데이터 포인트를 준다.
> 질문자는 자기 학습 결과가 83.0%인데 논문이 90%라 학습을 의심하고 있다 —
> **공식 체크포인트가 우리 하네스에서 더 낮게 나온다면, 의심 범위가 학습 밖으로 넓어진다.**

---

## 게시할 내용 (영문 초안)

> ✅ **2026-08-20 n=500 완료 — 자리표시자를 실측으로 채웠다.**
> ns=10 **57.4% (287/500)** · ns=1 **52.2% (261/500)** · McNemar 짝지음 p=0.018.

---

Not a training answer — I haven't retrained. But I evaluated the **released**
`lerobot/smolvla_libero` checkpoint on LIBERO-Spatial and found a few things that bear on
Q2 and Q3, in case they narrow the search.

**Setup.** `lerobot` **0.6.1** (pip), torch **2.11.0+cu130**, Python 3.12.3, RTX 4080 SUPER 16GB,
WSL2 Ubuntu 24.04, `MUJOCO_GL=egl`.
Policy loaded at bf16 (450.0M params, 927MB VRAM). Standard protocol: all 10 spatial tasks
× 50 init states = **500 episodes** per configuration.

**1. The released checkpoint is a single multi-task model, not per-suite.**
Its `train_config.json` lists one dataset — `lerobot/libero` (1,693 episodes, all four
suites merged). So at least for the *released* checkpoint the answer to your Q2 is
"one multi-task policy", not separate policies per suite. Worth confirming whether the
paper's table used the same artifact.

**2. Evaluating that checkpoint, I get **57.4% (287/500)** on LIBERO-Spatial (n=500), below your 83.0%.**
If a third party evaluating the released weights lands below your from-scratch reproduction,
the remaining gap is probably not all on the training side.

**3. Two evaluation-side details that changed my numbers.**

- **`policy.empty_cameras`.** The eval path let me run with padded (all −1) image slots
  the policy never saw in training. The training dataset `lerobot/libero` has **2** cameras,
  even though the policy config declares three (`camera1/2/3`) — `pad_vector` pads whatever
  arrives, so the declared count is never used. Setting `empty_cameras=0` matches training.
  (A/B at n=50: 64% vs 60%. Since the eval is deterministic that 4pp is not run noise, but
  only 2 episodes flipped — McNemar exact p = 0.5, so it is undetermined rather than
  "no effect". Either way it is a silent train/eval mismatch.)
- **`env.fps` is load-bearing and inconsistent across three files.** The training dataset's
  `meta/info.json` says **10.0**, the env stanza in `train_config.json` says **30**, and
  eval `LiberoEnv.fps` is **20** (passed straight to robosuite as `control_freq`,
  `envs/configs.py:421`). Running eval at 10 instead of 20 gives **0.0% (0/50)** against 60.0% (30/50) at 20 —
  disjoint Wilson intervals, and this one is large enough to read at n=50. Anyone comparing numbers
  across setups should pin this.

**4. Small-episode runs are biased, not just noisy.** `init_state_id = episode_index`
(`envs/libero.py:175`), so `--episodes 5` evaluates the *first 5* of the 50 init states —
not a random sample. Taking the same first-5 subset out of my n=500 run gives **66.0%**
against **57.4%** for the full 500: **+8.6pp optimistic**, concentrated in a few tasks
(task 2: 5/5 vs 31/50). So a 50-episode number is not just a wide interval around the right
value, its center is off. Worth stating n whenever a success rate is quoted here.

Also, in case it saves someone a run: this eval is **deterministic**. I ran the identical
configuration three times and got byte-identical per-episode success vectors (64.0% each).
All of the uncertainty in a success rate here comes from *which init states you sampled*,
none from run-to-run variation. (Re-confirmed independently on 2026-09-10, three weeks later,
same machine: a fresh from-scratch rerun of the n=500, num_steps=10 protocol reproduced
**57.4% (287/500) exactly**.)

**5. Denoising steps cost accuracy — but only visibly at n=500.** `num_steps` 10 → 1 gives
**52.2% (261/500)** vs **57.4% (287/500)**. The Wilson intervals overlap ([47.8, 56.5] vs
[53.0, 61.7]), so an unpaired read says "no difference" — but both configs ran the *same*
500 (task, init state) pairs, so it is paired: 69 episodes flipped to failure and 43 to
success, **McNemar exact two-sided p = 0.018**. At n=50 I had concluded the speedup was free;
it is not. Policy-only latency drops ~3–5× over that range (measured on
`predict_action_chunk` alone — LIBERO `env.step` costs ~118ms p50 here and would otherwise
dominate).

Happy to share the harness or rerun anything specific — scripts are small wrappers around
`lerobot-eval`.

---

## 게시 전 점검

- [x] n=500 실측 반영 완료 (`results/full_protocol_ns10.json` · `full_protocol_ns1.json`)
- [x] `env.fps` 0% 는 **n=50 기준임을 초안에 명시** (n=500 재확인은 안 했다 — 필요하면 82분)
- [x] `empty_cameras` A/B는 **"판정 불가(p=0.5)"** 로 표현 — 결정론 확인 후 "노이즈" 표현은 폐기
- [x] `lerobot` **0.6.1**(pip 설치본) · torch 2.11.0+cu130 · py 3.12.3 명시. **git 체크아웃이 아니라 커밋 해시는 없다** — 초안에도 버전으로만 적었다
- [x] 논문 수치를 기준선으로 안 씀 — 영문 초안에 90%가 없다(질문자 83.0%만 비교 대상으로 인용)
- [x] **2026-09-10 독립 재현 완료** — n=500·ns10 프로토콜을 처음부터 다시 돌려 57.4%(287/500) 정확히
  일치 확인. 결정론 주장을 뒷받침하는 세 번째 근거로 본문에 반영.
- [ ] **본인이 직접 게시.** 계정·문체 확인 후 올린다
