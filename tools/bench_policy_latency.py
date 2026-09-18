"""정책 **단독** 추론 지연 — 이 프로젝트의 핵심 계측기.

왜 단독으로 재나: `tools/bench_env_cost.py`에서 확인한 대로 LIBERO의 `env.step`이 스텝당 ~118ms다.
sim 종단 wall-clock으로 제어 주기를 재면 정책이 아니라 **시뮬레이터를 재는 것**이 된다.
실기에서는 물리가 공짜이므로, 실기로 전이되는 값은 **정책 추론 시간뿐**이다. 그래서 env는
관측을 한 번 얻는 데만 쓰고, 측정은 정책 호출만 감싼다.

무엇을 재나:
  - `predict_action_chunk` 1회 = **청크 전체(H=50스텝)를 만드는 비용**
  - `select_action` 1회 = 캐시가 비었으면 청크 계산, 아니면 캐시에서 꺼내기
    → 이 둘을 섞어 재면 "가끔 느린 호출"로 뭉개진다. **청크 계산 비용을 따로 잰다.**

제어 주기 판정(2026-08-18 확정): LIBERO는 `control_freq=20Hz`이므로 1스텝 = 50ms.
청크 H스텝을 개루프로 실행하면 1회 추론이 감당할 로봇 시간은 **H × 50ms**다.

사용:
  python tools/bench_policy_latency.py --policy lerobot/smolvla_libero \\
      --num-steps 1,2,3,5,10 --n 30
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import measure_latency, peak_vram_mb, reset_vram_peak, save  # noqa: E402

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def _first_vec_env(envs):
    """`make_env`는 env 타입에 따라 vec env를 바로 주기도, {suite: {task_id: vec}} 로 주기도 한다."""
    if hasattr(envs, "reset"):
        return envs
    cur = envs
    while isinstance(cur, dict):
        cur = cur[list(cur)[0]]
    return cur


def build_policy_and_batch(policy_path: str, suite: str, task_id: int, device: str,
                           empty_cameras: int = 0):
    """공식 평가 경로(`lerobot_eval.py`)와 **같은 순서로** 정책·전처리기·관측을 만든다.

    체인: env obs → preprocess_observation → env_preprocessor → preprocessor → policy
    직접 텐서를 지어내면 정규화·키 이름이 어긋나 지연이 달라진다. 평가와 같은 배치를 써야
    "성공률을 낸 그 구성의 지연"이 된다.
    """
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.envs import make_env_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    # 2026-08-19 정정: 예전엔 여기서 무조건 1로 올렸다. 그런데 학습(`train_config.json`)은
    # `empty_cameras=0`이고 학습 데이터셋에는 카메라가 2대뿐이라 **학습은 이미지 2장으로 이뤄졌다.**
    # 1로 올리면 학습에 없던 -1 패딩 이미지 1장이 매 추론에 들어가 **비전 토큰이 늘고 지연이 부풀린다.**
    # 기본을 0(학습 조건)으로 두고, 두 조건을 비교할 수 있게 손잡이로 뺐다.
    policy_cfg.empty_cameras = empty_cameras

    env_cfg = LiberoEnv(task=suite, task_ids=[task_id])

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=RENAME_MAP)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    # **`make_env`를 쓴다.** `create_libero_envs`를 직접 부르면 `obs_type`(기본
    # "pixels_agent_pos")이 실리지 않아 관측에 `robot_state`가 빠지고, 그러면
    # LiberoProcessorStep이 `observation.state`를 만들지 못해 정책에서 KeyError가 난다.
    # 공식 평가와 같은 경로로 만들어야 같은 배치가 나온다.
    envs = make_env(env_cfg, n_envs=1)
    vec = _first_vec_env(envs)
    raw_obs, _ = vec.reset(seed=0)

    obs = preprocess_observation(raw_obs)

    # 언어 지시문(task)을 넣어야 한다. 토크나이저 전처리기가 complementary_data["task"]를
    # 요구하고, 없으면 KeyError로 죽는다. 평가 루프(`lerobot_eval.rollout`)가 매 스텝
    # `env.call("task_description")`로 채워 넣는 값이라, 여기서도 같은 경로로 얻는다.
    try:
        obs["task"] = list(vec.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            obs["task"] = list(vec.call("task"))
        except (AttributeError, NotImplementedError):
            obs["task"] = [""] * vec.num_envs
    print("    지시문: {}".format(obs["task"]))

    obs = env_preprocessor(obs)
    batch = preprocessor(obs)
    vec.close()

    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)
    print("    배치 키: {}".format(sorted(k for k in batch if isinstance(k, str))))
    print("    액션 청크 shape: {}".format(tuple(chunk.shape)))
    return policy, batch, tuple(chunk.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-steps", default="1,2,3,5,10",
                    help="디노이징 스텝 스윕(쉼표 구분). 기본 설정은 10")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=3,
                    help="스윕 전체를 R회 반복해 실행 간 편차를 본다")
    ap.add_argument("--control-hz", type=float, default=20.0, help="LIBERO 기본 20Hz")
    ap.add_argument("--empty-cameras", type=int, default=0,
                    help="0 = 학습 조건(기본). 1이면 학습에 없던 패딩 카메라가 추가된다")
    ap.add_argument("--out", default="results/policy_latency.json")
    args = ap.parse_args()

    import torch

    print("=== 정책 로드: {} ===".format(args.policy), flush=True)
    t0 = time.time()
    policy, batch, chunk_shape = build_policy_and_batch(
        args.policy, args.suite, args.task_id, args.device, args.empty_cameras)
    horizon = chunk_shape[1] if len(chunk_shape) >= 2 else 1
    print("    로드 {:.1f}s · 청크 H={} · 제어주기 {:.0f}Hz(1스텝 {:.0f}ms)".format(
        time.time() - t0, horizon, args.control_hz, 1000 / args.control_hz), flush=True)

    n_params = sum(p.numel() for p in policy.parameters())
    print("    파라미터 {:.1f}M".format(n_params / 1e6))

    steps_list = [int(x) for x in args.num_steps.split(",") if x.strip()]
    default_steps = getattr(policy.config, "num_steps", None)
    step_ms = 1000.0 / args.control_hz

    # 스윕 전체를 R회 반복한다. 1차 측정에서 CV가 17~34%였고 num_steps=1의 p95가 p50의 1.9배라,
    # 단일 스윕의 p50 배수를 그대로 쓰면 지터를 개선으로 읽을 수 있다.
    rows = []
    for ns in steps_list:
        policy.config.num_steps = ns
        p50s, lats = [], []
        vram = None
        for r in range(args.repeat):
            reset_vram_peak()

            def call():
                with torch.no_grad():
                    policy.predict_action_chunk(batch)

            lat = measure_latency(call, name="num_steps={}#{}".format(ns, r),
                                  segment="policy.predict_action_chunk(청크 전체)",
                                  n=args.n, warmup=args.warmup)
            p50s.append(lat.p50_ms)
            lats.append(lat)
            vram = peak_vram_mb()

        mean = statistics.mean(p50s)
        sd = statistics.stdev(p50s) if len(p50s) > 1 else 0.0
        rep_cv = 100 * sd / mean if mean else 0.0
        best = min(lats, key=lambda x: x.cv_pct)     # 대표값은 스윕 중 가장 안정된 실행
        min_h = best.p95_ms / step_ms
        rows.append({
            "num_steps": ns, "latency": best, "vram_mb": vram,
            "p50_runs_ms": [round(x, 2) for x in p50s],
            "p50_mean_ms": round(mean, 2), "p50_sd_ms": round(sd, 2),
            "p50_repeat_cv_pct": round(rep_cv, 1),
            "p95_over_p50": round(best.p95_ms / best.p50_ms, 2),
            "min_open_loop_steps": round(min_h, 2),
            "fits_closed_loop": bool(best.p95_ms <= step_ms),
            "chunk_horizon": horizon,
        })
        print("  num_steps={:2d} | p50 {:6.1f} ± {:4.1f}ms (실행간CV {:4.1f}%) | "
              "실행내CV {:4.1f}%  p95/p50 {:.2f} | VRAM {:5.0f}MB | 최소 개루프 {:4.1f}스텝".format(
                  ns, mean, sd, rep_cv, best.cv_pct, best.p95_ms / best.p50_ms,
                  vram or -1, min_h), flush=True)

    if default_steps is not None:
        policy.config.num_steps = default_steps

    base = next((r for r in rows if r["num_steps"] == default_steps), rows[-1])
    print()
    print("=== 요약 (실행 {}회 평균 기준) ===".format(args.repeat))
    print("기본값 num_steps={} : p50 {:.1f} ± {:.1f}ms".format(
        base["num_steps"], base["p50_mean_ms"], base["p50_sd_ms"]))
    for r in rows:
        if r is base:
            continue
        gap = base["p50_mean_ms"] - r["p50_mean_ms"]
        two_sigma = 2 * max(base["p50_sd_ms"], r["p50_sd_ms"])
        if abs(gap) < two_sigma:
            print("  num_steps {}→{}: 차이 {:.1f}ms 가 2σ({:.1f}ms) 안 — **배수 인용 안 함**".format(
                base["num_steps"], r["num_steps"], gap, two_sigma))
        else:
            print("  num_steps {}→{}: {:.2f}배 ({:.1f}→{:.1f}ms)".format(
                base["num_steps"], r["num_steps"],
                base["p50_mean_ms"] / r["p50_mean_ms"],
                base["p50_mean_ms"], r["p50_mean_ms"]))

    # 고정비/가변비 분해 — VLM prefix 1회 + 액션 전문가 num_steps회 라는 가설을 최소제곱으로 확인
    xs = [r["num_steps"] for r in rows]
    ys = [r["p50_mean_ms"] for r in rows]
    n_pts = len(xs)
    if n_pts >= 2:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
        intercept = my - slope * mx
        ss_tot = sum((y - my) ** 2 for y in ys)
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        print()
        print("선형 적합  p50 ≈ {:.1f}ms + {:.1f}ms × num_steps   (R²={:.3f})".format(
            intercept, slope, r2))
        print("  → 고정비 {:.0f}ms 는 VLM prefix 1회 통과, 가변비 {:.0f}ms/스텝 은 액션 전문가 반복이라는 가설.".format(
            intercept, slope))
        print("  → R²가 낮으면 이 분해가 틀린 것이다. 잔차:")
        for x, y in zip(xs, ys):
            pred = intercept + slope * x
            print("     num_steps={:2d}  측정 {:6.1f}  예측 {:6.1f}  잔차 {:+6.1f}ms".format(
                x, y, pred, y - pred))
    print()
    print("주의: 위 값은 **정책 단독**이다. sim의 env.step(~118ms)은 포함하지 않았고, "
          "결론에도 넣지 않는다.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save(rows, args.out, meta={
        "policy": args.policy, "suite": args.suite, "task_id": args.task_id,
        "device": args.device, "n": args.n, "warmup": args.warmup,
        "control_hz": args.control_hz, "chunk_horizon": horizon,
        "empty_cameras": args.empty_cameras,
        "params_m": round(n_params / 1e6, 2),
        "default_num_steps": default_steps,
        "repeat": args.repeat,
        "note": "policy.predict_action_chunk만 측정. env.step 제외.",
    })
    print("저장: {}".format(args.out))


if __name__ == "__main__":
    main()
