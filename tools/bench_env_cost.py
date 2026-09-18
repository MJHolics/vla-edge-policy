"""환경 자체의 비용을 먼저 잰다 — 정책을 붙이기 전에.

왜 이걸 먼저 하나: 스모크 테스트에서 **정책 없이도 스텝당 108.5ms**가 나왔다(5스텝 측정).
사실이라면 sim 종단으로 "제어 주기 몇 Hz"를 재는 것은 **정책이 아니라 시뮬레이터를 재는 것**이 된다.
이 프로젝트의 결론 지표가 제어 주기라서, 이 값을 모르고 가면 결론이 통째로 오염된다.

그래서 분해해서 잰다:
  - `env.step` 전체 (물리 + 오프스크린 렌더링)
  - 렌더링을 뺀 물리만 (카메라 수를 줄여 차이를 본다)
  - 카메라 1대 vs 2대

실제 로봇에서는 물리가 공짜다(현실이 계산해 준다). 카메라 캡처는 30fps면 33ms이고 파이프라인과
병렬로 돈다. 따라서 **sim의 env 비용은 실기 제어 주기와 무관한 오버헤드**이며, 정책 지연과
반드시 분리해서 보고해야 한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import measure_latency, save  # noqa: E402


def build(suite: str, task_id: int, cameras: str):
    import gymnasium as gym
    from lerobot.envs.libero import create_libero_envs
    envs = create_libero_envs(
        suite, n_envs=1, env_cls=gym.vector.SyncVectorEnv,
        camera_name=cameras, gym_kwargs={"task_ids": [task_id]},
    )
    suite_name = list(envs)[0]
    return envs[suite_name][list(envs[suite_name])[0]]


def _leaf_keys(obs) -> list:
    """중첩 dict 관측에서 리프 경로만 뽑는다. 구성 간 관측 구조가 정말 달라졌는지 비교용."""
    out = []
    def walk(o, path):
        if hasattr(o, "items"):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else str(k))
        else:
            out.append(path)
    walk(obs, "")
    return out


def describe_obs(obs, indent: str = "    ") -> None:
    """관측 구조를 재귀로 펼쳐 찍는다. `pixels`가 중첩 dict라 한 겹으로는 안 보인다."""
    def walk(o, path: str) -> None:
        if hasattr(o, "items"):
            for k, v in o.items():
                walk(v, f"{path}[{k}]")
        else:
            shape = getattr(o, "shape", None)
            dtype = getattr(o, "dtype", None)
            print(f"{indent}obs{path}: shape={shape} dtype={dtype}")
    walk(obs, "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=3, help="env 인스턴스를 새로 만들어 반복")
    ap.add_argument("--out", default="results/env_cost.json")
    args = ap.parse_args()

    configs = [
        ("카메라 2대(기본)", "agentview_image,robot0_eye_in_hand_image"),
        ("카메라 1대", "agentview_image"),
    ]

    # **인스턴스마다 다시 만들어 R회 반복한다.** 1차 측정에서 카메라를 줄였는데 오히려 느리게
    # 나왔는데(109.3 → 119.4ms), 단일 인스턴스 비교로는 그게 렌더링 때문인지 인스턴스 간
    # 편차인지 가릴 수 없다. 반복하지 않으면 부호조차 믿을 수 없다.
    rows = []
    for name, cams in configs:
        print()
        print(f"=== {name} ({cams}) ===", flush=True)
        p50s, keys_seen = [], None
        lat = None
        for r in range(args.repeat):
            vec = build(args.suite, args.task_id, cams)
            obs, _ = vec.reset(seed=r)
            if keys_seen is None:
                keys_seen = sorted(_leaf_keys(obs))
                print(f"    관측 리프: {keys_seen}")
                describe_obs(obs)
            space = vec.action_space

            def step_once():
                vec.step(space.sample())

            lat = measure_latency(step_once, name=f"{name}#{r}",
                                  segment="env.step(물리+렌더링)",
                                  n=args.n, warmup=args.warmup)
            p50s.append(lat.p50_ms)
            print(f"    [{r+1}/{args.repeat}] p50 {lat.p50_ms:.1f}ms · p95 {lat.p95_ms:.1f}ms "
                  f"· CV {lat.cv_pct:.1f}%")
            vec.close()

        mean = statistics.mean(p50s)
        sd = statistics.stdev(p50s) if len(p50s) > 1 else 0.0
        print(f"    → p50 평균 {mean:.1f} ± {sd:.1f}ms (인스턴스 {args.repeat}회)")
        rows.append({"name": name, "cameras": cams, "obs_keys": keys_seen,
                     "p50_runs_ms": [round(x, 2) for x in p50s],
                     "p50_mean_ms": round(mean, 2), "p50_sd_ms": round(sd, 2),
                     "latency": lat})

    two, one = rows[0], rows[1]
    delta = one["p50_mean_ms"] - two["p50_mean_ms"]      # 1대 - 2대. 음수면 1대가 빠른 것
    pooled_sd = max(two["p50_sd_ms"], one["p50_sd_ms"], 1e-9)

    print()
    print("=== 결론 ===")
    print(f"카메라 2대 {two['p50_mean_ms']:.1f} ± {two['p50_sd_ms']:.1f}ms  vs  "
          f"1대 {one['p50_mean_ms']:.1f} ± {one['p50_sd_ms']:.1f}ms   (차이 {delta:+.1f}ms)")

    same_obs = two["obs_keys"] == one["obs_keys"]
    if same_obs:
        print("⚠ **관측 리프가 두 구성에서 같다** — `camera_name`이 실제 렌더링 대수를 줄이지 못했다는 뜻이다.")
        print("  즉 이 비교는 '렌더링 비용'을 잰 게 아니다. 손잡이 자체가 작동하지 않았다.")
    if abs(delta) < 2 * pooled_sd:
        print(f"→ 차이({abs(delta):.1f}ms)가 인스턴스 간 편차(2σ={2*pooled_sd:.1f}ms) 안이다. "
              "**어느 쪽이 빠르다고 결론내지 않는다.**")
    else:
        faster = "1대" if delta < 0 else "2대"
        print(f"→ {faster}가 {abs(delta):.1f}ms 빠르다(2σ 밖).")

    print()
    print(f"env.step 비용은 어느 구성이든 대략 {min(two['p50_mean_ms'], one['p50_mean_ms']):.0f}"
          f"~{max(two['p50_mean_ms'], one['p50_mean_ms']):.0f}ms → 종단 8~9Hz 상한.")
    print("→ 이 값은 **시뮬레이터 오버헤드**다. 실기에서는 물리가 공짜이므로 "
          "정책 지연과 반드시 분리해 보고한다.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save(rows, args.out, meta={
        "suite": args.suite, "task_id": args.task_id, "n": args.n, "warmup": args.warmup,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "repeat": args.repeat,
        "note": "정책 없음. env.step만 측정(무작위 액션). 인스턴스를 새로 만들어 반복.",
    })
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
