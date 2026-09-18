"""조기 재추론 트리거 — 개루프 드리프트를 "고정 청킹 길이"가 아니라 "필요할 때만" 고친다.

`sweep_chunk.py`가 확인한 것: n_action_steps(개루프 실행 길이)를 1로 고정하면 82.7%,
50으로 고정하면 48.7%다. 즉 **자주 재추론하면 성공률이 오른다 — 그런데 그건 "항상 자주
재추론"해야 얻는 이득이라 효율을 통째로 버린다.** 이 스크립트가 확인하려는 건 "쉬운 구간은
길게 개루프로 밀고, 드리프트 조짐이 보이는 구간에서만 조기에 재추론하면, nas=1 수준의
성공률을 nas=1보다 훨씬 적은 추론 횟수로 살 수 있는가"다.

트리거 신호 — 왜 이 값인가:
  `LiberoProcessorStep`(env_processor.py)을 직접 읽어 확인했다 — `observation.state[0:3]`은
  raw `robot_state.eef.pos`를 그대로 통과시킨 값이고(회전·그리퍼는 뒤에 붙는다), action의
  0:2번째 차원은 정규화 통계상 대략 0-중심·소범위([-0.94,0.94])라 **델타 위치 명령**으로 보인다
  (robosuite OSC_POSE 컨트롤러의 표준 7D 액션: dpos(3)·drot(3)·gripper(1)과 일치).
  그래서 "청크가 커밋한 이후 명령한 누적 변위(pred_cum = Σ action[:3])"와 "실제로 움직인
  누적 변위(actual_cum = state[:3]_now − state[:3]_at_infer)"를 비교할 수 있다 — **명령은
  계속 나가는데 실제로는 안 움직이거나(막힘) 엉뚱한 방향으로 가면(진동/서성임)** 그게 곧
  영상에서 본 "375프레임까지 접시 근처를 서성이는" 실패 패턴의 수치 버전이다.

  ratio = |actual_cum| / |pred_cum|  (진행이 안 됨 → 작다)
  cos   = actual_cum·pred_cum / (|actual_cum||pred_cum|)  (방향이 어긋남 → 작거나 음수)

구현 — `lerobot-eval` CLI를 우회하고 `lerobot.scripts.lerobot_eval`의 `rollout`/`eval_policy`를
**그대로** 쓴다(성공 판정·시드·에피소드 회계를 재구현해 틀릴 위험을 피한다). 대신 그 함수들에
넘기는 `env_preprocessor`/`env_postprocessor`를 얇은 계측 래퍼로 감싸고, `policy.select_action`을
인스턴스 속성으로 교체해 큐가 빌 때(=새 추론 직전)를 잡는다. 트리거가 뜨면
`policy._queues[ACTION].clear()`로 큐를 강제로 비워 **다음 select_action 호출이 즉시 재추론하게**
만든다 — `n_action_steps`가 원래 하는 일(큐가 비면 재추론)을 그대로 이용하는 것이라
정책 내부 로직을 재구현하지 않는다.

배치=1 고정: `_queues[ACTION]`은 배치 전체가 공유하는 큐라서(각 원소가 (batch, action_dim)),
배치>1이면 트리거가 배치 내 한 env에서만 필요해도 전체가 같이 재추론된다 — 개별 env의
독립적인 트리거를 보려면 배치=1이 맞다. 대가는 속도(5배 병렬을 못 씀).

사용: python tools/adaptive_chunk_rollout.py --episodes 5 --tasks all --min-check 5 \
          --tau-ratio 0.4 --tau-cos 0.3 --max-open-loop 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import ACTION, OBS_STATE

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def _first_vec_env(envs):
    if hasattr(envs, "reset"):
        return envs
    cur = envs
    while isinstance(cur, dict):
        cur = cur[list(cur)[0]]
    return cur


def build_policy(policy_path: str, suite: str, device: str, num_steps: int,
                  n_action_steps_ceiling: int, empty_cameras: int):
    """정책·전/후처리기를 한 번만 만든다(태스크마다 새로 만들지 않음 — env만 태스크별로 새로 만든다)."""
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    policy_cfg.empty_cameras = empty_cameras
    policy_cfg.num_steps = num_steps
    # 이 천장까지는 트리거가 알아서 판단하게 둔다. 자연 소진(트리거 없이 천장 도달)도
    # "실패한 트리거"가 아니라 "이 구간은 안전했다"는 신호로 따로 집계한다.
    policy_cfg.n_action_steps = n_action_steps_ceiling

    env_cfg = LiberoEnv(task=suite, task_ids=[0])  # 정책 생성엔 task_ids가 영향 없음(스펙만 씀)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=RENAME_MAP)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    return policy, preprocessor, postprocessor


def make_task_env(suite: str, task_id: int, policy_cfg):
    env_cfg = LiberoEnv(task=suite, task_ids=[task_id])
    envs = make_env(env_cfg, n_envs=1)
    vec = _first_vec_env(envs)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg)
    return vec, env_preprocessor, env_postprocessor


def install_trigger(policy, env_preprocessor, env_postprocessor,
                     min_check: int, max_open_loop: int, tau_ratio: float, tau_cos: float,
                     episode_log: list):
    """`env_preprocessor`/`env_postprocessor`를 계측 래퍼로 감싸고 `policy.select_action`을
    인스턴스 속성으로 교체한다. 반환값은 `eval_policy`에 그대로 넘길 (env_pre, env_post) 쌍.
    `policy.select_action`은 함수가 끝나도 policy에 남으므로 다음 태스크에서 재사용해도 안전하다
    (매번 같은 `install_trigger`를 다시 불러 새 `episode_log`로 교체하면 됨).
    """
    orig_select_action = type(policy).select_action.__get__(policy)  # 클래스 원본에 바인딩

    state = {
        "current_pos": None,   # 이번 스텝 관측(공정처리 후, raw 물리 단위) 위치
        "baseline_pos": None,  # 마지막 재추론 시점의 위치
        "pred_cum": np.zeros(3),
        "steps_since": 0,
        "n_infer": 0,
        "n_trigger": 0,
        "n_exhaust": 0,
        "cur_ep_events": [],
    }

    def wrapped_env_preprocessor(observation):
        obs = env_preprocessor(observation)
        pos = obs[OBS_STATE][:, :3].detach().cpu().numpy().copy()

        if state["baseline_pos"] is not None and state["steps_since"] >= min_check:
            actual_cum = pos[0] - state["baseline_pos"][0]
            pred_cum = state["pred_cum"]
            pred_norm = float(np.linalg.norm(pred_cum))
            act_norm = float(np.linalg.norm(actual_cum))
            ratio = act_norm / (pred_norm + 1e-6)
            cos = (float(np.dot(actual_cum, pred_cum) / (pred_norm * act_norm + 1e-6))
                   if pred_norm > 1e-6 and act_norm > 1e-6 else 1.0)
            if (ratio < tau_ratio or cos < tau_cos) and state["steps_since"] < max_open_loop:
                policy._queues[ACTION].clear()
                state["n_trigger"] += 1
                state["cur_ep_events"].append({
                    "kind": "trigger", "step": state["steps_since"],
                    "ratio": round(ratio, 3), "cos": round(cos, 3),
                })

        state["current_pos"] = pos
        return obs

    def wrapped_env_postprocessor(action_transition):
        result = env_postprocessor(action_transition)
        act_np = result[ACTION].detach().cpu().numpy()
        state["pred_cum"] = state["pred_cum"] + act_np[0, :3]
        state["steps_since"] += 1
        return result

    def patched_select_action(observation, *a, **kw):
        was_empty = len(policy._queues[ACTION]) == 0
        if was_empty:
            if state["baseline_pos"] is not None and 0 < state["steps_since"] < max_open_loop:
                # 트리거가 아니라 정책 리셋(새 에피소드) 등으로 큐가 비었다면 별도 표시 없음.
                pass
            elif state["baseline_pos"] is not None and state["steps_since"] >= max_open_loop:
                state["n_exhaust"] += 1
                state["cur_ep_events"].append({"kind": "exhaust", "step": state["steps_since"]})
            state["baseline_pos"] = state["current_pos"]
            state["steps_since"] = 0
            state["pred_cum"] = np.zeros(3)
            state["n_infer"] += 1
        return orig_select_action(observation, *a, **kw)

    policy.select_action = patched_select_action

    def flush_episode():
        """`rollout()`이 한 에피소드를 끝낼 때마다(=policy.reset 호출 시) 이전 에피소드
        통계를 episode_log에 밀어넣는다. `policy.reset`도 감싼다."""
        episode_log.append({
            "n_infer": state["n_infer"], "n_trigger": state["n_trigger"],
            "n_exhaust": state["n_exhaust"], "events": state["cur_ep_events"],
        })
        state["n_infer"] = 0
        state["n_trigger"] = 0
        state["n_exhaust"] = 0
        state["cur_ep_events"] = []
        state["baseline_pos"] = None
        state["steps_since"] = 0
        state["pred_cum"] = np.zeros(3)

    orig_reset = type(policy).reset.__get__(policy)

    first_call = {"seen": False}

    def patched_reset(*a, **kw):
        if first_call["seen"]:
            flush_episode()
        first_call["seen"] = True
        return orig_reset(*a, **kw)

    policy.reset = patched_reset

    def finalize():
        """마지막 에피소드는 다음 reset이 없으므로 수동으로 flush."""
        if state["n_infer"] > 0 or state["cur_ep_events"]:
            flush_episode()

    return wrapped_env_preprocessor, wrapped_env_postprocessor, finalize


def run_task(policy, preprocessor, postprocessor, suite: str, task_id: int, episodes: int,
             start_seed: int, min_check: int, max_open_loop: int, tau_ratio: float, tau_cos: float) -> dict:
    vec, env_preprocessor, env_postprocessor = make_task_env(suite, task_id, policy.config)
    episode_log: list = []
    wrapped_pre, wrapped_post, finalize = install_trigger(
        policy, env_preprocessor, env_postprocessor,
        min_check, max_open_loop, tau_ratio, tau_cos, episode_log)

    t0 = time.time()
    try:
        info = eval_policy(
            env=vec, policy=policy,
            env_preprocessor=wrapped_pre, env_postprocessor=wrapped_post,
            preprocessor=preprocessor, postprocessor=postprocessor,
            n_episodes=episodes, start_seed=start_seed,
        )
    finally:
        finalize()
        vec.close()
    dt = time.time() - t0

    successes = [bool(ep["success"]) for ep in info["per_episode"]]
    k, n = sum(successes), len(successes)
    lo, hi = wilson(k, n)

    # episode_log는 policy.reset 호출 순서로 쌓이므로 eval_policy의 에피소드 순서와 같다고 기대되나
    # 배치=1이라 rollout()이 배치당 1회씩 순차 호출되므로 정확히 대응한다.
    n_infers = [e["n_infer"] for e in episode_log[:n]]
    n_triggers = [e["n_trigger"] for e in episode_log[:n]]
    n_exhausts = [e["n_exhaust"] for e in episode_log[:n]]

    return {
        "task_id": task_id, "success": k, "n_episodes": n,
        "success_rate": round(100.0 * k / n, 1) if n else None,
        "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
        "avg_inferences_per_ep": round(float(np.mean(n_infers)), 2) if n_infers else None,
        "avg_triggers_per_ep": round(float(np.mean(n_triggers)), 2) if n_triggers else None,
        "avg_exhausts_per_ep": round(float(np.mean(n_exhausts)), 2) if n_exhausts else None,
        "seconds": round(dt, 1),
        "episode_log": episode_log[:n],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="all", help='"all" 또는 "0,1,2"')
    ap.add_argument("--episodes", type=int, default=5, help="태스크당 에피소드")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--empty-cameras", type=int, default=0)
    ap.add_argument("--max-open-loop", type=int, default=50, help="트리거 없어도 강제 재추론하는 천장")
    ap.add_argument("--min-check", type=int, default=5, help="이 스텝 전엔 트리거 판정 안 함")
    ap.add_argument("--tau-ratio", type=float, default=0.4, help="실제/명령 변위 비율이 이보다 작으면 트리거")
    ap.add_argument("--tau-cos", type=float, default=0.3, help="방향 코사인이 이보다 작으면 트리거")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/adaptive_chunk.json")
    args = ap.parse_args()

    task_ids = list(range(10)) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]

    print("정책={} suite={} tasks={} 에피소드/태스크={} min_check={} max_open_loop={} "
          "tau_ratio={} tau_cos={}".format(args.policy, args.suite, task_ids, args.episodes,
                                            args.min_check, args.max_open_loop, args.tau_ratio, args.tau_cos),
          flush=True)

    # 2026-09-12: WSL 자체(Windows Update의 "Windows Subsystem for Linux" 스토어 앱 갱신)가
    # 실행 중이던 VM을 통째로 재시작시켜 10태스크 중 7개까지 돌린 실행이 파일 저장 직전에 날아갔다
    # (overnight5.sh가 이미 겪은 "마지막에 한 번만 저장하면 도중에 죽을 때 전부 사라진다"와 같은 증상,
    # 원인만 절전이 아니라 WSL 자체 업데이트였다). 그래서 태스크마다 즉시 저장하고, 이미 있는
    # task_id는 건너뛴다 — 같은 --out으로 다시 실행하면 이어서 한다.
    out_path = Path(args.out)
    rows_by_task: dict[int, dict] = {}
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for r in prev.get("rows", []):
            rows_by_task[r["task_id"]] = r
        if rows_by_task:
            print("이어서 실행: 이미 있는 task_id={} 는 건너뜀".format(sorted(rows_by_task)), flush=True)

    def save(rows_by_task: dict[int, dict]) -> None:
        rows = [rows_by_task[t] for t in sorted(rows_by_task)]
        k = sum(r["success"] for r in rows)
        n = sum(r["n_episodes"] for r in rows)
        lo, hi = wilson(k, n)
        total_infer = sum(r["avg_inferences_per_ep"] * r["n_episodes"] for r in rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "meta": {"policy": args.policy, "suite": args.suite, "tasks": sorted(rows_by_task),
                     "episodes_per_task": args.episodes, "num_steps": args.num_steps,
                     "empty_cameras": args.empty_cameras, "max_open_loop": args.max_open_loop,
                     "min_check": args.min_check, "tau_ratio": args.tau_ratio, "tau_cos": args.tau_cos,
                     "batch": 1,
                     "note": "배치=1(큐가 배치 공유라 개별 트리거를 보려면 배치=1 필요). "
                             "trigger 신호=실제/명령 누적변위 비율·방향코사인(observation.state[:3]=eef_pos, "
                             "action[:3]=델타위치 가정, env_processor.py LiberoProcessorStep 확인 근거)."},
            "rows": rows,
            "overall": {"success": k, "n_episodes": n, "success_rate": round(100 * k / n, 1) if n else None,
                        "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
                        "avg_inferences_per_ep": round(total_infer / n, 2) if n else None},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    policy, preprocessor, postprocessor = build_policy(
        args.policy, args.suite, args.device, args.num_steps, args.max_open_loop, args.empty_cameras)

    for tid in task_ids:
        if tid in rows_by_task:
            continue
        print("--- task {} 실행 중 ---".format(tid), flush=True)
        r = run_task(policy, preprocessor, postprocessor, args.suite, tid, args.episodes,
                     args.start_seed + tid * 1000, args.min_check, args.max_open_loop,
                     args.tau_ratio, args.tau_cos)
        rows_by_task[tid] = r
        print("   성공률 {}% ({}/{})  Wilson95 {}  평균추론 {}회/ep(트리거 {}·소진 {})  {}초".format(
            r["success_rate"], r["success"], r["n_episodes"], r["wilson95"],
            r["avg_inferences_per_ep"], r["avg_triggers_per_ep"], r["avg_exhausts_per_ep"],
            r["seconds"]), flush=True)
        save(rows_by_task)  # 태스크마다 즉시 저장 — 중간에 죽어도 여기까지는 남는다

    rows = [rows_by_task[t] for t in sorted(rows_by_task)]
    k = sum(r["success"] for r in rows)
    n = sum(r["n_episodes"] for r in rows)
    lo, hi = wilson(k, n)
    total_infer = sum(r["avg_inferences_per_ep"] * r["n_episodes"] for r in rows)
    print()
    print("=== 전체 ===")
    print("성공률 {:.1f}% ({}/{})  Wilson95 [{:.1f}, {:.1f}]  평균추론 {:.1f}회/ep".format(
        100.0 * k / n, k, n, 100 * lo, 100 * hi, total_infer / n))
    print()
    print("저장: {}".format(args.out))


if __name__ == "__main__":
    main()
