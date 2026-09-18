"""LIBERO 헤드리스 스모크 테스트 — env가 진짜 만들어지고 스텝이 도는지 확인한다.

`import`가 통과하는 것과 **렌더링이 되는 것은 다른 문제**다. MuJoCo는 화면이 없으면
`MUJOCO_GL`(egl | osmesa)을 지정해야 하고, 지정해도 드라이버가 안 맞으면 env 생성에서 죽는다.
그래서 관측 텐서 shape까지 실제로 찍어 본다 — 여기가 통과해야 정책을 붙일 수 있다.

API 메모(직접 읽고 확인, 2026-08-18):
  create_libero_envs(task, n_envs, env_cls=..., gym_kwargs={"task_ids": [...]})
    → dict[suite][task_id] = vec_env      # env_cls가 팩토리 리스트를 감싼다
  `env_cls`는 필수다. None이면 ValueError. `gym.vector.SyncVectorEnv`를 쓴다.
  `task_ids`로 태스크를 제한하면 suite 전체(10개)를 만들지 않아 스모크가 빨라진다.

사용: MUJOCO_GL=egl python tools/smoke_libero.py [--suite libero_spatial]
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()

    def log(msg: str) -> None:
        print("[{:6.1f}s] {}".format(time.time() - t0, msg), flush=True)

    log("MUJOCO_GL={}".format(os.environ.get("MUJOCO_GL", "(미지정)")))

    import gymnasium as gym
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, create_libero_envs
    log("import 완료 · suite 최대 스텝 = {}".format(TASK_SUITE_MAX_STEPS.get(args.suite)))

    envs = create_libero_envs(
        args.suite,
        n_envs=1,
        env_cls=gym.vector.SyncVectorEnv,
        gym_kwargs={"task_ids": [args.task_id]},
    )
    log("env 생성 완료 · suite {}개".format(len(envs)))

    suite = list(envs)[0]
    task_ids = list(envs[suite])
    vec = envs[suite][task_ids[0]]
    log("suite={} · task_ids={} · vec_env={}".format(suite, task_ids, type(vec).__name__))

    obs, info = vec.reset(seed=0)
    log("reset 완료")
    if hasattr(obs, "items"):
        for k, v in obs.items():
            print("    obs[{}] shape={} dtype={}".format(
                k, getattr(v, "shape", None), getattr(v, "dtype", None)))
    else:
        print("    obs type={}".format(type(obs)))
    print("    action_space: {}".format(vec.action_space))

    t_step = time.time()
    for i in range(args.steps):
        a = vec.action_space.sample()
        obs, rew, term, trunc, info = vec.step(a)
        if i == 0:
            log("step 1 OK · reward={} term={} trunc={}".format(rew, term, trunc))
    dt = (time.time() - t_step) / args.steps
    log("step {}회 완료 · 평균 {:.1f}ms/step (물리+렌더링, 정책 없음)".format(args.steps, dt * 1000))

    vec.close()
    log("정상 종료 — 환경 사용 가능")


if __name__ == "__main__":
    main()
