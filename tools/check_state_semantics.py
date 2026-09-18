"""런타임 state가 **학습 때와 같은 의미**인지 분포로 확인한다.

차원이 같아도 표현이 다르면(예: 회전을 axis-angle 로 학습해 놓고 평가에서 quaternion 을 주면)
정규화도 통과하고 에러도 안 나며 **조용히 틀린다.** 이런 결함은 코드를 읽어서는 안 보이고,
학습 정규화 통계의 범위 안에 런타임 값이 들어오는지로만 잡힌다.

방법: 학습 통계(min/q01/q50/q99/max)와 실제 env 관측을 차원별로 나란히 찍는다.
런타임 값이 학습 q01~q99 밖으로 크게 벗어나는 차원이 있으면 그 차원은 의미가 다르다.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
# 정책 레지스트리를 채우기 위한 import — 이게 없으면
# `PreTrainedConfig.from_pretrained` 가 "Policy type 'smolvla' is not registered" 로 죽는다.
from lerobot.policies.factory import make_policy  # noqa: F401
from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import preprocess_observation

REPO = "lerobot/smolvla_libero"
N_STEPS = 40


def first_vec_env(envs):
    if hasattr(envs, "reset"):
        return envs
    cur = envs
    while isinstance(cur, dict):
        cur = cur[list(cur)[0]]
    return cur


def main() -> None:
    stats = load_file(hf_hub_download(
        REPO, "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    keys = ["observation.state.min", "observation.state.q01", "observation.state.q50",
            "observation.state.q99", "observation.state.max", "observation.state.mean",
            "observation.state.std"]
    tr = {k.split(".")[-1]: stats[k].float().numpy() for k in keys}
    dim_train = tr["mean"].shape[0]
    print("=== 학습 정규화 통계 (observation.state) · {}차원 ===".format(dim_train))

    # `observation.state` 는 preprocess_observation 이 아니라 **env 전처리기**가 만든다
    # (LiberoProcessorStep 이 robot_state 를 합친다). 평가와 같은 체인을 그대로 쓴다.
    env_cfg = LiberoEnv(task="libero_spatial", task_ids=[0])
    policy_cfg = PreTrainedConfig.from_pretrained(REPO)
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    envs = make_env(env_cfg, n_envs=1)
    vec = first_vec_env(envs)
    obs, _ = vec.reset(seed=0)

    samples = []
    for i in range(N_STEPS):
        o = env_preprocessor(preprocess_observation(obs))
        if i == 0:
            print("   전처리 후 키: {}".format(sorted(k for k in o if isinstance(k, str))))
        st = o["observation.state"]
        if hasattr(st, "detach"):
            st = st.detach().cpu()
        samples.append(np.asarray(st).squeeze())
        act = np.zeros((vec.num_envs, 7), dtype=np.float32)
        obs, _, term, trunc, _ = vec.step(act)
        if bool(np.any(term)) or bool(np.any(trunc)):
            obs, _ = vec.reset(seed=0)

    run = np.stack(samples)  # (N, D)
    print("런타임 관측: shape={} (무동작 {}스텝)".format(run.shape, N_STEPS))
    print()

    if run.shape[1] != dim_train:
        print("!! 차원부터 다르다: 런타임 {} vs 학습 {}".format(run.shape[1], dim_train))

    print("{:>4} | {:>32} | {:>26} | {}".format(
        "dim", "학습 [q01, q99]  (min~max)", "런타임 [min, max]", "판정"))
    bad = []
    for d in range(min(run.shape[1], dim_train)):
        lo, hi = float(tr["q01"][d]), float(tr["q99"][d])
        rlo, rhi = float(run[:, d].min()), float(run[:, d].max())
        span = max(hi - lo, 1e-6)
        # 런타임 값이 학습 q01~q99 구간에서 얼마나 벗어났나 (구간 폭 기준)
        out = max(0.0, (lo - rlo) / span, (rhi - hi) / span)
        verdict = "OK" if out < 0.5 else ("의심 {:.1f}배 밖".format(out) if out < 3
                                          else "**어긋남 {:.1f}배 밖**".format(out))
        if out >= 0.5:
            bad.append(d)
        print("{:>4} | [{:>8.3f},{:>8.3f}] ({:>6.3f}~{:>6.3f}) | [{:>10.4f},{:>10.4f}] | {}".format(
            d, lo, hi, float(tr["min"][d]), float(tr["max"][d]), rlo, rhi, verdict))

    print()
    if bad:
        print("어긋난 차원: {} → **표현이 다르다는 뜻일 수 있다**(회전 표현·단위·부호).".format(bad))
    else:
        print("모든 차원이 학습 분포 안에 있다 → state 의미 불일치는 원인이 아니다.")

    vec.close()


if __name__ == "__main__":
    main()
