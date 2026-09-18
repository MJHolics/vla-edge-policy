"""ROS2 실험용 고정 관측 1개를 저장한다.

vla_ros_bench의 policy_server/bench_client가 둘 다 이 파일을 읽어 같은 관측을 쓴다.
같은 데이터를 반복 전송해야 재는 게 "메시징 오버헤드"이지 "관측마다 다른 추론 시간"이
되지 않는다 (tools/bench_policy_latency.py의 고정 batch 재사용 원칙과 동일).

state 벡터는 raw robot_state(중첩 dict: eef pos/mat 등)를 직접 손대지 않고,
`build_policy_and_batch`가 만드는 전처리 완료 배치(observation.state)에서 그대로 뽑는다 —
직접 재구성하면 정규화·평탄화 규칙이 어긋날 수 있어서다(SETUP.md의 "공식 경로를 써야
같은 배치가 나온다" 원칙과 동일).
"""
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from bench_policy_latency import build_policy_and_batch  # noqa: E402
from lerobot.envs import make_env  # noqa: E402
from lerobot.envs.configs import LiberoEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402


def to_uint8_hwc(img_chw: np.ndarray) -> np.ndarray:
    arr = np.transpose(img_chw, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def main() -> None:
    # 1) 전송용 raw 이미지(uint8, HWC) — 정책 preprocessor를 타기 전 값
    env_cfg = LiberoEnv(task="libero_spatial", task_ids=[0])
    envs = make_env(env_cfg, n_envs=1)
    vec = envs
    while isinstance(vec, dict):
        vec = vec[list(vec)[0]]
    raw_obs, _ = vec.reset(seed=0)
    obs = preprocess_observation(raw_obs)
    vec.close()

    img1 = to_uint8_hwc(obs["observation.images.image"][0].cpu().numpy())
    img2 = to_uint8_hwc(obs["observation.images.image2"][0].cpu().numpy())

    # 2) 정책 입력용 state 벡터·task 문자열 — 공식 경로(build_policy_and_batch)에서 그대로 추출
    print("=== 정책 로드(공식 전처리 배치 확보용, 1회) ===", flush=True)
    policy, batch, chunk_shape = build_policy_and_batch(
        "lerobot/smolvla_libero", "libero_spatial", 0, "cuda", empty_cameras=0)
    state = batch["observation.state"][0].detach().cpu().numpy().astype(np.float32)
    task = batch.get("task", [""])[0] if isinstance(batch.get("task"), list) else ""

    out = Path(__file__).resolve().parent / "obs_sample.npz"
    np.savez(out, image1=img1, image2=img2, state=state, task=np.array(task))
    print("saved:", out)
    print("image1 shape:", img1.shape, "state shape:", state.shape, "task:", task,
          "chunk_shape:", chunk_shape)


if __name__ == "__main__":
    main()
