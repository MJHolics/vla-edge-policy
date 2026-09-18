"""SmolVLA 정책을 ROS2 서비스로 감싼 노드.

설계 원칙(vla-edge-policy 레포의 규율을 그대로 따른다):
- 추론에 쓰는 배치는 시작 시 한 번 만든 고정 batch를 재사용한다. 매 요청마다 다시
  전처리하면 "메시징 오버헤드"가 아니라 "전처리 시간 편차"를 재게 된다.
- 요청으로 들어온 이미지(image1/image2)는 cv_bridge로 실제로 디코딩한다 — 페이로드
  전송·역직렬화 비용은 진짜로 발생시키고, 그 다음 추론 입력은 고정 batch를 쓴다.
  (재구성이 아니라 통제된 실험 설계임을 README/카드에 명시한다.)
- GPU 호출은 동기화하고 잰다(measure.py 규칙과 동일) — 그래야 커널 완료 전에 시간을
  끊어 실제보다 빠르게 보이는 착시가 없다.
- 서비스 콜백 진입 직후·응답 직전 두 시각(server_recv_stamp/server_done_stamp)을 함께
  돌려줘서, 클라이언트가 왕복시간에서 "서버가 실제로 쓴 시간"을 분리할 수 있게 한다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from vla_ros_bench_msgs.srv import InferAction


def imgmsg_to_numpy(msg) -> np.ndarray:
    """cv_bridge 없이 직접 파싱한다 — bench_client.numpy_to_imgmsg와 대칭.

    이 프로젝트의 venv(numpy 2.2.6)와 apt cv_bridge_boost(numpy ABI 다름)가
    충돌해 cv2_to_imgmsg/imgmsg_to_cv2가 KeyError로 죽었다. rgb8 uint8 HWC는
    필드가 단순해 수동 파싱으로 그 의존을 아예 없앴다.
    """
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return arr.reshape(msg.height, msg.width, 3)

TOOLS_DIR = Path.home() / "vla_tools_link"
if not TOOLS_DIR.exists():
    # 실제 레포 경로(Windows 마운트) — colcon 빌드 산출물과 분리해 소스만 참조한다.
    TOOLS_DIR = Path(
        "/mnt/c/Users/apple/Desktop/AI개발/로보틱스/vla-edge-policy/tools"
    )
sys.path.insert(0, str(TOOLS_DIR))


class PolicyServer(Node):
    def __init__(self) -> None:
        super().__init__("vla_policy_server")

        import torch
        self.torch = torch

        self.get_logger().info("정책 로드 중 (lerobot/smolvla_libero) ...")
        from bench_policy_latency import build_policy_and_batch  # noqa: WPS433

        t0 = time.time()
        self.policy, self.batch, self.chunk_shape = build_policy_and_batch(
            "lerobot/smolvla_libero", "libero_spatial", 0, "cuda", empty_cameras=0)
        self.policy.eval()
        self.get_logger().info(
            f"로드 완료 {time.time() - t0:.1f}s · 청크 shape={self.chunk_shape}")

        self._inprocess_baseline()

        self.srv = self.create_service(InferAction, "vla/infer_action", self.on_infer)
        self.get_logger().info("서비스 준비: /vla/infer_action")

    def _inprocess_baseline(self, n: int = 60, warmup: int = 20) -> None:
        """ROS2 트래픽 시작 전, 같은 프로세스·같은 GPU 상태에서 순수 호출 지연을 먼저 잰다."""
        sys.path.insert(0, str(TOOLS_DIR))
        from measure import measure_latency  # noqa: WPS433

        torch = self.torch

        def call():
            with torch.no_grad():
                self.policy.predict_action_chunk(self.batch)

        lat = measure_latency(call, name="inprocess_direct_call", segment="policy.predict_action_chunk(고정 batch, ROS2 미개입)",
                              n=n, warmup=warmup)
        out = Path("/mnt/c/Users/apple/Desktop/AI개발/로보틱스/vla-edge-policy/results")
        out.mkdir(parents=True, exist_ok=True)
        import json
        (out / "inprocess_baseline.json").write_text(
            json.dumps(lat.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.get_logger().info(
            f"in-process 기준선: mean={lat.mean_ms}ms p50={lat.p50_ms}ms p95={lat.p95_ms}ms "
            f"cv={lat.cv_pct}% (n={n}) -> results/inprocess_baseline.json")

    def on_infer(self, request: InferAction.Request, response: InferAction.Response):
        server_recv_stamp = time.time()

        # 실제 디코딩(전송·역직렬화 비용을 진짜로 발생시킨다). 추론 입력에는 안 쓴다 — docstring 참조.
        _img1 = imgmsg_to_numpy(request.image1)
        _img2 = imgmsg_to_numpy(request.image2)
        assert _img1.shape[2] == 3 and _img2.shape[2] == 3

        torch = self.torch
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(self.batch)
            torch.cuda.synchronize()

        action = chunk[0, 0].detach().cpu().numpy().astype(np.float32)
        server_done_stamp = time.time()

        response.action = action.tolist()
        response.action_horizon = int(chunk.shape[1])
        response.server_recv_stamp = server_recv_stamp
        response.server_done_stamp = server_done_stamp
        return response


def main() -> None:
    rclpy.init()
    node = PolicyServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
