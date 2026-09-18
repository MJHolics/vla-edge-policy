"""InferAction 서비스를 n회 호출해 ROS2 왕복 지연을 잰다.

policy_server가 시작 시 저장한 results/inprocess_baseline.json(같은 프로세스·같은 GPU
상태에서 잰 순수 호출 지연)과 이 스크립트가 저장하는 results/ros2_roundtrip.json을
비교하면, "ROS2 메시징이 추가하는 지연"만 분리해서 볼 수 있다.

같은 관측(obs_sample.npz)을 매번 그대로 보낸다 — 관측이 바뀌면 추론 시간도 흔들려서
메시징 오버헤드와 뒤섞인다(policy_server.py의 고정 batch 원칙과 동일한 이유).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from vla_ros_bench_msgs.srv import InferAction


def numpy_to_imgmsg(arr: np.ndarray) -> Image:
    """cv_bridge 없이 직접 채운다.

    cv_bridge_boost(컴파일된 확장)가 apt의 system numpy로 빌드돼 있어, venv의
    numpy 2.2.6과 ABI가 안 맞았다 — `cv2_to_imgmsg`가 `KeyError: 16`으로 죽었다
    (encoding→cv-type 매핑 테이블이 numpy dtype 코드로 만들어지는데 그 코드가
    ABI 불일치로 어긋남). rgb8 uint8 HWC는 필드가 단순해 직접 채우는 쪽이 이
    ABI 문제를 아예 피한다.
    """
    msg = Image()
    msg.height, msg.width, _ = arr.shape
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(arr).tobytes()
    return msg

REPO = Path("/mnt/c/Users/apple/Desktop/AI개발/로보틱스/vla-edge-policy")
OBS_PATH = REPO / "ros2_ws" / "obs_sample.npz"
RESULTS_DIR = REPO / "results"


def summarize(samples_ms: list[float], name: str) -> dict:
    s = sorted(samples_ms)
    mean = statistics.mean(s)
    std = statistics.stdev(s) if len(s) > 1 else 0.0

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]

    cv = round(100 * std / mean, 1) if mean else 0.0
    return {
        "name": name, "n": len(s),
        "mean_ms": round(mean, 3), "p50_ms": round(q(0.50), 3),
        "p95_ms": round(q(0.95), 3), "p99_ms": round(q(0.99), 3),
        "std_ms": round(std, 3), "cv_pct": cv, "quotable": cv <= 50.0,
        "samples_ms": [round(x, 3) for x in s],
    }


class BenchClient(Node):
    def __init__(self, n: int, warmup: int) -> None:
        super().__init__("vla_bench_client")
        self.cli = self.create_client(InferAction, "vla/infer_action")
        self.n, self.warmup = n, warmup

        data = np.load(OBS_PATH, allow_pickle=True)
        self.img1_msg = numpy_to_imgmsg(data["image1"])
        self.img2_msg = numpy_to_imgmsg(data["image2"])
        self.state = [float(x) for x in data["state"]]
        self.task = str(data["task"])

    def call_once(self, seq: int) -> tuple[float, InferAction.Response]:
        req = InferAction.Request()
        req.seq = seq
        req.image1 = self.img1_msg
        req.image2 = self.img2_msg
        req.state = self.state
        req.task = self.task
        req.client_send_stamp = time.time()

        t0 = time.time()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        t1 = time.time()
        return (t1 - t0) * 1000.0, future.result()

    def run(self) -> None:
        self.get_logger().info("서비스 대기 중 ...")
        self.cli.wait_for_service()

        self.get_logger().info(f"워밍업 {self.warmup}회")
        for i in range(self.warmup):
            self.call_once(-i - 1)

        self.get_logger().info(f"측정 {self.n}회")
        round_trip_ms, server_proc_ms, overhead_ms = [], [], []
        for i in range(self.n):
            rt_ms, resp = self.call_once(i)
            proc_ms = (resp.server_done_stamp - resp.server_recv_stamp) * 1000.0
            round_trip_ms.append(rt_ms)
            server_proc_ms.append(proc_ms)
            overhead_ms.append(rt_ms - proc_ms)

        rt = summarize(round_trip_ms, "ros2_round_trip")
        proc = summarize(server_proc_ms, "server_side_inference(ROS2 콜백 내부)")
        overhead = summarize(overhead_ms, "ros2_messaging_overhead(왕복 - 서버추론)")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path = RESULTS_DIR / "inprocess_baseline.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None

        out = {"round_trip": rt, "server_side_inference": proc, "messaging_overhead": overhead,
              "inprocess_baseline": baseline, "action_horizon": None}
        (RESULTS_DIR / "ros2_roundtrip.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        self.get_logger().info(
            f"왕복: mean={rt['mean_ms']}ms p50={rt['p50_ms']}ms p95={rt['p95_ms']}ms cv={rt['cv_pct']}%")
        self.get_logger().info(
            f"서버측 추론: mean={proc['mean_ms']}ms p50={proc['p50_ms']}ms")
        self.get_logger().info(
            f"메시징 오버헤드(왕복-서버추론): mean={overhead['mean_ms']}ms p50={overhead['p50_ms']}ms "
            f"cv={overhead['cv_pct']}% quotable={overhead['quotable']}")
        if baseline:
            self.get_logger().info(
                f"in-process 기준선(ROS2 미개입): mean={baseline['mean_ms']}ms p50={baseline['p50_ms']}ms")
        self.get_logger().info("-> results/ros2_roundtrip.json 저장")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rclpy.init()
    node = BenchClient(n, warmup)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
