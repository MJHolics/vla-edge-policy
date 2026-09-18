"""loop_publisher의 /vla/action_stream 도착 간격을 구독자 쪽에서 독립적으로 잰다.

퍼블리셔가 자기 틱 간격을 스스로 재는 값과, 구독자가 실제 메시지를 받는 간격이
같은지 대조한다 — 다르면 그 차이가 토픽 전달 자체가 더하는 지터다.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

N_EXPECTED = 30


class LoopSubscriber(Node):
    def __init__(self) -> None:
        super().__init__("vla_loop_subscriber")
        self.sub = self.create_subscription(Header, "vla/action_stream", self.on_msg, 10)
        self.arrival_times: list[float] = []
        self.last: float | None = None
        self.intervals_ms: list[float] = []

    def on_msg(self, msg: Header) -> None:
        now = time.time()
        self.arrival_times.append(now)
        if self.last is not None:
            self.intervals_ms.append((now - self.last) * 1000.0)
        self.last = now
        if len(self.intervals_ms) >= N_EXPECTED:
            self.finish()

    def finish(self) -> None:
        s = sorted(self.intervals_ms)
        mean = statistics.mean(s)
        std = statistics.stdev(s) if len(s) > 1 else 0.0
        cv = round(100 * std / mean, 1) if mean else 0.0
        out = {"name": "subscriber_arrival_interval", "n": len(s), "mean_ms": round(mean, 3),
              "p50_ms": round(s[len(s)//2], 3), "cv_pct": cv,
              "samples_ms": [round(x, 3) for x in s]}
        out_path = Path("/mnt/c/Users/apple/Desktop/AI개발/로보틱스/vla-edge-policy/results/loop_subscriber_arrival.json")
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        self.get_logger().info(f"구독자 도착간격: mean={out['mean_ms']}ms cv={out['cv_pct']}% -> 저장")
        rclpy.shutdown()


def main() -> None:
    rclpy.init()
    node = LoopSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
