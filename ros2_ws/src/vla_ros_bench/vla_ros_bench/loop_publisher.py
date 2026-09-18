"""20Hz 타이머로 정책을 돌리는 폐루프 노드 — 로드맵 P4의 원래 목표(토픽·주기 지터).

policy_server.py(서비스/RPC)와는 다른 질문을 묻는다: "실기 컨트롤러가 흔히 쓰는 형태
그대로(고정 주기 타이머 콜백 안에서 추론) 붙였을 때, 요청한 주기를 실제로 얼마나
못 지키는가?"

설계:
- `create_timer(1/20, on_tick)`으로 20Hz(LIBERO 기준 주기)를 요청한다.
- 콜백은 동기 호출이다(SingleThreadedExecutor 기본) — 추론이 끝나기 전엔 다음 틱이
  못 들어온다. 이게 바로 재려는 대상이다: **요청 주기(50ms) < 콜백 소요(추론)**일 때
  ROS2 타이머가 실제로 어떤 간격으로 도는가.
- 매 틱마다 벽시계 시각을 찍어 틱-간 간격을 직접 측정한다(rclpy가 "몇 번 놓쳤다"를
  알려주지 않으므로, 간격 자체가 곧 지터·오버런의 증거다).
- 액션은 `/vla/action_stream`(std_msgs/Header만 헤더로 스탬프 전달 — 실제 로봇이라면
  Float32MultiArray 등을 같이 보내겠지만, 이 실험은 "몇 Hz로 도는가"만 본다)으로 발행해,
  구독자 쪽 도착 간격과 대조한다(§ 대조 참고).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

TOOLS_DIR = Path("/mnt/c/Users/apple/Desktop/AI개발/로보틱스/vla-edge-policy/tools")
sys.path.insert(0, str(TOOLS_DIR))

NOMINAL_HZ = 20.0
NOMINAL_PERIOD_S = 1.0 / NOMINAL_HZ
N_TICKS = 30
WARMUP_TICKS = 5


class LoopPublisher(Node):
    def __init__(self) -> None:
        super().__init__("vla_loop_publisher")
        self.pub = self.create_publisher(Header, "vla/action_stream", 10)

        import torch
        self.torch = torch

        self.get_logger().info("정책 로드 중 (lerobot/smolvla_libero) ...")
        from bench_policy_latency import build_policy_and_batch  # noqa: WPS433

        t0 = time.time()
        self.policy, self.batch, self.chunk_shape = build_policy_and_batch(
            "lerobot/smolvla_libero", "libero_spatial", 0, "cuda", empty_cameras=0)
        self.policy.eval()
        self.get_logger().info(f"로드 완료 {time.time() - t0:.1f}s")

        self.tick_count = 0
        self.tick_times: list[float] = []
        self.callback_durations_ms: list[float] = []
        self.last_tick_wall: float | None = None

        self.get_logger().info(
            f"타이머 시작: 목표 {NOMINAL_HZ}Hz(주기 {NOMINAL_PERIOD_S*1000:.0f}ms), "
            f"워밍업 {WARMUP_TICKS}틱 + 측정 {N_TICKS}틱")
        self.timer = self.create_timer(NOMINAL_PERIOD_S, self.on_tick)

    def on_tick(self) -> None:
        now = time.time()
        if self.last_tick_wall is not None:
            self.tick_times.append((now - self.last_tick_wall) * 1000.0)
        self.last_tick_wall = now

        t0 = time.perf_counter()
        with self.torch.no_grad():
            self.policy.predict_action_chunk(self.batch)
            self.torch.cuda.synchronize()
        callback_ms = (time.perf_counter() - t0) * 1000.0

        self.tick_count += 1
        if self.tick_count > WARMUP_TICKS:
            self.callback_durations_ms.append(callback_ms)

        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = str(self.tick_count)
        self.pub.publish(msg)

        if self.tick_count >= WARMUP_TICKS + N_TICKS:
            self.finish()

    def finish(self) -> None:
        self.timer.cancel()
        # 워밍업 이후 구간의 틱-간 간격만 집계(첫 WARMUP_TICKS개 간격은 버림)
        measured_intervals = self.tick_times[WARMUP_TICKS:]

        def summarize(vals: list[float], name: str) -> dict:
            s = sorted(vals)
            mean = statistics.mean(s)
            std = statistics.stdev(s) if len(s) > 1 else 0.0
            cv = round(100 * std / mean, 1) if mean else 0.0
            return {"name": name, "n": len(s), "mean_ms": round(mean, 3),
                    "p50_ms": round(s[len(s)//2], 3), "p95_ms": round(s[int(len(s)*0.95)], 3),
                    "std_ms": round(std, 3), "cv_pct": cv, "quotable": cv <= 50.0,
                    "samples_ms": [round(x, 3) for x in s]}

        interval_stats = summarize(measured_intervals, "tick_interval(실제 틱-간 벽시계 간격)")
        cb_stats = summarize(self.callback_durations_ms, "callback_duration(추론+동기화)")

        overrun = interval_stats["mean_ms"] / (NOMINAL_PERIOD_S * 1000.0)
        achieved_hz = 1000.0 / interval_stats["mean_ms"] if interval_stats["mean_ms"] else 0.0

        out = {
            "nominal_hz": NOMINAL_HZ, "nominal_period_ms": NOMINAL_PERIOD_S * 1000.0,
            "tick_interval": interval_stats, "callback_duration": cb_stats,
            "overrun_ratio_mean": round(overrun, 2), "achieved_hz_mean": round(achieved_hz, 2),
        }
        out_path = Path("/mnt/c/Users/apple/Desktop/AI개발/로보틱스/vla-edge-policy/results/loop_jitter.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        self.get_logger().info(
            f"틱 간격: mean={interval_stats['mean_ms']}ms(목표 {NOMINAL_PERIOD_S*1000:.0f}ms) "
            f"p50={interval_stats['p50_ms']}ms cv={interval_stats['cv_pct']}%")
        self.get_logger().info(
            f"오버런 배수={overrun:.2f}x · 실제 달성 Hz={achieved_hz:.2f}(목표 {NOMINAL_HZ}Hz)")
        self.get_logger().info("-> results/loop_jitter.json 저장, 종료")
        rclpy.shutdown()


def main() -> None:
    rclpy.init()
    node = LoopPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
