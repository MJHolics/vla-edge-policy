"""측정 유틸 — 지연·VRAM을 이 레포군의 규율대로 잰다. 프레임워크 비의존(호출 가능 객체만 받는다).

여기 박아 둔 규칙은 전부 앞선 프로젝트에서 **틀렸다가 고친 결과**다:

1. **워밍업을 버린다.** 첫 호출은 커널 컴파일·메모리 할당이 섞여 본 값이 아니다.
2. **반복하고 표준편차를 함께 낸다.** `video-stream-serving`에서 단일 실행으로 쓴 "p95 7.5배"가
   반복해 보니 비교 대상의 흔들림이었고 2.3배로 정정해야 했다.
3. **CV가 50%를 넘는 지표는 배수로 인용하지 않는다.** `quotable()`이 이걸 기계로 판정한다.
4. **어느 구간을 쟀는지 이름에 남긴다.** "TensorRT 2.4배"가 구간을 바꾸니 1.23배였던 전례가 있다.
5. **GPU는 동기화하고 잰다.** CUDA 호출은 비동기라 동기화 없이 재면 커널 시간이 빠진다.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable


def _sync() -> None:
    """CUDA가 있으면 동기화. 없으면 no-op(CPU 측정도 같은 코드로 돈다)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class Latency:
    """한 구성의 지연 분포. 배수 인용 가능 여부까지 함께 들고 다닌다."""

    name: str
    segment: str            # 무엇을 쟀나 — "policy.forward" / "obs→action(전처리 포함)" 등
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    cv_pct: float
    max_hz: float           # 1회 추론 기준 달성 가능 제어 주기(청킹 미적용)
    samples_ms: list = field(default_factory=list)

    @property
    def quotable(self) -> bool:
        """이 지표를 '몇 배'로 인용해도 되는가. CV 50% 초과면 안 된다(기존 규칙)."""
        return self.cv_pct <= 50.0

    def hz_with_chunk(self, horizon: int) -> float:
        """액션 청킹 H를 적용했을 때의 유효 제어 주기. 개루프 실행분을 곱한다."""
        return self.max_hz * horizon

    def fits(self, budget_hz: float, horizon: int = 1) -> bool:
        """제어 주기 예산 안에 들어가는가. **p95로 판정한다** — 평균으로 보면 주기를 넘긴다."""
        budget_ms = 1000.0 / budget_hz
        return (self.p95_ms / horizon) <= budget_ms

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quotable"] = self.quotable
        return d


def measure_latency(fn: Callable[[], object], *, name: str, segment: str,
                    n: int = 100, warmup: int = 10) -> Latency:
    """`fn`을 n회 호출해 지연 분포를 낸다. warmup회는 버린다."""
    for _ in range(warmup):
        fn()
    _sync()

    samples: list = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1000.0)

    s = sorted(samples)
    mean = statistics.mean(s)
    std = statistics.stdev(s) if len(s) > 1 else 0.0

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]

    return Latency(
        name=name, segment=segment, n=n,
        mean_ms=round(mean, 3), p50_ms=round(q(0.50), 3),
        p95_ms=round(q(0.95), 3), p99_ms=round(q(0.99), 3),
        std_ms=round(std, 3), cv_pct=round(100 * std / mean, 1) if mean else 0.0,
        max_hz=round(1000.0 / q(0.95), 2) if q(0.95) else 0.0,
        samples_ms=[round(x, 3) for x in samples],
    )


def peak_vram_mb() -> float | None:
    """직전 구간의 VRAM 피크(MB). CUDA 없으면 None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    except Exception:
        return None


def reset_vram_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def speedup(base: Latency, new: Latency, *, metric: str = "p50_ms") -> dict:
    """두 구성의 배수. **둘 다 인용 가능할 때만 배수를 낸다** — 아니면 이유와 함께 거절한다."""
    b, x = getattr(base, metric), getattr(new, metric)
    if not (base.quotable and new.quotable):
        bad = [c.name for c in (base, new) if not c.quotable]
        return {"metric": metric, "ratio": None,
                "refused": f"CV 50% 초과로 배수 인용 불가: {', '.join(bad)}",
                "base_ms": b, "new_ms": x}
    return {"metric": metric, "ratio": round(b / x, 3) if x else None,
            "base_ms": b, "new_ms": x,
            "note": f"{base.segment} 구간 기준"}


def constrained_best(rows: list, *, budget_hz: float, horizon_key: str = "horizon",
                     score_key: str = "success_rate") -> dict | None:
    """**제약 하 최대** — 제어 주기 예산을 만족하는 구성 중 성공률이 가장 높은 것.

    이 프로젝트의 결론 지표다. 정확도 최대가 아니라 "로봇 위에서 돌면서" 최대인 것을 고른다.
    각 row는 {"name", "latency": Latency, horizon_key, score_key}를 갖는다.
    """
    feasible = [r for r in rows
                if r["latency"].fits(budget_hz, r.get(horizon_key, 1))]
    if not feasible:
        return None
    return max(feasible, key=lambda r: r.get(score_key, 0.0))


def save(rows: list, path: str, meta: dict | None = None) -> None:
    """결과를 JSON으로. Latency는 dict로 펼쳐 저장한다(재분석 가능하게 표본까지)."""
    out = []
    for r in rows:
        r = dict(r)
        if isinstance(r.get("latency"), Latency):
            r["latency"] = r["latency"].to_dict()
        out.append(r)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta or {}, "rows": out}, f, ensure_ascii=False, indent=2)
