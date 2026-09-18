"""NVTX 콜 경계 안의 cudaLaunchKernel·ioctl 시간을 지연과 대조한다.

`probe_launch_overhead.py`를 `nsys profile -t cuda,osrt,nvtx`로 감싸 얻은 sqlite를 읽어,
각 `call_i` NVTX 레인지 구간에 속하는 CUDA API(cudaLaunchKernel 등)와 OS 런타임(ioctl 등)
호출의 총 시간·건수를 집계하고, 그 콜의 벽시계 지연(NVTX 레인지 길이)과 함께 표로 낸다.
"제출 경로 시간이 늘어난 콜이 곧 느린 콜인가"에 직접 답한다.
"""
from __future__ import annotations

import sqlite3
import statistics
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "results/nsys_launch.sqlite"


def main() -> None:
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # NVTX_EVENTS: text(레인지 이름) 또는 textId(StringIds 참조), start/end(ns)
    cols = [r[1] for r in cur.execute("PRAGMA table_info(NVTX_EVENTS)")]
    text_col = "text" if "text" in cols else None
    cur.execute("""
        SELECT ne.start, ne.end,
               COALESCE(ne.text, si.value) AS label
        FROM NVTX_EVENTS ne
        LEFT JOIN StringIds si ON ne.textId = si.id
        WHERE COALESCE(ne.text, si.value) LIKE 'call_%'
        ORDER BY ne.start
    """)
    calls = cur.fetchall()
    print("NVTX call_i 레인지 {}개 발견".format(len(calls)))
    if not calls:
        print("call_ 레인지를 못 찾았다 — NVTX_EVENTS 스키마를 확인할 것:", cols)
        return

    rows = []
    for start, end, label in calls:
        dur_ms = (end - start) / 1e6

        # 이 구간에 겹치는 cudaLaunchKernel/cudaStreamSynchronize (CUPTI_ACTIVITY_KIND_RUNTIME)
        cur.execute("""
            SELECT COALESCE(si.value,'?'), COUNT(*), SUM(r.end-r.start)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            LEFT JOIN StringIds si ON r.nameId = si.id
            WHERE r.start < ? AND r.end > ?
            GROUP BY si.value
        """, (end, start))
        cuda_api = {name: (cnt, tot) for name, cnt, tot in cur.fetchall()}

        # 이 구간에 겹치는 ioctl 등 OS 런타임 호출 (OSRT_API)
        cur.execute("""
            SELECT COALESCE(si.value,'?'), COUNT(*), SUM(o.end-o.start)
            FROM OSRT_API o
            LEFT JOIN StringIds si ON o.nameId = si.id
            WHERE o.start < ? AND o.end > ?
            GROUP BY si.value
        """, (end, start))
        osrt_api = {name: (cnt, tot) for name, cnt, tot in cur.fetchall()}

        launch_ns = cuda_api.get("cudaLaunchKernel", (0, 0))[1] or 0
        sync_ns = cuda_api.get("cudaStreamSynchronize", (0, 0))[1] or 0
        ioctl_cnt, ioctl_ns = osrt_api.get("ioctl", (0, 0))
        ioctl_ns = ioctl_ns or 0
        poll_cnt, poll_ns = osrt_api.get("poll", (0, 0))
        poll_ns = poll_ns or 0

        rows.append({
            "label": label, "dur_ms": dur_ms,
            "launch_ms": launch_ns / 1e6, "sync_ms": sync_ns / 1e6,
            "ioctl_ms": ioctl_ns / 1e6, "ioctl_cnt": ioctl_cnt or 0,
            "poll_ms": poll_ns / 1e6, "poll_cnt": poll_cnt or 0,
        })

    print()
    print("{:>8} {:>9} {:>9} {:>9} {:>10} {:>6} {:>9} {:>5}".format(
        "call", "dur_ms", "launch", "sync", "ioctl_ms", "#ioc", "poll_ms", "#pol"))
    for r in rows:
        print("{:>8} {:>9.1f} {:>9.2f} {:>9.2f} {:>10.2f} {:>6} {:>9.2f} {:>5}".format(
            r["label"], r["dur_ms"], r["launch_ms"], r["sync_ms"],
            r["ioctl_ms"], r["ioctl_cnt"], r["poll_ms"], r["poll_cnt"]))

    durs = [r["dur_ms"] for r in rows]
    med = statistics.median(durs)
    lo = [r for r in rows if r["dur_ms"] < med]
    hi = [r for r in rows if r["dur_ms"] >= med]

    def agg(bucket, key):
        vals = [r[key] for r in bucket]
        return statistics.mean(vals) if vals else 0.0

    print()
    print("=== 중앙값({:.1f}ms) 기준 저/고지연 버킷 비교 ===".format(med))
    for key, label in [("dur_ms", "지연(ms)"), ("launch_ms", "cudaLaunchKernel 합(ms)"),
                       ("sync_ms", "cudaStreamSynchronize 합(ms)"),
                       ("ioctl_ms", "ioctl 합(ms)"), ("ioctl_cnt", "ioctl 건수"),
                       ("poll_ms", "poll 합(ms)"), ("poll_cnt", "poll 건수")]:
        lo_v, hi_v = agg(lo, key), agg(hi, key)
        print("  {:28s} 저지연 {:8.3f}  고지연 {:8.3f}  차이 {:+8.3f}".format(
            label, lo_v, hi_v, hi_v - lo_v))


if __name__ == "__main__":
    main()
