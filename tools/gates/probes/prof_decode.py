"""Decode rounds in an nsys CUDA trace: one round per cudaGraphLaunch.

  python prof_decode.py <sqlite> [top kernels, default 15]

Per round: GPU span (first kernel start to last kernel end), busy (union of kernel time),
idle inside the round, and the gap from the round's last kernel to the next round's first.
Work between graph launches (host copies, small eager kernels) counts toward that gap.

Also attributes kernel time to NVTX ranges whose names contain decode.mtp.target,
decode.mtp.draft, or mtp.proposal, when those ranges overlap kernel timestamps.
CUDA-graph replay often has no such overlap: the ranges are pushed on the host while
the graph is captured, and replay is one cudaGraphLaunch. The script says so, and
still buckets replay kernels by name (proposal, moe, fp8 gemv, gdn, qsa, other).
"""
import collections
import sqlite3
import statistics
import sys

RANGES = ("decode.mtp.target", "decode.mtp.draft", "mtp.proposal")


def median(xs):
    return statistics.median(xs)


def quantile(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]


def tables(db):
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def strings(db):
    names = {}
    if "StringIds" in tables(db):
        names = dict(db.execute("SELECT id, value FROM StringIds"))
    return names


def kernel_class(name):
    low = name.lower()
    if "proposal" in low or "shortlist" in low or "argmax" in low:
        return "proposal"
    if "moe" in low or "expert" in low or "nvfp4" in low:
        return "moe"
    if "gemv" in low or "fp8" in low:
        return "fp8_gemv"
    if "gdn" in low or "gated_delta" in low or "conv1d" in low:
        return "gdn"
    if "qsa" in low or "attention" in low or "flash" in low:
        return "qsa"
    if "router" in low or "route_token" in low or "topk" in low:
        return "router"
    return "other"


def load_ranges(db, names):
    found = tables(db)
    if "NVTX_EVENTS" not in found:
        return []
    columns = [row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")]
    wanted = [name for name in ("start", "end", "text", "textId") if name in columns]
    if "start" not in wanted or "end" not in wanted:
        return []
    ranges = []
    for row in db.execute(f"SELECT {', '.join(wanted)} FROM NVTX_EVENTS"):
        record = dict(zip(wanted, row))
        text = record.get("text") or ""
        if not text and record.get("textId") is not None:
            text = names.get(record["textId"], "")
        if not text:
            continue
        start = record["start"]
        end = record["end"]
        if start is None or end is None or end <= start:
            continue
        ranges.append((start, end, text))
    return ranges


def attribute(kernels, ranges):
    share = collections.Counter()
    for k_start, k_end, _name in kernels:
        hits = [item for item in ranges if item[0] <= k_start and item[1] >= k_end]
        if not hits:
            continue
        text = min(hits, key=lambda item: item[1] - item[0])[2]
        for label in RANGES:
            if label in text:
                share[label] += k_end - k_start
                break
    return share


def main():
    db = sqlite3.connect(sys.argv[1])
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    names = strings(db)
    launch_ids = {i for i, value in names.items() if str(value).startswith("cudaGraphLaunch")}
    launches = [
        (start, end, corr)
        for start, end, name_id, corr in db.execute(
            "SELECT start, end, nameId, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"
        )
        if name_id in launch_ids
    ]
    by_corr = collections.defaultdict(list)
    all_kernels = []
    for k_start, k_end, corr, name_id in db.execute(
        "SELECT start, end, correlationId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        by_corr[corr].append((k_start, k_end, name_id))
        all_kernels.append((k_start, k_end, names.get(name_id, "?")))

    rounds = []
    share = collections.Counter()
    calls = collections.Counter()
    classes = collections.Counter()
    for h_start, h_end, corr in launches:
        ks = sorted(by_corr.get(corr, []))
        if not ks:
            continue
        busy = 0
        cursor = ks[0][0]
        for k_start, k_end, name_id in ks:
            if k_end > cursor:
                busy += k_end - max(k_start, cursor)
                cursor = k_end
            name = names.get(name_id, "?").split("(")[0]
            share[name[-70:]] += k_end - k_start
            calls[name[-70:]] += 1
            classes[kernel_class(name)] += k_end - k_start
        span_end = max(k[1] for k in ks)
        rounds.append(
            {
                "launch": h_start,
                "launch_api": h_end - h_start,
                "first": ks[0][0],
                "last": span_end,
                "busy": busy,
                "span": span_end - ks[0][0],
                "kernels": len(ks),
            }
        )

    if not rounds:
        sys.exit("no graph launches with kernels")
    gaps = [b["first"] - a["last"] for a, b in zip(rounds, rounds[1:]) if b["first"] - a["last"] < 50_000_000]
    lat = [row["first"] - row["launch"] for row in rounds]
    n = len(rounds)
    print(f"{n} graph rounds, kernels per round {median([row['kernels'] for row in rounds]):.0f}")
    print(
        f"span  p50 {median([row['span'] for row in rounds]) / 1e3:8.1f} us   "
        f"busy p50 {median([row['busy'] for row in rounds]) / 1e3:8.1f} us   "
        f"idle inside p50 {median([row['span'] - row['busy'] for row in rounds]) / 1e3:6.1f} us"
    )
    print(
        f"gap to next round p50 {median(gaps) / 1e3:7.1f} us  "
        f"p90 {quantile(gaps, 0.9) / 1e3:7.1f}  p99 {quantile(gaps, 0.99) / 1e3:7.1f}  "
        f"max {max(gaps) / 1e3:8.1f}"
    )
    print(
        f"launch API p50 {median([row['launch_api'] for row in rounds]) / 1e3:6.1f} us; "
        f"launch to first kernel p50 {median(lat) / 1e3:6.1f} us"
    )
    total_busy = sum(row["busy"] for row in rounds)
    total_wall = rounds[-1]["last"] - rounds[0]["first"]
    print(
        f"GPU busy in graph rounds {100 * total_busy / total_wall:.1f}% of the decode wall "
        f"time ({total_wall / 1e9:.2f} s)"
    )
    print("\ntop kernels (us per round, share of round busy):")
    for name, ns in share.most_common(top):
        print(f"  {ns / n / 1e3:8.1f} us  {100 * ns / total_busy:5.1f}%  {calls[name] / n:5.1f}/round  {name}")

    print("\nkernel class (share of graph-round busy):")
    class_total = sum(classes.values()) or 1
    for name, ns in classes.most_common():
        print(f"  {name:10} {ns / n / 1e3:8.1f} us/round  {100 * ns / class_total:5.1f}%")

    ranges = load_ranges(db, names)
    attributed = attribute(all_kernels, ranges)
    print(f"\nNVTX ranges in the trace: {len(ranges)}")
    if not attributed:
        print(
            "no kernel time fell inside decode.mtp.target / decode.mtp.draft / mtp.proposal. "
            "Those ranges are host-side during CUDA graph capture and are not replayed."
        )
        return
    round_busy = median([row["busy"] for row in rounds])
    if sum(attributed.values()) < 2 * round_busy:
        print(
            "attributed NVTX time is about one capture, not one slice per replayed round. "
            "Use the kernel-class table for the steady decode split."
        )
    attr_total = sum(attributed.values()) or 1
    print("kernel time inside MTP NVTX ranges:")
    for label in RANGES:
        ns = attributed[label]
        print(f"  {label:20} {ns / 1e6:8.2f} ms total  {100 * ns / attr_total:5.1f}% of attributed")


if __name__ == "__main__":
    main()
