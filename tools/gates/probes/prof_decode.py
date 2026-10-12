"""Decode rounds in an nsys CUDA trace: one round per cudaGraphLaunch.

  python prof_decode.py <sqlite> [top kernels, default 15]

Per round: GPU span (first kernel start to last kernel end), busy (union of kernel time),
idle inside the round, and the gap from the round's last kernel to the next round's first.
Work between graph launches (host copies, small eager kernels) counts toward that gap.
"""
import collections
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
names = dict(db.execute("SELECT id, value FROM StringIds"))
launch_ids = {i for i, v in names.items() if v.startswith("cudaGraphLaunch")}
launches = [(s, e, c) for s, e, n, c in db.execute(
    "SELECT start, end, nameId, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start") if n in launch_ids]
by_corr = collections.defaultdict(list)
for k_start, k_end, corr, name_id in db.execute(
        "SELECT start, end, correlationId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
    by_corr[corr].append((k_start, k_end, name_id))
other = sorted((s, e) for s, e in db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY"))

rounds = []
share = collections.Counter()
calls = collections.Counter()
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
    span_end = max(k[1] for k in ks)
    rounds.append({"launch": h_start, "launch_api": h_end - h_start, "first": ks[0][0], "last": span_end,
                   "busy": busy, "span": span_end - ks[0][0], "kernels": len(ks)})

if not rounds:
    sys.exit("no graph launches with kernels")
gaps = [b["first"] - a["last"] for a, b in zip(rounds, rounds[1:]) if b["first"] - a["last"] < 50_000_000]
lat = [r["first"] - r["launch"] for r in rounds]
med = lambda xs: statistics.median(xs)
q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
n = len(rounds)
print(f"{n} graph rounds, kernels per round {med([r['kernels'] for r in rounds]):.0f}")
print(f"span  p50 {med([r['span'] for r in rounds]) / 1e3:8.1f} us   busy p50 {med([r['busy'] for r in rounds]) / 1e3:8.1f} us   "
      f"idle inside p50 {med([r['span'] - r['busy'] for r in rounds]) / 1e3:6.1f} us")
print(f"gap to next round p50 {med(gaps) / 1e3:7.1f} us  p90 {q(gaps, 0.9) / 1e3:7.1f}  p99 {q(gaps, 0.99) / 1e3:7.1f}  max {max(gaps) / 1e3:8.1f}")
print(f"launch API p50 {med([r['launch_api'] for r in rounds]) / 1e3:6.1f} us; launch to first kernel p50 {med(lat) / 1e3:6.1f} us")
total_busy = sum(r["busy"] for r in rounds)
total_wall = rounds[-1]["last"] - rounds[0]["first"]
print(f"GPU busy in graph rounds {100 * total_busy / total_wall:.1f}% of the decode wall time ({total_wall / 1e9:.2f} s)")
print(f"\ntop kernels (us per round, share of round busy):")
for name, ns in share.most_common(top):
    print(f"  {ns / n / 1e3:8.1f} us  {100 * ns / total_busy:5.1f}%  {calls[name] / n:5.1f}/round  {name}")
