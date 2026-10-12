"""What happens between decode graph launches: runtime calls and GPU work per round.

  python prof_between.py <sqlite>
"""
import collections
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("SELECT id, value FROM StringIds"))
runtime = db.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start").fetchall()
launch_idx = [i for i, r in enumerate(runtime) if names.get(r[2], "").startswith("cudaGraphLaunch")]
if len(launch_idx) < 3:
    sys.exit("too few graph launches")
calls = collections.Counter()
host_ns = collections.Counter()
windows = []
for i0, i1 in zip(launch_idx, launch_idx[1:]):
    if runtime[i1][0] - runtime[i0][0] > 20_000_000:
        continue
    windows.append((runtime[i0][0], runtime[i1][0]))
    for start, end, name_id in runtime[i0 + 1:i1]:
        name = names.get(name_id, "?")
        calls[name] += 1
        host_ns[name] += end - start
rounds = len(windows)
print(f"{rounds} decode intervals (< 20 ms) between graph launches")
for name, n in calls.most_common(14):
    print(f"  {n / rounds:7.1f}/round  {host_ns[name] / rounds / 1e3:8.1f} us/round  {name}")


def inside(rows):
    out, w = [], 0
    for row in sorted(rows):
        while w < len(windows) and windows[w][1] <= row[0]:
            w += 1
        if w < len(windows) and windows[w][0] <= row[0] < windows[w][1]:
            out.append(row)
    return out


eager = inside(db.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId IS NULL").fetchall())
copies = inside(db.execute("SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchall())
print(f"\neager kernels in decode: {len(eager) / rounds:.1f}/round, {sum(e - s for s, e, _ in eager) / rounds / 1e3:.1f} us/round")
for name, n in collections.Counter(names.get(k, '?').split('(')[0][-60:] for _, _, k in eager).most_common(6):
    print(f"  {n / rounds:5.1f}/round {name}")
kinds = collections.defaultdict(list)
for s, e, kind, size in copies:
    kinds[kind].append((e - s, size))
for kind, rows in kinds.items():
    print(f"memcpy kind {kind}: {len(rows) / rounds:.1f}/round, {sum(r[0] for r in rows) / rounds / 1e3:.1f} us/round, "
          f"median {statistics.median(r[1] for r in rows):.0f} bytes")
