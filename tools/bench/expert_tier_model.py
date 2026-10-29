"""Predict Flash-Next decode with a VRAM expert budget, and write hot-expert profiles.

Reads NINFER_ROUTING_DUMP files. Each line is one MoE call:

    call router_address tokens id id ... id

The router address names the layer. The first 48 addresses seen (the first prefill walks
every main layer in order) are layers 0..47; any other address is the MTP layer and is
skipped, since the MTP bank always stays in VRAM. A window is one pass over layers 0..47;
windows routing more than 16 tokens are prefill and are skipped.

    python tools/bench/expert_tier_model.py eval.txt --rank train1.txt train2.txt \
        --pcie-gbps 25 --write-profile hot.json --profile-gib 21

Without --rank the hot set is ranked on the evaluation dump itself (in-sample, optimistic).
Miss bytes are reported two ways: per assignment (the decode route kernel reads one expert
copy per token that routes to it) and per unique expert (a staging copy would read it once).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

LAYERS = 48
EXPERTS = 512
PAIR_BYTES = 1_843_200 + 921_600  # gate_up codes+scales, down codes+scales
DECODE_TOKENS = 16


def read_windows(path):
    order = {}
    windows = []
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        router, tokens = parts[1], int(parts[2])
        if router not in order and len(order) < LAYERS:
            order[router] = len(order)
        layer = order.get(router)
        if layer is None:
            continue
        if layer == 0:
            current = {"tokens": tokens, "layers": []}
            windows.append(current)
        if current is None or len(current["layers"]) != layer:
            current = None
            continue
        current["layers"].append([int(x) for x in parts[3:]])
    return [w for w in windows if len(w["layers"]) == LAYERS and w["tokens"] <= DECODE_TOKENS]


def rank(windows, slots):
    counts = Counter()
    for window in windows:
        for layer, ids in enumerate(window["layers"]):
            for expert in ids:
                counts[(layer, expert)] += 1
    chosen = [key for key, _ in counts.most_common(slots)]
    # Unseen experts fill the rest in expert-id order, as the loader does.
    seen = set(chosen)
    for expert in range(EXPERTS):
        for layer in range(LAYERS):
            if len(chosen) >= slots:
                return chosen
            if (layer, expert) not in seen:
                chosen.append((layer, expert))
                seen.add((layer, expert))
    return chosen


def replay(windows, hot):
    hits = misses = 0
    assign_bytes = []
    unique_bytes = []
    tokens = []
    for window in windows:
        per_assign = per_unique = 0
        for layer, ids in enumerate(window["layers"]):
            for expert in ids:
                if (layer, expert) in hot:
                    hits += 1
                else:
                    misses += 1
                    per_assign += PAIR_BYTES
            per_unique += PAIR_BYTES * len({e for e in ids if (layer, e) not in hot})
        assign_bytes.append(per_assign)
        unique_bytes.append(per_unique)
        tokens.append(window["tokens"])
    return hits, misses, assign_bytes, unique_bytes, tokens


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", nargs="+", help="evaluation dumps")
    parser.add_argument("--rank", nargs="*", default=None, help="dumps to rank the hot set on")
    parser.add_argument("--budgets", nargs="*", type=int, default=[21, 13])
    parser.add_argument("--pcie-gbps", type=float, default=25.0)
    parser.add_argument("--ram-gbps", type=float, default=60.0)
    parser.add_argument("--write-profile", default="")
    parser.add_argument("--profile-gib", type=int, default=0,
                        help="write this many GiB of slots (0 writes all seen experts)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    evaluate = [w for path in args.dump for w in read_windows(path)]
    train = evaluate if args.rank is None else [w for p in args.rank for w in read_windows(p)]
    report = {
        "eval_windows": len(evaluate),
        "rank_windows": len(train),
        "in_sample": args.rank is None,
        "mean_window_tokens": round(mean([w["tokens"] for w in evaluate]), 2),
        "budgets": {},
    }
    for gib in args.budgets:
        slots = int((gib << 30) / PAIR_BYTES)
        hot = set(rank(train, slots))
        hits, misses, assign_bytes, unique_bytes, _ = replay(evaluate, hot)
        total = hits + misses
        report["budgets"][f"{gib}GiB"] = {
            "slots": slots,
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "miss_mib_per_window_assign": round(mean(assign_bytes) / 2**20, 1),
            "miss_mib_per_window_unique": round(mean(unique_bytes) / 2**20, 1),
            "pcie_ms_per_window_assign": round(mean(assign_bytes) / 1e6 / args.pcie_gbps, 2),
            "pcie_ms_per_window_unique": round(mean(unique_bytes) / 1e6 / args.pcie_gbps, 2),
            "ram_ms_per_window_unique": round(mean(unique_bytes) / 1e6 / args.ram_gbps, 2),
        }
    if args.write_profile:
        slots = int((args.profile_gib << 30) / PAIR_BYTES) if args.profile_gib else LAYERS * EXPERTS
        counts = Counter()
        for window in train:
            for layer, ids in enumerate(window["layers"]):
                for expert in ids:
                    counts[(layer, expert)] += 1
        pairs = [[layer, expert] for (layer, expert), _ in counts.most_common(slots)]
        Path(args.write_profile).write_text(json.dumps(pairs), encoding="utf-8")
        report["profile"] = {"path": args.write_profile, "pairs": len(pairs)}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
