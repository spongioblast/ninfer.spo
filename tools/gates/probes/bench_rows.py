"""Decode rows from ninfer_bench JSON reports.

  python bench_rows.py file.json [file.json ...]

Prints, per file, one entry per -pg case in command-line order: engine decode tok/s, MTP acceptance.
"""
import json
import sys


def walk(node, out):
    if isinstance(node, dict):
        if "decode_engine_tok_s_mean" in node:
            out.append(node)
        for value in node.values():
            walk(value, out)
    elif isinstance(node, list):
        for value in node:
            walk(value, out)


def find(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            hit = find(value, key)
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for value in node:
            hit = find(value, key)
            if hit is not None:
                return hit
    return None


for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    rows = []
    walk(report, rows)
    parts = []
    for index, row in enumerate(rows, 1):
        accept = find(row, "acceptance_rate")
        tps = row["decode_engine_tok_s_mean"]
        std = row.get("decode_engine_tok_s_stddev", 0)
        acc = f" acc {accept:.2f}" if isinstance(accept, (int, float)) else ""
        parts.append(f"case {index}: {tps:.1f} +/- {std:.1f} tok/s{acc}")
    print(f"{path}: " + " | ".join(parts))
