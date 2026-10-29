"""Fit Flash-Next MTP round time against draft count, and the 64k point.

  python tools/bench/round_cost.py --run
  python tools/bench/round_cost.py --json-dir D:\\01_PROJECTS\\PICOT\\tmp\\ninfer\\genre\\rounds

--run starts ninfer_bench once per draft count 1..7 (8k prefill, 256 output, lm-head draft)
and once more at draft auto for a 64k prefill plus 512 output. The GPU must be free.
The fit is round_ms(S) = a + b*S, with round_ms = acceptance_length / decode_tok_s * 1000.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO.parents[0] / "build-spo-dev" / "bench" / "ninfer_bench.exe"
DEFAULT_OUT = Path(os.environ.get("NINFER_GENRE_OUT", REPO.parents[1] / "tmp" / "ninfer" / "genre"))
MODEL = os.environ.get(
    "NINFER_MODEL",
    r"D:\99_LLM_MODELS\ninfer\qwen3_8_flash_next_125b_a6b_nvfp4_fp8proj_nvfp4mtp.ninfer",
)
CORPUS = os.environ.get("NINFER_CORPUS", r"D:\01_PROJECTS\PICOT\tmp\ninfer\corpus\common_mix.ids")


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


def load_case(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    walk(report, rows)
    if not rows:
        raise SystemExit(f"no decode row in {path}")
    row = rows[-1]
    spec = find(row, "speculative") or {}
    tok_s = row["decode_engine_tok_s_mean"]
    length = spec.get("acceptance_length")
    drafts = spec.get("draft_window")
    round_ms = None
    if length and tok_s:
        round_ms = 1000.0 * length / tok_s
    prefill = find(row, "prefill_tok_s_mean")
    return {
        "file": path.name,
        "drafts": drafts,
        "tok_s": tok_s,
        "acceptance_length": length,
        "acceptance_rate": spec.get("acceptance_rate"),
        "round_ms": round_ms,
        "prefill_tok_s": prefill,
        "prompt_tokens": find(row, "prompt_tokens"),
    }


def fit(points):
    usable = [item for item in points if item["drafts"] and item["round_ms"]]
    n = len(usable)
    if n < 2:
        return None
    sx = sum(item["drafts"] for item in usable)
    sy = sum(item["round_ms"] for item in usable)
    sxx = sum(item["drafts"] ** 2 for item in usable)
    sxy = sum(item["drafts"] * item["round_ms"] for item in usable)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return intercept, slope


def project(intercept, slope, drafts, tokens_per_round):
    ms = intercept + slope * drafts
    if ms <= 0:
        return None
    return 1000.0 * tokens_per_round / ms


def report(folder):
    paths = sorted(folder.glob("draft_*.json"))
    if not paths:
        sys.exit(f"no draft_*.json in {folder}")
    rows = [load_case(path) for path in paths]
    print("drafts  tok/s  tok/round  round_ms  prefill")
    for row in rows:
        ms = "n/a" if row["round_ms"] is None else f"{row['round_ms']:.2f}"
        length = "n/a" if row["acceptance_length"] is None else f"{row['acceptance_length']:.2f}"
        prefill = "n/a" if row["prefill_tok_s"] is None else f"{row['prefill_tok_s']:.0f}"
        print(f"{row['file']:16} {row['tok_s']:.1f}  {length:8}  {ms:8}  {prefill}")
    fitted = fit(rows)
    if fitted is None:
        print("not enough points to fit")
        return
    intercept, slope = fitted
    print(f"fit round_ms(S) = {intercept:.3f} + {slope:.3f} * S")
    at7 = next((row for row in rows if row["drafts"] == 7 and row["acceptance_length"]), None)
    if at7 is None:
        return
    rate = at7["acceptance_rate"] or 0
    length15 = at7["acceptance_length"] + 8 * rate
    tok7 = project(intercept, slope, 7, at7["acceptance_length"])
    tok15 = project(intercept, slope, 15, length15)
    print(
        f"at measured S=7 acceptance, holding the last-position accept rate {rate:.3f} out to S=15: "
        f"{tok7:.1f} tok/s at 7 vs {tok15:.1f} tok/s at 15"
    )
    out = folder / "round_fit.json"
    out.write_text(
        json.dumps(
            {
                "intercept_ms": intercept,
                "slope_ms": slope,
                "s7_tok_s": tok7,
                "s15_tok_s_if_tail_holds": tok15,
                "s7_acceptance_rate": rate,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


def run_one(out_dir, name, drafts, profile):
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{name}.json"
    if not BENCH.exists():
        sys.exit(f"missing {BENCH}")
    if not Path(CORPUS).exists():
        sys.exit(f"missing corpus {CORPUS}")
    command = [
        str(BENCH),
        "--weights",
        MODEL,
        "--corpus",
        CORPUS,
        "-pg",
        profile,
        "--max-ctx",
        "262144",
        "--prefill-chunk",
        "8192",
        "--kv-dtype",
        "bf16",
        "--warmup",
        "1",
        "-r",
        "1",
        "--numa-node",
        "1",
        "--spec",
        "mtp",
        "--draft-tokens",
        str(drafts),
        "--lm-head-draft",
        "-o",
        "json",
        "--output-file",
        str(output),
    ]
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["CUDA_MODULE_LOADING"] = "EAGER"
    env["NINFER_DECODE_WAIT"] = "spin"
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, env=env)
    if completed.returncode != 0:
        sys.exit(f"{name} failed with {completed.returncode}")


def run_all(out_dir):
    for drafts in range(1, 8):
        run_one(out_dir, f"draft_{drafts}", drafts, "8192,256")
    run_one(out_dir, "draft_auto_64k", "auto", "65536,512")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_OUT / "rounds")
    args = parser.parse_args()
    if args.run:
        run_all(args.json_dir)
    report(args.json_dir)


if __name__ == "__main__":
    main()
