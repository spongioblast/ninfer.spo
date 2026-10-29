"""Flash-Next genre bench on the hyperburst held-out prompts.

  python tools/bench/genre_bench.py --suite short --sampling both
  python tools/bench/genre_bench.py --suite long --sampling both --restart

Reads prompts from the hyperburst reference checkout. Does not copy them.
Streams /v1/chat/completions. Client decode rate is
(completion_tokens - 1) / (time of last content chunk - time of first).
JSONL goes to the workspace tmp/ninfer/genre/.

--sampling pi sends no temperature, top_k, or top_p, plus a fixed seed.
--sampling greedy sends temperature 0 and the same seed.
One unmeasured 8-token request warms the server before the suite.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETS = REPO.parents[0] / "references" / "hyperburst-rtxpro6000" / "bench" / "workloads" / "sets"
OUT_DIR = Path(os.environ.get("NINFER_GENRE_OUT", REPO.parents[1] / "tmp" / "ninfer" / "genre"))
MODEL = "Qwen/Qwen3.8-Flash-Next"
PORT = int(os.environ.get("NINFER_PORT", "8013"))
SEED = 1

PI_TURNS = (
    ("pi-condvar", "Explain how a condition variable wait can miss a notification, in one paragraph."),
    ("pi-diff", "In one paragraph, how does a unified diff decide which lines are context?"),
    ("pi-json", "Write a short JSON object with keys city and population for Tokyo. No preamble."),
    ("pi-bug", "A loop prints one past the last item. Name the bug in one sentence and the fix in one sentence."),
)


def load_genre(name):
    folder = SETS / f"{name}-v1"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    cases = []
    for prompt in manifest["prompts"]:
        text = (folder / prompt["file"]).read_text(encoding="utf-8")
        cases.append(
            {
                "id": prompt["id"],
                "genre": manifest["domain"],
                "prompt": text,
                "max_tokens": prompt["max_tokens"],
                "shape": "long",
            }
        )
    return cases


def short_cases():
    cases = []
    for name, text in PI_TURNS:
        for cap in (128, 256):
            for thinking in (False, True):
                cases.append(
                    {
                        "id": f"{name}-{cap}-think{int(thinking)}",
                        "genre": "short",
                        "prompt": text,
                        "max_tokens": cap,
                        "shape": "short",
                        "thinking": thinking,
                    }
                )
    for genre in ("code-edit", "prose-en", "prose-ja", "agent-loop"):
        base = load_genre(genre)[0]
        for cap in (128, 256):
            for thinking in (False, True):
                cases.append(
                    {
                        "id": f"{base['id']}-short-{cap}-think{int(thinking)}",
                        "genre": genre,
                        "prompt": base["prompt"],
                        "max_tokens": cap,
                        "shape": "short",
                        "thinking": thinking,
                    }
                )
    return cases


def select_cases(suite):
    if suite == "short":
        return short_cases()
    genres = {
        "long": ("code-edit", "prose-en", "prose-ja", "agent-loop"),
        "coding": ("code-edit", "agent-loop"),
    }[suite]
    cases = []
    for genre in genres:
        for case in load_genre(genre):
            case["thinking"] = True
            cases.append(case)
    return cases


def body(case, sampling):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": case["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": case.get("thinking", True),
        "seed": SEED,
    }
    if sampling == "greedy":
        payload["temperature"] = 0
    return payload


def stream(payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    first = None
    last = None
    usage = {}
    timings = {}
    pieces = []
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            now = time.perf_counter()
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            blob = line[5:].strip()
            if blob == "[DONE]":
                break
            event = json.loads(blob)
            if event.get("usage"):
                usage = event["usage"]
            if event.get("timings"):
                timings = event["timings"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                pieces.append(delta.get("reasoning_content") or "")
                pieces.append(delta.get("content") or "")
                if first is None:
                    first = now
                last = now
    return started, first, last, usage, timings, "".join(pieces)


def one(case, sampling, log_path):
    before = log_path.stat().st_size if log_path and log_path.exists() else None
    started, first, last, usage, timings, text = stream(body(case, sampling))
    completion = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
    client = None
    if first is not None and last is not None and last > first and completion > 1:
        client = (completion - 1) / (last - first)
    speculative = read_speculative(log_path, before)
    rounds = speculative.get("rounds")
    accepted = speculative.get("accepted_tokens")
    per_round = None
    if rounds:
        per_round = 1.0 + (accepted or 0) / rounds
    return {
        "id": case["id"],
        "genre": case["genre"],
        "shape": case["shape"],
        "sampling": sampling,
        "thinking": case.get("thinking", True),
        "max_tokens": case["max_tokens"],
        "completion_tokens": completion,
        "client_tok_s": client,
        "ttft_s": None if first is None else first - started,
        "server_tok_s": timings.get("predicted_per_second"),
        "prompt_tok_s": timings.get("prompt_per_second"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "rounds": rounds,
        "tokens_per_round": per_round,
        "accepted_per_position": speculative.get("accepted_per_position"),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


def read_speculative(log_path, before):
    if log_path is None or not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if before is not None:
        text = text[before:]
    last = None
    for line in text.splitlines():
        line = line.strip()
        if line:
            last = line
    if not last:
        return {}
    try:
        record = json.loads(last)
    except json.JSONDecodeError:
        return {}
    return record.get("speculative") or {}


def warmup():
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with the single word ready."}],
        "max_tokens": 8,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900):
        return


def mean_band(rows):
    values = [row["client_tok_s"] for row in rows if row["client_tok_s"]]
    if not values:
        return None
    return sum(values) / len(values), min(values), max(values)


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["shape"], row["genre"], row["sampling"], row["thinking"])
        groups.setdefault(key, []).append(row)
    print("shape genre sampling thinking  n  client tok/s (min-max)  tok/round  ttft_ms")
    for key in sorted(groups):
        group = groups[key]
        band = mean_band(group)
        rounds = [row["tokens_per_round"] for row in group if row["tokens_per_round"]]
        ttft = [row["ttft_s"] for row in group if row["ttft_s"] is not None]
        rate = "n/a" if band is None else f"{band[0]:.1f} ({band[1]:.1f}-{band[2]:.1f})"
        per = "n/a" if not rounds else f"{sum(rounds) / len(rounds):.2f}"
        first = "n/a" if not ttft else f"{1000 * sum(ttft) / len(ttft):.0f}"
        shape, genre, sampling, thinking = key
        print(f"{shape:5} {genre:12} {sampling:7} think={int(thinking)}  {len(group):2}  {rate:22}  {per:8}  {first}")


def restart(log_name):
    probes = REPO / "tools" / "gates" / "probes"
    sys.path.insert(0, str(probes))
    import serve_ctl

    os.environ.setdefault(
        "NINFER_MODEL",
        r"D:\99_LLM_MODELS\ninfer\qwen3_8_flash_next_125b_a6b_nvfp4_fp8proj_nvfp4mtp.ninfer",
    )
    serve_ctl.restart("..\\build-spo-release", log_name, [])
    return serve_ctl.OUT / log_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("short", "long", "coding"), required=True)
    parser.add_argument("--sampling", choices=("greedy", "pi", "both"), default="both")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--log", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-genre", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0, help="cap every case's output")
    parser.add_argument("--genre-offset", type=int, default=0,
                        help="skip the first N prompts of each genre")
    args = parser.parse_args()
    log_path = None
    if args.restart:
        log_path = restart(f"genre_{args.suite}.jsonl")
    elif args.log:
        log_path = Path(args.log)
    try:
        warmup()
    except urllib.error.URLError as error:
        sys.exit(f"server not reachable on {PORT}: {error}")
    cases = select_cases(args.suite)
    if args.per_genre or args.genre_offset:
        taken = {}
        kept = []
        for case in cases:
            taken[case["genre"]] = taken.get(case["genre"], 0) + 1
            index = taken[case["genre"]] - args.genre_offset
            if index >= 1 and (not args.per_genre or index <= args.per_genre):
                kept.append(case)
        cases = kept
    if args.max_tokens:
        for case in cases:
            case["max_tokens"] = min(case["max_tokens"], args.max_tokens)
    if args.limit:
        cases = cases[: args.limit]
    samplings = ("greedy", "pi") if args.sampling == "both" else (args.sampling,)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.suite}_{args.sampling}.jsonl"
    rows = []
    with out_path.open("w", encoding="utf-8") as handle:
        for sampling in samplings:
            for index, case in enumerate(cases, 1):
                print(f"{sampling} {index}/{len(cases)} {case['id']}", flush=True)
                row = one(case, sampling, log_path)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                rate = row["client_tok_s"]
                print(
                    f"  client {0 if rate is None else rate:.1f} tok/s  "
                    f"ttft {0 if row['ttft_s'] is None else 1000 * row['ttft_s']:.0f} ms  "
                    f"per_round {row['tokens_per_round']}",
                    flush=True,
                )
    print(f"wrote {out_path}")
    summarize(rows)


if __name__ == "__main__":
    main()
