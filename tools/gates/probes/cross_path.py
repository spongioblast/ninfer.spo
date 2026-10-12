"""Cold and shared-prefix continuations must produce the same bytes.

  python cross_path.py --build ..\\build-spo-dev --out cross.jsonl

Restarts ninfer-spo-serve (port NINFER_PORT, default 8013). First a fresh server answers
continuation B (shared system prompt S plus user U2) at temperature 0. Then the server
restarts, answers A (S plus a different user), then B again. B warm must equal B cold byte
for byte, and its cache_n must be at least the token length of S. An exact repeat of one
prompt is not used: that would take response replay and never decode.
"""
import argparse
import json
import os
import sys
import urllib.request

import serve_ctl

MODEL = "Qwen/Qwen3.8-Flash-Next"
SYSTEM = ("You are a coding agent. The tools below stay on every turn. " * 500).strip()
PORT = int(os.environ.get("NINFER_PORT", "8013"))


def token_count(text):
    tokenizer = os.environ.get("NINFER_TOKENIZER", "")
    if not tokenizer:
        raise SystemExit("set NINFER_TOKENIZER to the tokenizer directory")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    return len(tok.encode(text, add_special_tokens=False))


def complete(user):
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "max_tokens": 64,
            "temperature": 0,
            "enable_thinking": False,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode())
    message = data["choices"][0]["message"]
    text = message.get("content") or ""
    cache_n = (data.get("timings") or {}).get("cache_n") or 0
    return text, cache_n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default="..\\build-spo-dev")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    system_tokens = token_count(SYSTEM)
    serve_ctl.restart(args.build, "reqlog_cross_cold.jsonl", [])
    cold, cold_cache = complete("Summarise the tools in one sentence. cold-b")
    serve_ctl.restart(args.build, "reqlog_cross_warm.jsonl", [])
    complete("Name the first tool only. warm-a")
    warm, warm_cache = complete("Summarise the tools in one sentence. cold-b")
    same = cold == warm
    reused = warm_cache >= system_tokens
    record = {
        "system_tokens": system_tokens,
        "cold_cache_n": cold_cache,
        "warm_cache_n": warm_cache,
        "byte_equal": same,
        "reused_system": reused,
        "cold": cold,
        "warm": warm,
    }
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps(record) + "\n")
    print(f"byte_equal {same}, warm cache_n {warm_cache}, system tokens {system_tokens}")
    if not same or not reused:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
