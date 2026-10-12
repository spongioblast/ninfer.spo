"""Interleaved served-sampling draft comparison.

  python served_draft.py --build ..\\build-spo-dev --out draft_served.jsonl

Pi samples, so a greedy bench is not the serving decision. This restarts ninfer-spo-serve
for each arm and measures decode tokens/s from the response timings at temperature 0.6.
The order is A B B A for every pair. Pairs are draft-tokens 4 against auto, and
--lm-head-draft against the default, at 4 drafts. A setting is better only when every
pair's decode rate improves and the ranges do not overlap. The script prints the table.
It does not change the server flags on its own; the decision is the printed verdict.
"""
import argparse
import json
import os
import statistics
import sys
import urllib.request

import serve_ctl

MODEL = "Qwen/Qwen3.8-Flash-Next"
PORT = int(os.environ.get("NINFER_PORT", "8013"))
PROMPT = "Explain how a condition variable wait can miss a notification, in one paragraph. " * 4


def one_request():
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 128,
            "temperature": 0.6,
            "top_p": 0.95,
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
    timings = data.get("timings") or {}
    return timings.get("predicted_per_second") or 0.0


def arm(build, name, setting):
    flags, env = setting
    serve_ctl.restart(build, f"reqlog_draft_{name}.jsonl", flags, env)
    rates = [one_request(), one_request()]
    print(f"{name}: {rates}")
    return rates


def pairs(build, left_name, left_flags, right_name, right_flags):
    # A B B A, two requests each, so four rates per side.
    sequence = (
        (left_name, left_flags),
        (right_name, right_flags),
        (right_name, right_flags),
        (left_name, left_flags),
    )
    left, right = [], []
    for name, flags in sequence:
        rates = arm(build, name, flags)
        (left if name == left_name else right).extend(rates)
    return left, right


def better(candidate, baseline):
    if not candidate or not baseline:
        return False
    if max(baseline) >= min(candidate):
        return False
    return statistics.mean(candidate) > statistics.mean(baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default="..\\build-spo-dev")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    # serve.cmd passes --lm-head-draft unless NINFER_LM_HEAD_DRAFT=0.
    four, auto = pairs(args.build, "draft4", (["--draft-tokens", "4"], {}),
                       "draft-auto", (["--draft-tokens", "auto"], {}))
    headed, plain = pairs(args.build, "lm-head", (["--draft-tokens", "4"], {}),
                          "no-lm-head", (["--draft-tokens", "4"], {"NINFER_LM_HEAD_DRAFT": "0"}))
    record = {
        "temperature": 0.6,
        "draft4": four,
        "auto": auto,
        "lm_head": headed,
        "no_lm_head": plain,
        "draft4_beats_auto": better(four, auto),
        "auto_beats_draft4": better(auto, four),
        "lm_head_beats_plain": better(headed, plain),
        "plain_beats_lm_head": better(plain, headed),
    }
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps(record) + "\n")
    print(json.dumps({key: record[key] for key in record if key.endswith("beats_auto") or "beats" in key}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
