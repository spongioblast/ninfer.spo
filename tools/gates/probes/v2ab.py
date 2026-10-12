"""Live logits A/B: prefill graphs off against on (V2 plan, recipe R3).

Each side gets a fresh dev server with a fresh logits dump directory. The same three chats are
played on both sides (fixed chat ids, thinking on), so both servers see the same requests in the
same order and run the same passes. The "on" side meets each pass length again in chats 2 and 3,
so those passes replay a captured graph. The script then compares the dumps and the transcripts.

  python v2ab.py --tag 1 [--turns 4] [--draft 3]

The MTP draft count is fixed (default 3): with `--draft-tokens auto` the window adapts during the
run, verification widths differ between the two servers, and greedy replies can split on a
near-tie for reasons unrelated to prefill graphs.
"""
import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
PY = sys.executable


def run(args, env=None):
    print(">", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=HERE, env=env, check=True)


def side(name, graphs, turns, draft):
    dumps = HERE / f"v2ab_{name}"
    if dumps.exists():
        raise SystemExit(f"{dumps} exists; use a new --tag")
    dumps.mkdir()
    # V2 builds read the old name, spo builds the NINFER_DEBUG_ one.
    env = dict(os.environ, NINFER_FLASH_NEXT_LOGITS_DIR=str(dumps), NINFER_DEBUG_LOGITS_DIR=str(dumps))
    if not graphs:
        env["NINFER_FLASH_NEXT_PREFILL_GRAPHS"] = "0"
    run(["serve_ctl.py", "restart", "--build", "build-dev", "--log", f"reqlog_v2ab_{name}.jsonl",
         "--", "--draft-tokens", draft], env)
    try:
        for chat in ("a", "b", "c"):
            run(["live_probe.py", "--turns", str(turns), "--thinking",
                 "--chat-id", f"v2abfixed{chat}", "--transcript", f"v2ab_{name}_{chat}.json"])
    finally:
        run(["serve_ctl.py", "stop"])
    run(["v2_graph_rows.py", f"reqlog_v2ab_{name}.jsonl", "--quiet"])
    return dumps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--draft", default="3", help="MTP draft tokens for both servers")
    args = parser.parse_args()
    off = side(f"off{args.tag}", False, args.turns, args.draft)
    on = side(f"on{args.tag}", True, args.turns, args.draft)
    for chat in ("a", "b", "c"):
        print(f"== transcript {chat}")
        run(["cmp_logits.py", str(off), str(on), f"v2ab_off{args.tag}_{chat}.json", f"v2ab_on{args.tag}_{chat}.json"])
    print("Pass: every prefill line says 'differing logits 0', every transcript is equal,")
    print("decode rows are all bit-equal, and the 'on' summary has hits > 0.")


if __name__ == "__main__":
    main()
