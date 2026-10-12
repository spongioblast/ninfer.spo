"""Correctness of interleaved staged prefill (short request admitted during a long prefill).

  python hol_check.py together --out hol_together.json   (cold server)
  python hol_check.py alone    --out hol_alone.json      (restart the server first)
  python hol_check.py compare  hol_together.json hol_alone.json
  python hol_check.py cancel                              (cancel either side mid-prefill)

Every prompt starts with a fixed nonce, so nothing is shared between the cases and a cold
server computes every prompt from token 0 in both phases.
"""
import argparse
import http.client
import json
import socket
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from p8_probe import MODEL, URL, chunk_pool  # noqa: E402

POOL = chunk_pool()


def long_messages(nonce, start, chars):
    body, i = [], start
    while sum(len(b) for b in body) < chars:
        body.append(POOL[i % len(POOL)][1])
        i += 1
    return [
        {"role": "system", "content": f"Run {nonce}. You are a careful reader."},
        {"role": "user", "content": "\n\n".join(body) + "\n\nSummarize the text above in three sentences."},
    ]


def short_messages(nonce, question):
    return [
        {"role": "system", "content": f"Run {nonce}. Answer briefly."},
        {"role": "user", "content": question},
    ]


CASES = {
    "long_a": lambda: (long_messages("L-a-7f3c", 4000, 400_000), 48, False),
    "short_a": lambda: (short_messages("S-a-91be", "Explain what a hash map is and when to avoid one."), 96, False),
    "long_b": lambda: (long_messages("L-b-22d0", 9000, 240_000), 48, True),
    "short_b": lambda: (short_messages("S-b-5a17", "Write a Python function that reverses words in a sentence."), 160, True),
    "mid_a": lambda: (long_messages("M-a-0c44", 20000, 120_000), 32, False),
}
COMPARED = ("long_a", "short_a", "long_b", "short_b")


def run(messages, max_tokens, thinking, abort_after=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": thinking,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    target = urllib.parse.urlsplit(URL)
    conn = http.client.HTTPConnection(target.hostname, target.port, timeout=3600)
    started = time.perf_counter()
    first, content, reasoning, final = None, [], [], None
    conn.connect()
    timer = None
    if abort_after is not None:
        timer = threading.Timer(abort_after, lambda: conn.sock.shutdown(socket.SHUT_RDWR))
        timer.start()
    try:
        conn.request("POST", target.path, body=json.dumps(payload).encode(),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
    except OSError:
        resp = None
    try:
        while True:
            try:
                line = resp.readline() if resp is not None else b""
            except (OSError, http.client.HTTPException):
                line = b""
            if timer is not None and not timer.is_alive() and time.perf_counter() - started >= abort_after:
                return {"aborted": True, "after_s": round(time.perf_counter() - started, 3)}
            if not line:
                break
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("timings") or chunk.get("usage"):
                final = chunk
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta", {})
                if (delta.get("content") or delta.get("reasoning_content")) and first is None:
                    first = time.perf_counter() - started
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
    finally:
        if timer is not None:
            timer.cancel()
        conn.close()
    timings = (final or {}).get("timings") or {}
    return {
        "ttft_ms": round((first or 0.0) * 1000.0, 1),
        "total_s": round(time.perf_counter() - started, 3),
        "prompt_n": timings.get("prompt_n"),
        "cache_n": timings.get("cache_n"),
        "content": "".join(content),
        "reasoning": "".join(reasoning),
    }


def background(name, results, delay=0.0, abort_after=None):
    def work():
        time.sleep(delay)
        messages, max_tokens, thinking = CASES[name]()
        results[name] = run(messages, max_tokens, thinking, abort_after)

    thread = threading.Thread(target=work)
    thread.start()
    return thread


def together(out):
    results = {}
    for long_name, short_name in (("long_a", "short_a"), ("long_b", "short_b")):
        threads = [background(long_name, results), background(short_name, results, delay=1.5)]
        for thread in threads:
            thread.join()
        print(long_name, results[long_name]["ttft_ms"], short_name, results[short_name]["ttft_ms"], flush=True)
    json.dump(results, open(out, "w", encoding="utf-8"), indent=1)


def custom(spec, out):
    """spec like 'short_a@0,long_a@0.02;long_b@0,short_b@1.5': groups run one after another."""
    results = {}
    for group in spec.split(";"):
        threads = []
        for item in group.split(","):
            name, delay = item.split("@")
            threads.append(background(name, results, delay=float(delay)))
        for thread in threads:
            thread.join()
        print(group, {k: results[k]["ttft_ms"] for k in results}, flush=True)
    json.dump(results, open(out, "w", encoding="utf-8"), indent=1)


def alone(out):
    results = {}
    for name in ("long_a", "short_a", "long_b", "short_b"):
        messages, max_tokens, thinking = CASES[name]()
        results[name] = run(messages, max_tokens, thinking)
        print(name, results[name]["ttft_ms"], flush=True)
    json.dump(results, open(out, "w", encoding="utf-8"), indent=1)


def compare(a_path, b_path):
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    bad = 0
    for name in COMPARED:
        if name not in a or name not in b:
            continue
        same = a[name]["content"] == b[name]["content"] and a[name]["reasoning"] == b[name]["reasoning"]
        bad += not same
        print(f"{name:8} {'SAME' if same else 'DIFF'} prompt_n {a[name]['prompt_n']}/{b[name]['prompt_n']} "
              f"cache_n {a[name]['cache_n']}/{b[name]['cache_n']} ttft {a[name]['ttft_ms']}/{b[name]['ttft_ms']} ms")
        if not same:
            print("  A:", repr((a[name]["reasoning"] + "|" + a[name]["content"])[:300]))
            print("  B:", repr((b[name]["reasoning"] + "|" + b[name]["content"])[:300]))
    sys.exit(1 if bad else 0)


def cancel():
    def nonce(name, tag):
        messages, max_tokens, thinking = CASES[name]()
        messages[0]["content"] = messages[0]["content"].replace("Run ", f"Run {tag}-")
        return messages, max_tokens, thinking

    # c1: the long prompt is cancelled mid-prefill while a short request runs beside it.
    # c2: a 30k prompt is cancelled mid-prefill while the long prompt keeps prefilling.
    for tag, first, second, abort_first, abort_second in (
        ("c1", "long_a", "short_a", 1.8, None),
        ("c2", "long_a", "mid_a", None, 0.4),
    ):
        results = {}

        def go(name, delay, abort_after):
            time.sleep(delay)
            messages, max_tokens, thinking = nonce(name, tag)
            results[name] = run(messages, max_tokens, thinking, abort_after)

        threads = [
            threading.Thread(target=go, args=(first, 0.0, abort_first)),
            threading.Thread(target=go, args=(second, 1.5, abort_second)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print(tag, {k: (v.get("aborted"), v.get("ttft_ms"), v.get("total_s")) for k, v in results.items()}, flush=True)
    print("after cancel:", run(short_messages("after-cancel", "Say OK."), 8, False)["content"][:40])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["together", "alone", "compare", "cancel", "custom"])
    parser.add_argument("files", nargs="*")
    parser.add_argument("--out")
    parser.add_argument("--spec")
    args = parser.parse_args()
    if args.mode == "custom":
        custom(args.spec, args.out)
    elif args.mode == "together":
        together(args.out)
    elif args.mode == "alone":
        alone(args.out)
    elif args.mode == "compare":
        compare(*args.files)
    else:
        cancel()


if __name__ == "__main__":
    main()
