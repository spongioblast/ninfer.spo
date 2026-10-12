"""Cancellation probe for a running ninfer-spo-serve.

  python cancel_probe.py --request-log reqlog.jsonl --out cancel.jsonl

Arms are {live, abandoned} x {prefill, generation}, plus one short idle request after the
abandoned prefill. Each arm uses a different cold prompt. An abandoned client closes with
SO_LINGER 0 (RST). Abandoned prefill waits until the request log shows request_start, so
the reset happens after the server has the body. An arm is cancelled when the request log
shows computed_prefill_tokens < prompt_tokens. The idle arm's client time is the follower
wait. Exit 1 when an abandoned prefill was not cancelled, or when the follower waited
about as long as the live prefill.
"""
import argparse
import json
import os
import pathlib
import socket
import struct
import time
import urllib.request

HOST = "127.0.0.1"
PORT = int(os.environ.get("NINFER_PORT", "8013"))
MODEL = "Qwen/Qwen3.8-Flash-Next"


def body(user, max_tokens):
    return json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "enable_thinking": False,
            "stream": True,
        }
    ).encode()


def request_bytes(payload):
    head = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()
    return head + payload


def wait_event(path, after_ms, event, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") != event:
                    continue
                if row.get("timestamp_unix_ms", 0) <= after_ms:
                    continue
                return row
        time.sleep(0.05)
    return None


def abandon(payload, after_token, log_path, after_ms):
    sock = socket.create_connection((HOST, PORT), timeout=30)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.sendall(request_bytes(payload))
    if after_token:
        sock.settimeout(30)
        seen = b""
        deadline = time.time() + 180
        while time.time() < deadline and b"data: " not in seen:
            chunk = sock.recv(4096)
            if not chunk:
                break
            seen += chunk
    elif wait_event(log_path, after_ms, "request_start", 120) is None:
        sock.close()
        return
    sock.close()


def live(payload):
    started = time.perf_counter()
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        resp.read()
    return time.perf_counter() - started


def next_done(path, after_ms):
    deadline = time.time() + 240
    while time.time() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") != "request_done":
                    continue
                if row.get("timestamp_unix_ms", 0) <= after_ms:
                    continue
                return row
        time.sleep(0.5)
    return None


def log_offset(path):
    if not path.exists():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    stamps = [row.get("timestamp_unix_ms", 0) for row in rows]
    return max(stamps) if stamps else 0


def prefill_text(tag):
    sentence = f" {tag} the lane keeps its place in the queue while the prompt is still arriving."
    return (sentence * 4000).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--request-log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    log_path = pathlib.Path(args.request_log)
    records = []
    live_prefill_s = None
    idle_s = None

    def arm(name, payload, abandoned, after_token=False):
        nonlocal live_prefill_s, idle_s
        stamp = log_offset(log_path)
        started = time.perf_counter()
        if abandoned:
            abandon(payload, after_token, log_path, stamp)
        else:
            elapsed = live(payload)
            if name == "live-prefill":
                live_prefill_s = elapsed
            if name == "idle":
                idle_s = elapsed
        done = next_done(log_path, stamp)
        result = (done or {}).get("result") or {}
        prompt = result.get("prompt_tokens") or 0
        computed = result.get("computed_prefill_tokens")
        cancelled = computed is not None and prompt and computed < prompt
        record = {
            "arm": name,
            "client_s": round(time.perf_counter() - started, 3),
            "prompt_tokens": prompt,
            "computed_prefill_tokens": computed,
            "cancelled": bool(cancelled),
            "logged": done is not None,
        }
        records.append(record)
        print(f"{name}: client {record['client_s']} s, computed {computed} / prompt {prompt}, cancelled {record['cancelled']}")

    arm("live-prefill", body(prefill_text("live-prefill"), 8), False)
    arm("abandoned-prefill", body(prefill_text("abandoned-prefill"), 8), True)
    arm("idle", body("Reply with the single word idle.", 8), False)
    arm("live-generation", body("Count from one to twenty, one number per line. live-generation.", 64), False)
    arm("abandoned-generation", body("Count from one to twenty, one number per line. abandoned-generation.", 128), True, True)

    with open(args.out, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")
    failed = False
    for record in records:
        if not record["logged"]:
            print(f"{record['arm']} has no request_done row in the request log")
            failed = True
    abandoned = next(record for record in records if record["arm"] == "abandoned-prefill")
    if not abandoned["cancelled"]:
        print("abandoned prefill was not cancelled")
        failed = True
    if live_prefill_s and idle_s and idle_s > live_prefill_s * 0.5:
        print(f"follower wait {idle_s:.1f} s is not in the idle band (live prefill {live_prefill_s:.1f} s)")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
