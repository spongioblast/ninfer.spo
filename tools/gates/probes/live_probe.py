"""One live chat: short user turns, thinking off, streamed; prints client TTFT per turn.

  python live_probe.py [--turns 12] [--wait-ready]
"""
import argparse
import http.client
import json
import socket
import sys
import time
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from p8_probe import MODEL, URL  # noqa: E402

QUESTIONS = [
    "What is a hash map?", "And when is a tree map better?", "Give one example in Python.",
    "What about Rust?", "How does resizing work?", "What is the load factor?",
    "Name a common hash function.", "Is it secure?", "What is a collision?",
    "How does open addressing differ?", "Which one is faster in practice?", "Thanks, summarize.",
]


def wait_ready():
    for _ in range(600):
        try:
            urllib.request.urlopen(URL.replace("/v1/chat/completions", "/v1/models"), timeout=2)
            return
        except Exception:
            time.sleep(1)
    sys.exit("server not ready")


THINKING = False


CONNECTION = None


def connection():
    # A Python connect with a timeout waits a full Windows timer tick (13 ms) on loopback; connect
    # blocking, then set the timeout, and keep the connection alive like a real client.
    global CONNECTION
    if CONNECTION is None:
        parts = urllib.parse.urlsplit(URL)
        CONNECTION = http.client.HTTPConnection(parts.hostname, parts.port)
        CONNECTION.sock = socket.create_connection((parts.hostname, parts.port))
        CONNECTION.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        CONNECTION.sock.settimeout(120)
    return CONNECTION


def turn(messages):
    payload = {"model": MODEL, "messages": messages, "max_tokens": 160 if THINKING else 48,
               "temperature": 0, "enable_thinking": THINKING, "stream": True}
    body = json.dumps(payload).encode()
    start = time.perf_counter()
    first = event = None
    text, reasoning = [], []
    client = connection()
    client.request("POST", urllib.parse.urlsplit(URL).path, body,
                   {"Content-Type": "application/json"})
    with client.getresponse() as response:
        head = time.perf_counter() - start
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            if event is None:
                event = time.perf_counter() - start
            delta = json.loads(line[5:])["choices"][0].get("delta", {})
            piece = delta.get("content") or ""
            thought = delta.get("reasoning_content") or ""
            if (piece or thought) and first is None:
                first = time.perf_counter() - start
            text.append(piece)
            reasoning.append(thought)
    message = {"role": "assistant", "content": "".join(text)}
    if THINKING:
        message["reasoning_content"] = "".join(reasoning)
    return (head, event, first or 0.0), time.perf_counter() - start, message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--wait-ready", action="store_true")
    parser.add_argument("--chat-id", help="fixed chat id instead of a random one (repeatable runs)")
    parser.add_argument("--transcript", help="write the replies here")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between turns")
    args = parser.parse_args()
    global THINKING
    THINKING = args.thinking
    if args.wait_ready:
        wait_ready()
    chat_id = args.chat_id or uuid.uuid4().hex
    messages = [{"role": "system", "content": f"Chat {chat_id}. Answer in one or two sentences."}]
    for question in QUESTIONS[: args.turns]:
        time.sleep(args.pause)
        messages.append({"role": "user", "content": question})
        (head, event, first), total, message = turn(messages)
        messages.append(message)
        print(f"ttft {first * 1000:7.1f} ms  head {head * 1000:6.1f}  event {event * 1000:6.1f}  "
              f"total {total * 1000:7.1f} ms  {question}", flush=True)
    if args.transcript:
        with open(args.transcript, "w", encoding="utf-8") as handle:
            json.dump(messages, handle, indent=1)


if __name__ == "__main__":
    main()
