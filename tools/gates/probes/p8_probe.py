"""Pi-shaped TTFT and session-switching probe for ninfer-serve (Chat Completions, streaming).

  python p8_probe.py ttft   --sizes 20000,60000,120000,250000 --turns 3 --out ttft.jsonl
  python p8_probe.py switch --sessions 4 --size 60000 --rounds 3 --out switch.jsonl

Sessions look like Pi: a system prompt, a user task, then assistant tool calls with tool results
(file chunks from the common-sample eval corpus and NInfer sources). Each session starts at a
different place in the chunk pool and has its own system line, so sessions diverge early.
"""
import argparse
import json
import os
import pathlib
import time
import urllib.request

URL = f"http://127.0.0.1:{os.environ.get('NINFER_PORT', '8013')}/v1/chat/completions"
MODEL = "Qwen/Qwen3.8-Flash-Next"
TOKENIZER = os.environ.get("NINFER_TOKENIZER", "")
NINFER = pathlib.Path(os.environ.get("NINFER_SRC", pathlib.Path(__file__).resolve().parents[3]))
CHUNK_CHARS = 6000
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file from the workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def chunk_pool():
    files = []
    data = NINFER / "eval" / "corpora" / "perplexity-1m" / "data"
    if not data.is_dir():
        raise SystemExit(f"{data} is missing; set NINFER_SRC to a tree that has eval/corpora/perplexity-1m")
    for domain in ("pg19", "wikitext", "ninfer"):
        files += sorted((data / domain).glob("*.txt"))
    files += [p for p in sorted((NINFER / "src").rglob("*")) if p.suffix in {".cpp", ".h", ".cu", ".cuh"}]
    pool = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(NINFER).as_posix()
        for i in range(0, len(text), CHUNK_CHARS):
            pool.append((f"{rel}#{i // CHUNK_CHARS}", text[i : i + CHUNK_CHARS]))
    return pool


class Counter:
    def __init__(self):
        from transformers import AutoTokenizer

        if not TOKENIZER:
            raise SystemExit("set NINFER_TOKENIZER to the tokenizer directory")
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def __call__(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))


def tool_turn(call_id, name, body):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": name})}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


class Session:
    def __init__(self, sid, pool, start, count):
        self.sid = sid
        self.pool = pool
        self.cursor = start
        self.count = count
        self.calls = 0
        self.messages = [
            {"role": "system", "content": f"Session {sid}. You are a coding agent working in a large repository. Use the read tool to inspect files."},
            {"role": "user", "content": f"Session {sid}: read through these files and keep notes; answer briefly when asked."},
        ]
        self.tokens = 0

    def add_reads(self, target_tokens):
        while self.tokens < target_tokens:
            name, body = self.pool[self.cursor % len(self.pool)]
            self.cursor += 1
            self.messages += tool_turn(f"call_{self.sid}_{self.calls:04d}", name, body)
            self.calls += 1
            self.tokens += self.count(body) + 24

    def add_small_read(self, chars=2400):
        name, body = self.pool[self.cursor % len(self.pool)]
        self.cursor += 1
        self.messages += tool_turn(f"call_{self.sid}_{self.calls:04d}", name, body[:chars])
        self.calls += 1


SESSION_HEADERS = False  # --session-headers: send x-session-affinity per session like Pi can


def stream(messages, max_tokens, tools=TOOLS, thinking=False, session=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": thinking,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if SESSION_HEADERS and session:
        headers["x-session-affinity"] = f"p8-{session}"
    req = urllib.request.Request(URL, data=body, headers=headers)
    started = time.perf_counter()
    first = None
    first_content = None
    final = None
    text = []
    reasoning = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("timings") or chunk.get("usage"):
                final = chunk
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta", {})
                piece = delta.get("content") or ""
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if piece or delta.get("tool_calls") or delta.get("reasoning_content"):
                    if first is None:
                        first = time.perf_counter() - started
                    if (piece or delta.get("tool_calls")) and first_content is None:
                        first_content = time.perf_counter() - started
                    text.append(piece)
    total = time.perf_counter() - started
    timings = (final or {}).get("timings") or {}
    return {
        "client_ttft_ms": round((first or total) * 1000.0, 1),
        "first_content_ms": round((first_content or total) * 1000.0, 1),
        "total_s": round(total, 3),
        "prompt_n": timings.get("prompt_n"),
        "cache_n": timings.get("cache_n"),
        "prompt_ms": round(timings.get("prompt_ms", 0.0), 1),
        "decode_tps": round(timings.get("predicted_per_second", 0.0), 1),
        "text": "".join(text),
        "reasoning": "".join(reasoning),
    }


def ask(session, label, out, question, max_tokens=32):
    messages = session.messages + [{"role": "user", "content": question}]
    result = stream(messages, max_tokens, session=session.sid)
    record = {"label": label, "session": session.sid, "messages": len(messages), **result}
    record["text"] = record["text"][:120].replace("\n", " ")
    print(json.dumps(record), flush=True)
    out.write(json.dumps(record) + "\n")
    out.flush()
    session.messages = messages + [{"role": "assistant", "content": result["text"]}]
    return result


def run_ttft(args, pool, count, out):
    for index, size in enumerate(int(s) for s in args.sizes.split(",")):
        session = Session(f"t{index}", pool, start=index * 997, count=count)
        session.add_reads(size)
        ask(session, f"cold_{size}", out, "In one sentence, what was the last file about?")
        for turn in range(args.turns):
            session.add_small_read()
            ask(session, f"extend_{size}_{turn}", out, f"Turn {turn}: one sentence on that file.")
        repeat = session.messages[:-1]
        result = stream(repeat, 16, session=session.sid)
        record = {"label": f"repeat_{size}", "session": session.sid, "messages": len(repeat), **result}
        record["text"] = record["text"][:120].replace("\n", " ")
        print(json.dumps(record), flush=True)
        out.write(json.dumps(record) + "\n")


def run_switch(args, pool, count, out):
    stride = len(pool) // args.sessions
    sessions = [Session(f"{args.tag}{k}", pool, start=k * stride + 13, count=count) for k in range(args.sessions)]
    for session in sessions:
        session.add_reads(args.size)
        ask(session, f"build_{session.sid}", out, "In one sentence, what was the last file about?")
    for rnd in range(args.rounds):
        for session in sessions:
            session.add_small_read()
            ask(session, f"switch_r{rnd}_{session.sid}", out, f"Round {rnd}: one sentence on that file.")


PI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name}: " + desc * 6,
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Workspace-relative path."},
                               "content": {"type": "string", "description": "Text to write or the command to run."}},
                "required": ["path"],
            },
        },
    }
    for name, desc in [
        ("read", "Read a file. Output is truncated to 2000 lines or 50KB; use offset and limit for large files. "),
        ("bash", "Run a shell command in the workspace. Returns stdout and stderr; long output is truncated. "),
        ("edit", "Replace an exact string in a file. The old text must match exactly, including whitespace. "),
        ("write", "Create or overwrite a file with the given content. Parent directories are created. "),
        ("grep", "Search file contents with a regular expression; respects .gitignore. "),
        ("find", "Find files by glob pattern; respects .gitignore. "),
        ("ls", "List a directory, with sizes. "),
    ]
]


def text_of(pool, count, start, tokens):
    parts, have, k = [], 0, start
    while have < tokens:
        body = pool[k % len(pool)][1]
        k += 1
        need = tokens - have
        piece_tokens = count(body)
        if piece_tokens > need:
            body = body[: max(1, int(len(body) * need / piece_tokens))]
            piece_tokens = count(body)
        parts.append(body)
        have += piece_tokens
    return "\n\n".join(parts)


def record(out, label, result, **extra):
    rec = {"label": label, **extra, **result}
    rec["text"] = rec["text"][:80].replace("\n", " ")
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
    out.flush()


def with_cache_control(messages, tools):
    """Pi's openai-completions with compat.cacheControlFormat 'anthropic'."""
    marked = json.loads(json.dumps(messages))
    control = {"type": "ephemeral"}
    marked[0]["content"] = [{"type": "text", "text": marked[0]["content"], "cache_control": control}]
    marked[-1]["content"] = [{"type": "text", "text": marked[-1]["content"], "cache_control": control}]
    marked_tools = json.loads(json.dumps(tools))
    marked_tools[-1]["cache_control"] = control
    return marked, marked_tools


def run_shared(args, pool, count, out):
    import uuid

    tasks = ["list the files", "explain the build", "find the tests", "fix the typo"]
    for label, start, explicit in (("pi_newchat", 4242, False), ("pi_newchat_cc", 6464, True)):
        system = "You are an expert coding assistant operating inside pi, a coding agent harness.\n\n" + text_of(
            pool, count, start=start, tokens=args.system_tokens)
        for k in range(4):
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": f"Task {k}: {tasks[k]} in this repository."}]
            tools = PI_TOOLS
            if explicit:
                messages, tools = with_cache_control(messages, tools)
            result = stream(messages, 32, tools=tools)
            record(out, label, result, chat=k)
    question = "Summarize the text above in one sentence."
    for size in (1000, 8000):
        for rep in range(3):
            doc = text_of(pool, count, start=size + rep * 31 + 900, tokens=size)
            messages = [{"role": "system", "content": f"[{uuid.uuid4().hex}] You are a helpful assistant."},
                        {"role": "user", "content": f"{doc}\n\n{question}"}]
            record(out, f"new_{size}_plain", stream(messages, 48, tools=None), rep=rep)


def run_short(args, pool, count, out):
    import threading
    import uuid

    question = "Summarize the text above in one sentence."
    # (a) new chat, cold, no shared prefix (random first line), thinking off and on
    for thinking in (False, True):
        for size in (300, 1000, 2000, 8000):
            for rep in range(3):
                nonce = uuid.uuid4().hex
                doc = text_of(pool, count, start=size + rep * 31 + (500 if thinking else 0), tokens=size)
                messages = [{"role": "system", "content": f"[{nonce}] You are a helpful assistant."},
                            {"role": "user", "content": f"{doc}\n\n{question}"}]
                result = stream(messages, 256 if thinking else 48, tools=None, thinking=thinking)
                record(out, f"new_{size}_{'think' if thinking else 'plain'}", result, rep=rep)
    # (b) live multi-turn: short turns, growing history, thinking off then on
    for thinking in (False, True):
        nonce = uuid.uuid4().hex
        messages = [{"role": "system", "content": f"[{nonce}] You are a friendly assistant in a live chat. Keep answers short."}]
        topics = ["the weather in spring", "a good pasta recipe", "why the sky is blue", "how bicycles stay upright",
                  "the history of tea", "tips for learning chess", "what makes a good password", "how bees make honey",
                  "a short poem about rain", "the best way to brew coffee"]
        for turn in range(20):
            messages.append({"role": "user", "content": f"Tell me something about {topics[turn % len(topics)]}."})
            result = stream(messages, 256 if thinking else 60, tools=None, thinking=thinking)
            record(out, f"live_{'think' if thinking else 'plain'}", result, turn=turn, messages=len(messages))
            assistant = {"role": "assistant", "content": result["text"]}
            if result.get("reasoning"):
                assistant["reasoning_content"] = result["reasoning"]
            messages.append(assistant)
    # (c) Pi-sized system prompt + tools shared by new chats
    system = "You are an expert coding assistant operating inside pi, a coding agent harness.\n\n" + text_of(
        pool, count, start=4242, tokens=args.system_tokens)
    for k in range(4):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"Task {k}: {['list the files', 'explain the build', 'find the tests', 'fix the typo'][k]} in this repository."}]
        result = stream(messages, 32, tools=PI_TOOLS)
        record(out, "pi_newchat", result, chat=k)
    # (d) short chat while a long cold prefill runs
    long_doc = text_of(pool, count, start=9001, tokens=args.long_tokens)
    long_result = {}

    def long_request():
        msgs = [{"role": "system", "content": f"[{uuid.uuid4().hex}] long"},
                {"role": "user", "content": f"{long_doc}\n\n{question}"}]
        long_result.update(stream(msgs, 16, tools=None))

    thread = threading.Thread(target=long_request)
    started = time.perf_counter()
    thread.start()
    time.sleep(3.0)
    for k in range(3):
        doc = text_of(pool, count, start=777 + k, tokens=1000)
        msgs = [{"role": "system", "content": f"[{uuid.uuid4().hex}] You are a helpful assistant."},
                {"role": "user", "content": f"{doc}\n\n{question}"}]
        result = stream(msgs, 32, tools=None)
        record(out, "short_during_long", result, k=k, long_running=thread.is_alive(),
               at_s=round(time.perf_counter() - started, 2))
        time.sleep(3.0)
    thread.join()
    record(out, f"long_{args.long_tokens}", long_result)
    # (e) a longer reply while another long cold prefill runs: decode speed beside prefill
    long_result.clear()
    thread = threading.Thread(target=long_request)
    started = time.perf_counter()
    thread.start()
    time.sleep(3.0)
    msgs = [{"role": "system", "content": f"[{uuid.uuid4().hex}] You are a helpful assistant."},
            {"role": "user", "content": "Explain in detail how a hash map works, with an example in Python."}]
    result = stream(msgs, 256, tools=None)
    record(out, "reply256_during_long", result, long_running=thread.is_alive(),
           at_s=round(time.perf_counter() - started, 2))
    thread.join()
    record(out, f"long_{args.long_tokens}_b", long_result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ttft", "switch", "short", "shared"])
    parser.add_argument("--system-tokens", type=int, default=7000)
    parser.add_argument("--long-tokens", type=int, default=120000)
    parser.add_argument("--sizes", default="20000,60000,120000,250000")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--size", type=int, default=60000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tag", default="s")
    parser.add_argument("--out", required=True)
    parser.add_argument("--session-headers", action="store_true")
    args = parser.parse_args()
    global SESSION_HEADERS
    SESSION_HEADERS = args.session_headers
    pool = chunk_pool()
    count = Counter()
    with open(args.out, "a", encoding="utf-8") as out:
        if args.mode == "ttft":
            run_ttft(args, pool, count, out)
        elif args.mode == "short":
            run_short(args, pool, count, out)
        elif args.mode == "shared":
            run_shared(args, pool, count, out)
        else:
            run_switch(args, pool, count, out)


if __name__ == "__main__":
    main()
