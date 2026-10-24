"""HTTP smoke test of a Gemma 4 ninfer-spo-serve for voice chat, standard library only.

Start the server first, for example:

  ninfer-spo-serve <artifact> --port 8080 --model-id local --max-context 16384 --kv-dtype bf16 \
      --spec mtp --draft-tokens 4 --vision --no-thinking \
      --device-state-slots 4 --host-context-mib 8192

then run `python -m tools.smoke.serve_gemma4_voice --url http://127.0.0.1:8080`. Checks the
max_tokens=1 prewarm, SSE streaming with usage and llama.cpp-style timings (cache_n, draft_n,
draft_n_accepted), a developer message, a stop string, tool_choice required and named, a tool result turn,
image_url input (skip with --no-vision), prompt_cache_breakpoint resume after a side request,
cancelling a stream mid-round, and context_length_exceeded.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import struct
import sys
import time
import zlib
from typing import Any
from urllib.parse import urlparse

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current local time in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Start a kitchen timer.",
            "parameters": {
                "type": "object",
                "properties": {"minutes": {"type": "integer"}},
                "required": ["minutes"],
            },
        },
    },
]

SYSTEM = "You are Juno, a voice assistant. Reply in one or two short spoken sentences."


class SmokeError(RuntimeError):
    pass


def numbered(topic: str, lines: int, salt: int) -> str:
    return "".join(
        f"{topic} note {salt * 1000 + line}: keep answers short, friendly and spoken.\n"
        for line in range(lines)
    )


def red_png(size: int = 64) -> str:
    row = b"\x00" + b"\xdc\x14\x14" * size
    raw = zlib.compress(row * size)

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", raw) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


class Client:
    def __init__(self, url: str, model: str) -> None:
        parsed = urlparse(url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.model = model

    def connect(self, timeout: float = 600.0) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def post(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        conn = self.connect()
        conn.request(
            "POST",
            "/v1/chat/completions",
            json.dumps({"model": self.model, **body}).encode(),
            {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, json.loads(data or b"{}")

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        status, value = self.post(body)
        if status != 200:
            raise SmokeError(f"HTTP {status}: {json.dumps(value)[:400]}")
        return value

    def stream(self, body: dict[str, Any]) -> dict[str, Any]:
        conn = self.connect()
        payload = {
            "model": self.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            **body,
        }
        conn.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = conn.getresponse()
        if response.status != 200:
            raise SmokeError(f"stream HTTP {response.status}: {response.read()[:400]!r}")
        result: dict[str, Any] = {"content": "", "events": 0, "usage": None, "timings": None}
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                result["done"] = True
                break
            event = json.loads(line[6:])
            result["events"] += 1
            result["usage"] = event.get("usage") or result["usage"]
            result["timings"] = event.get("timings") or result["timings"]
            for choice in event.get("choices") or []:
                result["content"] += (choice.get("delta") or {}).get("content") or ""
                result["finish"] = choice.get("finish_reason") or result.get("finish")
        conn.close()
        return result

    def health(self) -> bool:
        try:
            conn = self.connect(timeout=5.0)
            conn.request("GET", "/health")
            ok = conn.getresponse().status == 200
            conn.close()
            return ok
        except OSError:
            return False


def message(role: str, text: str, breakpoint: bool = False) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "text", "text": text}
    if breakpoint:
        part["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return {"role": role, "content": [part]}


def voice(user: str, memory: str, breakpoint: bool = False) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM + "\n" + numbered("Style", 60, 1)},
        message("system", "Memory: " + memory, breakpoint),
        {"role": "user", "content": user},
    ]


class Smoke:
    def __init__(self, client: Client, vision: bool) -> None:
        self.client = client
        self.vision = vision
        self.failures = 0

    def check(self, ok: bool, label: str, detail: Any = "") -> None:
        print(("ok   " if ok else "FAIL ") + label + (f" ({detail})" if detail != "" else ""))
        if not ok:
            self.failures += 1

    def prewarm(self) -> None:
        reply = self.client.chat(
            {"messages": voice("Hi", "none"), "tools": TOOLS, "max_tokens": 1, "temperature": 0}
        )
        usage = reply.get("usage") or {}
        self.check(usage.get("completion_tokens") == 1, "max_tokens=1 prewarm", usage)

    def streaming(self) -> None:
        body = {
            "messages": voice("What should I cook tonight?", "The user likes spicy food."),
            "tools": TOOLS,
            "max_tokens": 48,
            "temperature": 0,
        }
        first = self.client.stream(body)
        again = self.client.stream(body)
        timings = again.get("timings") or {}
        self.check(
            first.get("done") and first["content"] != "" and first["usage"] is not None,
            "SSE stream has content, usage and [DONE]",
            first["events"],
        )
        keys = ("cache_n", "prompt_n", "predicted_n", "draft_n", "draft_n_accepted")
        self.check(all(isinstance(timings.get(key), int) for key in keys), "timings fields", timings)
        self.check(timings.get("cache_n", 0) > 0, "repeated prompt reports cache_n", timings.get("cache_n"))
        self.check(again["content"] == first["content"], "repeated greedy stream is identical")

    def developer(self) -> None:
        reply = self.client.chat(
            {
                "messages": [
                    {"role": "developer", "content": "Answer with the single word PINEAPPLE."},
                    {"role": "user", "content": "Say the word."},
                ],
                "max_tokens": 8,
                "temperature": 0,
            }
        )
        content = reply["choices"][0]["message"].get("content") or ""
        self.check("PINEAPPLE" in content.upper(), "developer message is followed", content)

    def stop(self) -> None:
        reply = self.client.chat(
            {
                "messages": [{"role": "user", "content": "Count from one to ten in words, separated by commas."}],
                "stop": ["five"],
                "max_tokens": 48,
                "temperature": 0,
            }
        )
        choice = reply["choices"][0]
        content = choice["message"].get("content") or ""
        self.check(
            "four" in content.lower() and "five" not in content.lower() and choice.get("finish_reason") == "stop",
            "stop string ends the reply",
            content,
        )

    def tools(self) -> None:
        messages = voice("Set something for my pasta.", "none")
        required = self.client.chat(
            {"messages": messages, "tools": TOOLS, "tool_choice": "required", "max_tokens": 64, "temperature": 0}
        )
        choice = required["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        self.check(
            choice.get("finish_reason") == "tool_calls" and len(calls) >= 1,
            "tool_choice required returns a tool call",
            [call["function"]["name"] for call in calls],
        )
        named = self.client.chat(
            {
                "messages": voice("Hello there.", "none"),
                "tools": TOOLS,
                "tool_choice": {"type": "function", "function": {"name": "get_time"}},
                "max_tokens": 64,
                "temperature": 0,
            }
        )
        calls = named["choices"][0]["message"].get("tool_calls") or []
        name = calls[0]["function"]["name"] if calls else None
        arguments: Any = None
        if calls:
            try:
                arguments = json.loads(calls[0]["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = None
        self.check(name == "get_time" and isinstance(arguments, dict), "named tool_choice", calls)
        followup = messages + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "set_timer", "arguments": json.dumps({"minutes": 9})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"ok": True, "minutes": 9})},
        ]
        result = self.client.chat({"messages": followup, "tools": TOOLS, "max_tokens": 48, "temperature": 0})
        content = result["choices"][0]["message"].get("content") or ""
        self.check("9" in content or "nine" in content.lower(), "tool result turn", content)

    def image(self) -> None:
        if not self.vision:
            print("skip image_url (--no-vision)")
            return
        reply = self.client.chat(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": red_png()}},
                            {"type": "text", "text": "What is the main color of this image? One word."},
                        ],
                    }
                ],
                "max_tokens": 8,
                "temperature": 0,
            }
        )
        content = reply["choices"][0]["message"].get("content") or ""
        self.check("red" in content.lower(), "image_url answers red", content)

    def breakpoint(self) -> None:
        memory = "The user is vegan.\n" + numbered("Memory", 15, 4)
        self.client.chat(
            {"messages": voice(numbered("Question", 16, 5), memory, True), "tools": TOOLS, "max_tokens": 4, "temperature": 0}
        )
        self.client.chat(
            {
                "messages": [
                    {"role": "system", "content": "You write short titles."},
                    {"role": "user", "content": numbered("Memo", 30, 3)},
                ],
                "max_tokens": 4,
                "temperature": 0,
            }
        )
        reply = self.client.chat(
            {"messages": voice("And a dessert?", memory, True), "tools": TOOLS, "max_tokens": 16, "temperature": 0}
        )
        timings = reply.get("timings") or {}
        self.check(
            timings.get("prompt_n", 10**9) <= 32,
            "breakpoint resumes after a side request",
            {key: timings.get(key) for key in ("cache_n", "prompt_n")},
        )

    def cancel(self) -> None:
        conn = self.client.connect()
        body = {
            "model": self.client.model,
            "stream": True,
            "messages": [{"role": "user", "content": "Count slowly from one to two hundred in words."}],
            "max_tokens": 400,
            "temperature": 0,
        }
        conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        response = conn.getresponse()
        events = 0
        for raw in response:
            if raw.startswith(b"data: "):
                events += 1
            if events >= 6:
                break
        if conn.sock is not None:
            conn.sock.close()
        conn.close()
        start = time.perf_counter()
        reply = self.client.chat({"messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 4, "temperature": 0})
        seconds = time.perf_counter() - start
        self.check(
            reply["choices"][0]["message"].get("content") is not None and self.client.health(),
            "request after a cancelled stream",
            f"{seconds:.2f} s",
        )

    def too_long(self) -> None:
        status, value = self.client.post(
            {"messages": [{"role": "user", "content": "word " * 70000}], "max_tokens": 4}
        )
        code = (value.get("error") or {}).get("code")
        self.check(status == 400 and code == "context_length_exceeded", "context_length_exceeded", (status, code))

    def run(self) -> int:
        deadline = time.monotonic() + 600
        while not self.client.health():
            if time.monotonic() > deadline:
                raise SmokeError("server is not healthy")
            time.sleep(0.5)
        for step in (
            self.prewarm,
            self.streaming,
            self.developer,
            self.stop,
            self.tools,
            self.image,
            self.breakpoint,
            self.cancel,
            self.too_long,
        ):
            try:
                step()
            except (SmokeError, OSError, KeyError, IndexError, json.JSONDecodeError) as error:
                self.check(False, step.__name__, error)
        print("FAIL" if self.failures else "OK", "gemma4 voice smoke")
        return 1 if self.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args()
    return Smoke(Client(args.url, args.model), not args.no_vision).run()


if __name__ == "__main__":
    sys.exit(main())
