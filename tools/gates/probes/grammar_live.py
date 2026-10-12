"""Live constrained-decoding checks against ninfer-spo-serve on NINFER_PORT (default 8013)."""
import json
import os
import sys
import urllib.error
import urllib.request

URL = f"http://127.0.0.1:{os.environ.get('NINFER_PORT', '8013')}/v1/chat/completions"
MODEL = "Qwen/Qwen3.8-Flash-Next"
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["city", "population"],
    "additionalProperties": False,
}
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}]


def post(body):
    body = {"model": MODEL, "max_tokens": 512, "temperature": 0.7, **body}
    request = urllib.request.Request(URL, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def message(text, thinking):
    return {"messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": thinking}}


results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")


def content(reply):
    choice = reply["choices"][0]
    return choice["message"].get("content") or "", choice["message"], choice["finish_reason"]


def json_case(name, thinking):
    status, reply = post({**message("Name a large city in Japan and its population.", thinking),
                          "response_format": {"type": "json_schema",
                                              "json_schema": {"name": "city", "schema": SCHEMA}}})
    if status != 200:
        check(name, False, f"HTTP {status} {reply}")
        return
    text, msg, finish = content(reply)
    try:
        value = json.loads(text)
        ok = set(value) == {"city", "population"} and isinstance(value["population"], int)
    except ValueError:
        ok = False
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    check(name, ok and finish == "stop",
          f"finish={finish} content={text!r} reasoning_chars={len(reasoning)}")


status, reply = post({**message("Is the sky blue? Answer yes or no.", False),
                      "structured_outputs": {"choice": ["yes", "no"]}})
text = content(reply)[0] if status == 200 else reply
check("choice, thinking off", status == 200 and text in ("yes", "no"), repr(text))

json_case("json_schema, thinking off", False)
json_case("json_schema, thinking on, budget 64 forces the close", True)

status, reply = post({**message("Weather in Paris?", False), "tools": TOOLS,
                      "response_format": {"type": "json_object"}})
check("json + active tools is rejected", status == 400, f"HTTP {status} {reply.get('error', reply)}")

status, reply = post({**message("Say hello in one word.", False), "max_tokens": 16})
check("plain request after the above", status == 200,
      f"HTTP {status} {content(reply)[0]!r}" if status == 200 else f"HTTP {status} {reply}")

status, reply = post({**message("Say hello in one word.", True), "max_tokens": 200})
check("plain thinking request with budget", status == 200,
      f"HTTP {status} finish={content(reply)[2]}" if status == 200 else f"HTTP {status} {reply}")

sys.exit(0 if all(results) else 1)
