"""Compare unconstrained and constrained answers at temperature 0 (ninfer-spo-serve on NINFER_PORT, default 8013)."""
import json
import os
import sys
import urllib.error
import urllib.request

URL = f"http://127.0.0.1:{os.environ.get('NINFER_PORT', '8013')}/v1/chat/completions"
PROMPT = ('Name the largest city in Japan and its population. Reply with only JSON of the form '
          '{"city": string, "population": integer}.')
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["city", "population"],
    "additionalProperties": False,
}


def run(label, extra, thinking=False):
    body = {"model": "Qwen/Qwen3.8-Flash-Next", "max_tokens": 400, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT}],
            "chat_template_kwargs": {"enable_thinking": thinking}, **extra}
    request = urllib.request.Request(URL, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            reply = json.loads(response.read())
        choice = reply["choices"][0]
        print(f"{label}: finish={choice['finish_reason']} "
              f"tokens={reply['usage']['completion_tokens']} {choice['message'].get('content')!r}")
    except urllib.error.HTTPError as error:
        print(f"{label}: HTTP {error.code} {error.read()[:300]!r}")


cases = sys.argv[1:] or ["free", "schema", "object", "regex", "choice"]
if "free" in cases:
    run("unconstrained", {})
if "schema" in cases:
    run("json_schema", {"response_format": {"type": "json_schema",
                                            "json_schema": {"name": "c", "schema": SCHEMA}}})
if "object" in cases:
    run("json_object", {"response_format": {"type": "json_object"}})
if "regex" in cases:
    run("regex", {"structured_outputs": {"regex": r'\{"city": "[A-Za-z ]+", "population": [0-9]+\}'}})
if "choice" in cases:
    run("choice", {"structured_outputs": {"choice": ['{"city": "Tokyo"}', '{"city": "Osaka"}']}})
