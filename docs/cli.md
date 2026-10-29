# CLI

`ninfer-spo` runs one request against one V3 `.ninfer` artifact. `scripts/build.cmd dev` writes it to `build-spo-dev/apps`. Flags that also exist on the server are in `docs/configuration.md`. The artifact is one of the two in the [README](../README.md): Qwen3.8 Flash-Next, or Gemma 4 for a voice-shaped run. `<artifact>` below means that file.

## Text input

```bash
ninfer-spo <artifact> \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Exactly one of `--prompt` and `--messages` is required. The CLI normally omits `--kv-capacity`, so
the shared Main Text KV pool follows the example's 32,768-token `--max-context`.

Answer content is streamed to stdout. Human-readable startup milestones and runtime errors are
written to stderr without service timestamps. Reasoning and the CLI result report (timings,
throughput, GPU memory, token IDs when requested, and speculative-decoding statistics) also use
stderr as unprefixed product output, so stdout can be redirected independently. On a terminal,
weight materialization is one transient progress line followed by a compact Engine-ready summary.
Redirected stderr contains persistent readable progress for long loads and no carriage returns or
ANSI escapes. `--log-level debug` exposes every startup phase. Option and local prompt/message input
failures remain direct command diagnostics:

```bash
ninfer-spo <artifact> \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype fp8 \
  > answer.txt 2> run.log
```

`--chat-template FILE` overrides the artifact's built-in template with a local Jinja file.
Changes to the file take effect after restarting NInfer:

```bash
ninfer-spo <artifact> \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

`--grammar-file FILE` constrains the answer with a GBNF grammar whose entry rule is `root`:

```bash
printf 'root ::= "yes" | "no"\n' > answer.gbnf
ninfer-spo <artifact> \
  --prompt "Is 17 prime?" --no-thinking --grammar-file answer.gbnf --max-new 64
```

`--choice TEXT` selects a literal candidate; repeat the flag to supply the candidate set.
`--regex PATTERN` constrains the complete answer to a regular expression:

```bash
ninfer-spo <artifact> \
  --prompt "Classify this review: the service was excellent." --no-thinking \
  --choice positive --choice neutral --choice negative

ninfer-spo <artifact> \
  --prompt "Return ticket TASK-0042." --no-thinking --regex '(BUG|TASK)-[0-9]{4}'
```

Candidates preserve exact text; `--choice ''` explicitly permits empty content. `--regex ''`
permits only empty content. See the [language contract](maintainer/constrained-decoding.md#41-gbnf--regex--choice)
for supported regex syntax.

`--json-object` constrains the answer to a JSON object. `--json-schema-file FILE` applies a JSON
Schema. All output constraint options are mutually exclusive:

```bash
printf '%s\n' '{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}' > answer.schema.json
ninfer-spo <artifact> \
  --prompt "Return the answer to 6 times 7 as JSON." --no-thinking \
  --json-schema-file answer.schema.json --max-new 64
```

GBNF supports recursive rules, Unicode character classes and repetition. All constraint modes work with
ordinary decoding, MTP, DFlash and DFlash2. Thinking may precede the constrained answer; an output
limit or cancellation can leave it incomplete. JSON modes can accompany tools supplied in messages:
the answer is either JSON or a tool-call sequence. GBNF, choice and regex require no active tools.
Constraints reject custom stops and `--raw-output`. JSON uses compact separators and declared property order. Describe
the desired content in the prompt; the schema is not added to it automatically. See the
[supported schema subset](maintainer/constrained-decoding.md#42-json-与-schema-的执行合同).

Omitted thinking and effort options use the selected template's defaults. `--no-thinking` or
`--reasoning-effort none` requests disabled thinking; other effort values cannot be combined with
`--no-thinking`. The template interprets the selected effort. `--greedy` selects exact argmax
decoding independently.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the
new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact
boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
and `</think>` to the same resident sequence without sampling, publishes the guidance through the
reasoning stream, then resumes ordinary generation from the updated context. A natural thinking
close, stop condition, cancellation, or total output/context limit at the boundary takes priority
and suppresses this insertion. The option cannot be combined with `--no-thinking`, but it can be
combined with `--reasoning-effort`.

`--max-new` counts every committed generated token, including internally inserted control tokens.
When the effective output capacity extends beyond the thinking budget, it must have room for the
complete tokenizer-derived control suffix plus one post-close model token; an undersized request is
rejected rather than truncating the suffix. Normal output sends the inserted guidance to stderr as
reasoning. `--print-token-ids` includes the inserted IDs, while `--raw-output` preserves the raw
control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
ninfer-spo <artifact> \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash/DFlash2 weights and state and the optimized proposal head;
- `--spec mtp`, `--spec dflash` (35B-A3B), and `--spec dflash2` (Qwen3.8-27B) load only
  the selected speculative backend;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input.
- the one-request CLI disables cross-request history and Host context backing, so it does not
  reserve an extra Device checkpoint StateImage or retain continuations.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an Engine
started without Vision rejects media and cannot enable Vision later. DFlash/DFlash2 and Vision may
be enabled together; these backends apply to generated-text decode after multimodal prefill and does not
accelerate Vision encode. The default speculative and Vision settings produce the smallest resident
profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
ninfer-spo <artifact> \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype fp8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
The selected template formats these roles. The maintained Qwen templates keep system/developer
messages at their input positions.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. MTP takes `--draft-tokens auto` or 1..7. DFlash and DFlash2 take 1..15. Both masked-draft backends may be combined with `--vision`.
`--lm-head-draft` selects the optimized proposal head and requires a selected backend:

```bash
ninfer-spo models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
ninfer-spo models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --kv-dtype fp8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For an artifact that contains the DFlash2 companion weights, select `--spec dflash2 --draft-tokens 7`, optionally with `--lm-head-draft` and `--vision`. DFlash2 accepts every draft count from 1 through 15; seven is the checkpoint recommendation. An artifact without the companion weights reports a missing DFlash2 component when selected. Vision, MTP, and DFlash follow the same rule: their weights are required only when that component is enabled at startup.

Only one speculative backend can be enabled per engine. DFlash accepts one to fifteen draft tokens; seven forms the measured block length eight, while fifteen uses the maximum supported block length sixteen.

## Common options

Flags and defaults are in `docs/configuration.md`. The CLI context and KV defaults are 2048, and its prefill chunk default is 1024. `ninfer-spo --help` prints the same contract.

When a sampling flag is omitted, the engine uses the loaded architecture and the rendered thinking mode. Flash-Next and Qwen3.5 use temperature 1.0, top-p 0.95, and top-k 20 while thinking, and temperature 0.7, top-p 0.80, top-k 20, and presence penalty 1.5 when thinking is off. Qwen3.5 MoE also uses presence penalty 1.5 while thinking. Gemma 4 uses temperature 1.0, top-k 20, and top-p 0.95. Frequency penalty is 0.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use `--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include generated token IDs in diagnostics.

## CUDA synchronization

`NINFER_CUDA_SYNC` selects the CUDA device synchronization schedule at startup for both the CLI
and HTTP server. When unset, it defaults to `spin`, prioritizing low synchronization latency at
the cost of CPU usage while waiting for the GPU. Use `blocking` to let the waiting thread sleep;
the decode performance cost depends on the host. `yield` yields the CPU while waiting, and `auto`
uses CUDA's scheduling heuristic, not an automatic performance benchmark.

```bash
NINFER_CUDA_SYNC=blocking ninfer-spo <artifact> --prompt "Hello"
```

The Engine-ready log reports the selected mode. Empty or unrecognized values, or failure to apply
the schedule, fail startup. This controls device scheduling (including stream synchronization);
it does not override individual CUDA event creation flags.

## Context and memory

The official artifacts have a native context limit of 262,144 tokens. The practical allocation on one GPU depends on the selected artifact, media workload, output budget, and KV-cache type.
The artifact describes its model configuration and weight representations;
`--kv-dtype` independently selects runtime KV storage. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, unified workspace,
and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program workspace, and a separate CUDA Graph driver allowance. With Vision enabled, that one
workspace contains a general execution prefix and a fixed item-output handoff region. Vision encode
may reuse the full backing before producing the output; Text/MTP/decode work remains inside the
general prefix while the handoff is live. The capacity is therefore the maximum legal simultaneous
extent, not the sum of Text, Vision scratch, and Vision output allocations. Text prefill uses
`min(--prefill-chunk,--max-context)`; Vision keeps the existing 32,768-token aggregate prompt budget
but plans Device execution for the registered 16,384-token maximum single item. Requests perform no
project-owned device allocation or growth. Context-cache capacity controls are intentionally absent
from this one-request interface; the persistent Engine and server routes own cross-request reuse and
optional Host backing.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.
