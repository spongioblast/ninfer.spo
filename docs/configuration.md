# Configuration

Serve flags win over the environment for the switches in `src/core/runtime_options.h`. Every `NINFER_*` variable the engine reads is read in `src/core/runtime_options.cpp`; the ones scripts and gates read are listed at the end. Diagnostics have no flag.

## Environment

| Variable | Effect |
|---|---|
| `NINFER_CUDA_SYNC` | CUDA sync mode: `spin`, `blocking`, `yield`, or `auto`. Any other value fails startup. Unset means spin. |
| `NINFER_NUMA_NODE` | NUMA node, `-1` through `63`. Also the default for `--numa-node`. |
| `NINFER_PRIORITY` | `normal` lowers process priority. Anything else, including unset, stays high. Windows only. |
| `NINFER_THREAD_TUNING` | `0` turns thread tuning off. Windows only. |
| `NINFER_PIN_CEILING_GIB` | Cap on pinned host memory in GiB. `0` or unset means no cap. A sign or a value that does not fit is ignored with a warning. |
| `NINFER_DECODE_WAIT` | `spin`, `sleep`, `block`, or `stock`. Empty keeps the platform default: spin on Windows, sleep elsewhere. Any other value is ignored with a warning. |
| `NINFER_WAKE_MARGIN_US` | Wake margin, clamped to 0..5000. Default 250 µs, or 1000 µs on Windows in sleep mode. |
| `NINFER_FLASH_NEXT_PREFILL_GRAPH_MAX_TOKENS` | Largest prefill graph, clamped to 0..512. Default 64. |
| `NINFER_FLASH_NEXT_PREFILL_GRAPH_CACHE` | How many prefill graphs to keep, at least 1. Default 64. |
| `NINFER_FLASH_NEXT_PREFILL_GRAPHS` | `0` turns prefill graphs off. |
| `NINFER_DEBUG_PLANNER_TRACE` | Present means trace the planner. |
| `NINFER_DEBUG_LOGITS_DIR` | Directory for logit dumps. |
| `NINFER_DEBUG_STATE_DIR` | Directory for state dumps. Setting it also turns prefill graphs off. |
| `NINFER_DEBUG_STATE_FRONTIER` | Frontier filter for the state dump. |
| `NINFER_DEBUG_PREFILL_GRAPH_AUDIT` | `1` audits prefill graphs. |
| `NINFER_DEBUG_PREFILL_GRAPH_MEMLOG` | `1` logs prefill-graph memory. |
| `NINFER_DEBUG_ROUND_STATS` | How many rounds to print, clamped to 0..10000. |

A numeric variable that is not a whole number is ignored with a warning and keeps its default.

The matching serve flags are `--cuda-sync`, `--priority`, `--no-thread-tuning`, `--pin-ceiling-gib`, `--decode-wait`, `--wake-margin-us`, `--prefill-graph-max-tokens`, `--prefill-graph-cache`, and `--no-prefill-graphs`.

## Scripts, tests, and gates

The engine does not read these. The scripts, tests, and probes that do are named in each row.

| Variable | Read by | Effect |
|---|---|---|
| `NINFER_MODEL` | `serve.cmd`, `serve.sh`, perf probes, `ppl_ab.cmd` | The `.ninfer` artifact. |
| `NINFER_PORT` | `serve.cmd`, `serve.sh`, probes | Server port. 8003 when unset; `run.py perf` and `soak` default it to 8013. |
| `NINFER_BUILD` | `serve.cmd`, `serve_ctl.py` | Build folder, relative to the repo root or absolute. Default `..\build-spo-release`. |
| `NINFER_LM_HEAD_DRAFT` | `serve.cmd` | `0` leaves out `--lm-head-draft`. |
| `NINFER_ALLOW_DEBUG_HEAP` | `serve.cmd` | Start even when the exe has an Image File Execution Options entry. |
| `NINFER_PRESETS`, `NINFER_MODEL_ID` | `serve.sh` | Cost presets file and public model id. |
| `NINFER_BUILD_TEMP` | `build.cmd` | Compiler temp folder. Default `<build>\.tmp`. |
| `NINFER_PERFORMANCE_TRACE` | CMake option (`build.cmd trace`) | Compiles the performance trace in. |
| `NINFER_QWEN38_FLASH_NEXT_WEIGHTS` | Flash-Next real tests | Artifact for the real tests. Unset skips them. |
| `NINFER_TEST_ARTIFACT` | qwen3_5 artifact tests | Artifact for those tests. Unset skips them. |
| `NINFER_GATE_OUT` | `run.py`, `serve_ctl.py` | Gate output folder. Default `<workspace>/tmp/ninfer`. |
| `NINFER_TOKENIZER` | `p8_probe.py`, `cross_path.py` | Tokenizer directory. |
| `NINFER_SRC` | `p8_probe.py` | Tree that has `eval/corpora/perplexity-1m`. Default this repo. |
| `NINFER_CORPUS` | `mtp_ab.cmd`, `prof_decode.cmd` | `common_mix.ids` for the benchmark runs. |
| `NINFER_PPL_CORPUS` | `ppl_ab.cmd` | Perplexity manifest. |
| `NINFER_NSYS` | `prof_decode.cmd` | `nsys.exe` path. |
| `NINFER_SOAK_MEDIA` | `soak.py` | Folder of images for the image turns. |

## Server

`ninfer-spo-serve` takes the artifact path first. Defaults are in `src/serve/serve_options.h`.

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8080` | Bind port. `scripts/serve.cmd` passes `NINFER_PORT`, or 8003. |
| `--api-key` | empty | Empty means no authentication. The value is redacted in the startup record. |
| `--model-id` | artifact name | Public model id. |
| `--request-log-jsonl` | empty | Request log path. Empty disables it. |
| `--max-context` | 8192 | Per-request context ceiling. |
| `--kv-capacity` | `--max-context` | KV pool in tokens, or `auto`. It must be at least `--max-context`. |
| `--kv-headroom-mib` | | Device memory `auto` leaves free. Requires `--kv-capacity auto`. |
| `--max-concurrency` | 1 | Resident lanes. The engine caps this at 8. |
| `--max-pending-requests` | 16 | Queue length. |
| `--pending-timeout-ms` | 30000 | How long a queued request waits. |
| `--prefill-chunk` | 1024 | Prefill unit size, a multiple of 128. `scripts/serve.cmd` passes 8192. |
| `--capture-path` | | Directory for activation captures. A request records only when its JSON sets `capture.record` to true. The record id comes back as `capture.record_id` on the chat completion, the responses object, the last streamed chat chunk, and the Anthropic message. Capture turns do not reuse another request's prefix. |
| `--capture-sites` | `prompt_last,completion_last` | Which positions to record. Each site is `prompt_last` or `completion_last`. |
| `--steering-pack` | | Steering pack to load at startup. A request may set `steering.strength` in `[0,1]`. |
| `--context-cost-presets` | | JSON cost presets. |
| `--log-stats-interval-ms` | 5000 | `0` disables the throughput log. |
| `--max-request-mib` | 384 | Request body cap. |
| `--media-cache-mib` | 1024 | Encoded media cache. |
| `--media-live-mib` | 2048 | Decoded media residency. |
| `--media-preprocess-threads` | 0 | `0` picks a bounded pool from the host. |
| `--device-state-slots` | | Extra device state slots beyond the live lanes. |
| `--host-context-mib` | 8192 plus eight StateImages | Host context budget. |
| `--response-store-max-records` | 1024 | Stored responses. |
| `--response-store-max-mib` | 256 | Stored response bytes. |
| `--device` | 0 | CUDA device. |
| `--kv-dtype` | bf16 | KV storage: `bf16`, `int8`, `fp8`, `nvfp4`, or `k8v4`. Flash-Next accepts `bf16`, `fp8`, and `int8`. |
| `--spec` | | Speculative backend: `mtp`, `dflash`, or `dflash2`. Flash-Next accepts only `mtp`, which `scripts/serve.cmd` uses. |
| `--draft-tokens` | | Draft length: `auto` or digits. Flash-Next MTP takes `auto` or 1..7. |
| `--default-max-tokens` | 8192 | Used when the client omits `max_tokens`. |
| `--default-thinking-budget` | | Thinking token budget. |
| `--vision` | off | Enable vision. |
| `--no-cuda-graph` | graphs on | Disable CUDA graphs. |
| `--no-prefix-reuse` | reuse on | Disable prefix reuse. |
| `--lm-head-draft` | off | Draft from the language-model head. |
| `--chat-template` | artifact | Override the chat template. |
| `--no-thinking` | | Disable thinking. |
| `--preserve-thinking` | | Keep reasoning for the next turn. |
| `--cors` | off | Permissive CORS. |
| `--temperature` | | 0..2. |
| `--top-p` | | 0..1. |
| `--top-k` | | 0..20. |
| `--min-p` | | 0..1. |
| `--presence-penalty` | | -2..2. |
| `--frequency-penalty` | | -2..2. |
| `--seed` | fresh per request | |
| `--greedy` | off | Temperature 0, exact argmax. |
| `--log-level` | info | |
| `--ple-residency` | resident on Windows, mapped elsewhere | `mapped` or `resident`. |
| `--ple-large-pages` | off | Falls back to locked normal pages without the lock right. |
| `--numa-node` | `NINFER_NUMA_NODE`, else the GPU's node | `N` or `off`. |
| `--host-ram-budget-gib` | 200 | `0` skips the host RAM check. |

`ninfer-spo`, `ninfer-spo-perplexity`, and `ninfer_bench` resolve the NUMA node the same way and tune the process before loading. `ninfer-spo` and `ninfer_bench` also take the four host flags above.

## CLI

`ninfer-spo` takes the artifact path first. Flags that also exist on the server mean the same thing. The CLI-only flags are `--prompt`, `--messages`, `--max-new` (default 128), `--grammar-file`, `--json-object`, `--json-schema-file`, `--regex`, `--choice`, `--raw-output`, `--print-token-ids`, `--thinking-budget`, `--reasoning-effort`, `--stop`, `--reasoning-stop`, and `--stop-token-id`. Its context and KV defaults are 2048, and its prefill chunk default is 1024. The definitions are in `apps/cli/options.h`.
