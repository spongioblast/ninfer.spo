# Changelog

## Unreleased

- `Bf16GemmContext` takes no cuBLAS workspace unless the caller asks for one. The 64 MiB buffer changed the summation order and moved the Flash-Next greedy prefix and the Gemma bf16 parity.
- Gemma 4 text prefill of 32, 64, 128, 256, 512, and 1024 tokens runs as one CUDA graph (`prefill_graphs.h`). Vision chunks stay eager.
- Gemma 4 FP8 A8 uses a 64-token tile at 128 tokens and a 128-row tile up to 512 tokens, where those beat the 128×256 bulk tile.
- Gemma 4 GeGLU can write the down projection's E4M3 codes in the same kernel, and the down GEMM then skips its own quantize (`ops::linear_a8_ready`).
- The Gemma 4 MTP assistant projections are quantized to FP8 at load and run on the dense GEMV.
- Gemma 4 31B-it is a third family (`Gemma4ForCausalLM`): text, the MTP assistant, and the vision tower. The converter is `tools/convert/gemma4.py`. The sliding window stays in the Gemma program ring.
- Flash-Next sizes its checkpoint catalog and KV address stores from the device state slots instead of a fixed 8 records. With `--device-state-slots` above 6 the catalog used to fill first: captures were skipped and finished continuations dropped, while the resource manager saw free slots and reclaimed nothing, so every later request resumed from one old checkpoint. New real test `--many-checkpoints`.
- `scripts/serve.cmd` passes `--device-state-slots 16`. With the default two, Pi sessions fell back to old checkpoints after pi-lens rewrites (TTFT up to 3.4 s), and two chats fell back to the system prompt on every request (`docs/serving.md`, "Flash-Next checkpoint slots").

## base

This tree is ninfer.spo: NInfer's V3 runtime, the Windows layer, and Qwen3.8 Flash-Next and Qwen3.5 on `sm_120a`.

- Binaries `ninfer-spo`, `ninfer-spo-serve`, and `ninfer-spo-perplexity`.
- Shared program storage under `src/program`, one KV page and address store for both families, plus the `Program` declarations, decode graph families, and causal-score staging. Model math stays in the model families.
- Context-cache shared prefixes, with a device state copy so a warm continuation matches a cold one. Thinking turns reuse their history too. Reclaim counts only the pages a release actually frees.
- One request that cannot obtain a unit returns a context-length error (400). The engine keeps serving.
- Image prompts: decode positions carry the image RoPE offset, and prefix reuse compares the image, not only the token ids.
- Pause, replay, and forced tokens map KV before they write it. Penalty counts and MTP counters stay per request across a pause.
- Windows waits, exclusive bind, encode-splice check, and console code page (also restored on Ctrl+C).
- On Linux a second server cannot bind the same port (no `SO_REUSEPORT`).
- Public types split under `include/ninfer/types/`. The agent guide and the operator docs are in `AGENTS.md` and `docs/`.
- `--capture-path` and `--steering-pack` record or steer a request. The record id is `capture.record_id`. Ordinary requests, with neither set, do not create that control.
- An unexpected engine exception finishes every held request and stops the engine. A context-length rejection still leaves the engine up.
- A lost CUDA context, including after sleep, throws instead of aborting, so that same stop path runs.
- `ninfer_bench` takes the host flags and binds to the GPU's NUMA node like the server. Before, it left the 47.68 GiB PLE table wherever Windows put it.
