# Changelog

## Unreleased

- Gemma 4 vision attention is one tensor-core flash kernel with FP32 softmax weights and no T² buffers. A 2,520-token image encodes in about 40 ms instead of about 500 ms. The encoder no longer syncs per padded GEMM, and patch scaling, RoPE tables, and pooling run on the GPU. `gemma4_vision_encode` returns before the embeddings are ready; `Program::encode_vision` syncs the stream first. Without that sync a red image was described as white.
- Gemma 4 FP8 trunk projections at 17 to 127 tokens have their own routes (`shapes/gemma4_a16.cu`). The new `fp8_a16_wide_gemv` kernel gives each CTA 32 or 64 rows and is 23–35% faster at 32 tokens and 34–60% faster at 48 than the generic route. A 32-token prompt's TTFT on `gemma4_31b_fp8_a8` drops from 33.9 to 25.4 ms. Decode and MTP verify (at most 16 tokens) are unchanged.
- Gemma 4 `--draft-tokens auto` drafts 6 tokens per round instead of 5: +4.9% decode on a code and docs corpus and +11.6% on ShareGPT at 4k context.
- `ops::topk_columns` merges with 1024 threads per column (about 13 µs instead of 650–745 µs on a 262,144-row column). It has a total order for NaN (a NaN at row 0 first, other NaNs after every number) and writes -1 for ranks past `valid_rows`.
- Gemma 4 vision attention no longer uses cuBLAS. The dots and the value multiply are ours, and the softmax between them is unchanged. The encoder's registered BF16 projections, including each half of gate/up, run on the BF16 tensor-core kernels.
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
