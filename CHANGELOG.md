# Changelog

## Unreleased

- Gemma 4 QAT text projections can be stored as `q4_g32_fp16` (signed 4-bit codes, groups of 32, one FP16 scale). `q4_g32_qat` recovers the scale already in the checkpoint instead of quantizing it again. Recipe `gemma4_31b_q4_qat`, override `gemma4_q4_a8`. Decode uses a bandwidth GEMV (1.45–1.53 TB/s from DRAM) and MTP verify an 8-token tensor-core GEMV. With `AllowA8`, the trunk shapes run W4A8 from 9 tokens: each group's codes become E4M3 under one BF16 multiplier per row (stored at load, 2 bytes per row), so the FP8 A8 tiles do the multiply. 9–384 tokens convert the codes inside the GEMM, 385 and up convert once per call into a workspace and reuse the FP8 A8 kernels. On the RTX PRO 6000, 31B decode is 67.5 tok/s (FP8 44.4), a 17-token prompt prefills in 19 ms instead of 51 ms, and 4096 tokens in 681 ms. KL against the QAT BF16 checkpoint is 0.0089 at context 4096 and 0.0107 at context 384, where every prefill takes the fused route (exact q4 prefill 0.0056 and 0.0063, `NINFER_Q4_EXACT=1`); GSM8K-200 is 98.5%. q4 never goes through `fp8_row_cublas`. MTP acceptance on ShareGPT prompts is low with the q4 target (length 1.95 against 6.74 on FP8) and is not investigated.
- Gemma 4 FP8 trunk projections with A8 (`gemma4_fp8_a8`) switch to A8 at 33 tokens instead of 128. Below 128 tokens the 5376-row shapes use a 32×64 tile up to 64 tokens and a split-K 64×128 tile above; the wide shapes use 64×128. TTFT of a 64-token prompt drops from 40.0 to 25.3 ms, 96 tokens from 56.7 to 33.2 ms, 127 tokens from 61.5 to 34.4 ms; 32 tokens and 128 and up are unchanged. KL at context 96 against BF16 is 0.0193, against 0.0106 with A16 only.
- Gemma 4 FP8 and NVFP4 recipes store one FP8 vocabulary table: the embedding binds the head's unscaled object, and the gather multiplies each row by the BF16 √hidden normalizer (`ops::embedding` with a scale), as Hugging Face does. The 31B artifact is 1.4 GB smaller and loads one copy. Artifacts with two tables still load. KL at context 96 is unchanged (0.0195 against 0.0193).
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
