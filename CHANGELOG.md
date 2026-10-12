# Changelog

## Unreleased

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
