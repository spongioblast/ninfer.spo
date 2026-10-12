# Architecture

One process loads one V3 `.ninfer` artifact onto one GPU. The registry (`src/models/registry.h`) picks Qwen3.5, Qwen3.5 MoE, or Qwen3.8 Flash-Next. `construct_model` in `src/runtime/engine/model_instance.h` builds either a `ModelInstance` or a `FlashNextInstance`. Each instance owns a frontend, the loaded weights, and a `Program`.

## Engine

`EngineCore` in `src/runtime/engine/engine_core.h` is the only generation loop. It is a template on the model instance. The instance's `ModelContract` (`RuntimeTypes` on each family) supplies the prompt, the program, the execution unit, and the checkpoint handle. The engine admits requests, forms prefill and decode units, runs one unit, and publishes output. A request that cannot obtain its legal unit after reclaim is finished with a context-length error (HTTP 400, not a retryable 429) and removed. The engine stays up. An unexpected exception completes every request the engine still holds, including one not yet in a lane, and then stops the engine.

The worker waits on a predicate. Submit, cancel, finish, and context-transfer completion wake it. A timed wait remains as a backstop. On Windows the server raises the timer resolution for the waits that are still timed.

Scheduling is a fixed set of resident lanes, chosen at startup (`--max-concurrency`, at most eight). Admission happens at unit boundaries. A prefill unit stops at the next split frontier, so a cold prompt and a resumed prompt take the same chunks.

## Context cache

`ResourceManager` in `src/runtime/engine/context_cache/resource_manager.h` decides which checkpoints a request may reuse and which ones reclaim may drop. The model program owns the bytes. A shared-prefix capture exports the lane's KV prefix as an immutable view (full pages shared, a partial tail page copied) and freezes the state slot for the checkpoint. The continuing lane keeps its KV address and works on a copy of the state slot. The next request forks that checkpoint instead of consuming it. When the device state pool is full, capture reports a state-slot shortage and the engine reclaims a victim. A checkpoint the engine still holds is not released out from under it. Flash-Next keeps one catalog record per state image, so the state pool is the only limit the resource manager has to see. A fixed record count below the pool size would drop captures and finished continuations without a shortage to reclaim for.

## Program layer

`src/program` holds the storage every family shares:

- `src/program/storage/kv_address_space.h` and `logical_kv_store.h` address KV pages for both families.
- `src/program/storage/host_kv_store.h` is the host copy.
- `src/program/storage/kv_pools.h` describes a model's pools. Retention is full-page only. Sliding-window retention is not implemented.
- `src/program/program_types.h` is the engine-facing values: sequence handle, checkpoint handle, source candidate.
- `src/program/program_methods.inc` and `program_names.inc` declare the engine-facing `Program` methods and the shared names once. Each family includes them in its own `Program` class and namespace.
- `src/program/speculative/mtp_alignment.h` is the shared MTP alignment.
- `src/program/execution/visual_scatter.h` places vision tokens.
- `src/program/execution/graph_family.h` holds the decode CUDA Graph families: one executable per topology class, updated in place to the profile that covers the batch and frontier.
- `src/program/execution/causal_score.h` is the perplexity path around a family's own prefill and projection: request checks, the scoring lane, tile staging, and target logprobs.

Model math does not live here. Each family still owns its program transactions under `src/models/<family>/program/`. Flash-Next's program is `src/models/qwen3_8_flash_next/program/v3_program.h`. Qwen3.5's is `src/models/qwen3_5/program/program.h`. The two transaction sets are shaped differently (Qwen3.5 keeps upstream's per-transaction files, Flash-Next its P3–P6 split), so they are not hoisted. `tests/program/test_attention_only_core.cpp` drives the shared storage with two KV pools of different geometry and no recurrent state.

## Checkpoints and replay

A checkpoint records a frontier, a role, and the KV and state needed to resume. Continuation checkpoints are consumed by the request that resumes them. A shared prefix is forked: the source stays, and the destination is a copy. Replay recomputes a suffix when the resident state was dropped. The engine's scheduling stats count restored tokens separately from the original prefill.

## Speculative decoding

`--spec mtp` drafts with the model's MTP head. `--draft-tokens` sets the draft length, and `auto` lets the model choose. `--lm-head-draft` drafts from the language-model head. The verify path accepts or rejects those tokens. A draft source does not change greedy output: the accepted tokens are the target model's.

## Grammar

Constrained output (grammar, JSON schema, JSON object, regex, choice) is applied in `src/text` and the request parsers under `src/serve`. The constraint observes the committed output. It does not change the model weights or the cache identity of an unconstrained prefix.

## Decisions that still hold

- The public surface is the Engine. There is no second inference loop.
- `getenv` for `NINFER_*` is read in `src/core/runtime_options.cpp`. Serve flags override those switches.
- Kernels stay as they are unless an op oracle passes and, when tokens move, perplexity is unchanged.
- Qwen3.5 and Flash-Next do not include each other.
- The build compiles only `sm_120a`.
