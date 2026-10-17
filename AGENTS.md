# AGENTS.md

ninfer.spo is a Windows and Linux CUDA inference engine for Qwen3.8 Flash-Next, Qwen3.5, and Gemma 4 31B on Blackwell (`sm_120`). One process serves one GPU and one resident model. The product artifact is a V3 `.ninfer` file. Generation, scoring, the CLI, and serving go through the Engine.

## Layers

`src/program` owns storage and transactions: KV address space, logical and host stores, context work, graph execution, and prefix identity. It does not own model math. Model math stays in `src/models` and `src/ops`.

`tools/repo_health/health.py` `layers` enforces the include direction:

- `src/core` does not include ops, program, models, or the runtime, serve, product, text, and artifact layers.
- `src/ops` does not include program or a model.
- `src/program` does not include a model family.
- `src/models/qwen3_5`, `src/models/qwen3_8_flash_next`, and `src/models/gemma4` do not include each other.
- `src/runtime/engine` does not include a model family, except the registry and model-instance files.

`getenv` is read only in `src/core/runtime_options.cpp`. Windows-only code stays in `src/core` or a `platform` directory.

## Size, shape, and language

`health.py sizes` is the gate. C++ headers, sources, and `.inl` files under `src/` and `include/` have a soft line count of 800 and a hard cap of 1250. CUDA `.cu` and `.cuh` files have a hard cap of 1500. Python has a hard cap of 1000. A soft overage prints. A hard breach fails unless the path is in `tools/repo_health/allowlist.txt`.

`health.py lang` fails on CJK text in sources and docs. Write English.

## Build and gates

`scripts/build.cmd dev` is the Windows build. `python tools/gates/run.py dev` is the inner loop: build, an optional test pattern, and health on the files just changed.

| Level | When |
|---|---|
| `dev` | While editing. |
| `quick` | Before a commit that touches shared code. Build, every test whose name does not contain `_real`, and `health.py all`. |
| `full` | After a model, kernel, or program change. The Flash-Next, QSA, grammar, and frontend tests, then quick perplexity. |
| `perf` | Before claiming a speed change. |
| `soak` | Before claiming a long run does not leak. |
| `linux` | When the WSL CUDA toolchain is present. |

## Numerical changes

An op change needs an oracle test, not a kernel-to-kernel comparison. A change that moves tokens needs equal perplexity and a log entry. A performance claim needs a `perf` run. Revert a trial whose pairs do not all improve.

## Large changes

Record the work in the inventory before editing a wide surface, and close the row when the check passes. One concern per commit.

## Where to read

`docs/architecture.md` is the engine and the program layer. `docs/code-map.md` is the directory map. `docs/adding-a-model.md` is the checklist for a new family. `docs/building.md`, `docs/configuration.md`, `docs/gates.md`, and `docs/pi.md` are the operator guides. Format and op references stay under `docs/maintainer/`.
