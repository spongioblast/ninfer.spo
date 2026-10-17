# ninfer.spo

A Windows and Linux CUDA inference engine for Qwen3.8 Flash-Next, Qwen3.5, and Gemma 4 31B on Blackwell (`sm_120`). One process, one GPU, one resident model. The artifact is a V3 `.ninfer` file. The namespace stays `ninfer`.

The binaries are `ninfer-spo`, `ninfer-spo-serve`, and `ninfer-spo-perplexity`. They speak the local CLI and OpenAI- and Anthropic-compatible HTTP. Build with `scripts/build.cmd` on Windows or `scripts/build.sh` on Linux. The steps, the CUDA architecture, and FFmpeg are in `docs/building.md`. Flags and `NINFER_*` switches are in `docs/configuration.md`.

`scripts/serve.cmd` starts the server from the release build beside this repo. It loads the artifact in `NINFER_MODEL`, listens on `NINFER_PORT` (8003 if unset), and enables MTP, vision, and preserve-thinking. The Pi wiring is in `docs/pi.md`.

The architectures the registry accepts are `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`, `Qwen3_8FlashNextForCausalLM`, and `Gemma4ForCausalLM` (`src/models/registry.h`). Gemma 4 31B-it is the ported size: text, the MTP assistant, and the vision tower.

On 2026-10-11, quick perplexity of the Flash-Next NVFP4 artifact against the 1M corpus scored 261,167 tokens at overall perplexity 3.554 (`ninfer-spo-perplexity --quick`). The release comparison against the previous engine is recorded when the acceptance gates run.

This engine is derived from NInfer (https://github.com/Neroued/ninfer), including the Windows build notes checked against the headpiece747 ninfer-5090-windows tree. Those projects remain under their own copyrights. The license is Apache-2.0. See `NOTICE`.
