# Code map

Each directory owns one layer. The gate names are `python tools/gates/run.py` levels. The size table at the bottom is written by `python tools/repo_health/health.py report`.

| Directory | Owns | Entry | Tests | Gate |
|---|---|---|---|---|
| `src/core` | Device, tensors, paged KV, process tuning, `NINFER_*` switches | `src/core/runtime_options.cpp` | `tests/test_device.cpp`, `test_arena.cpp`, `test_tensor.cpp`, `test_kv_cache.cpp` (`tests/cmake/CoreTests.cmake`) | `quick` |
| `src/ops` | CUDA kernels and their launches | the wrapper next to each kernel family | `tests/ops` | `quick`; `full` for Flash-Next and QSA |
| `src/program` | Shared KV storage, program value types, MTP alignment, vision scatter | `src/program/program_types.h` | through the model program tests | `full` |
| `src/models/qwen3_5` | Qwen3.5 frontend, execution, and program | `src/models/qwen3_5/program/program.h` | `tests/models` names containing `qwen3_5` | `full` (`frontend`, `grammar`) |
| `src/models/qwen3_8_flash_next` | Flash-Next frontend, execution, and V3 program | `src/models/qwen3_8_flash_next/program/v3_program.h` | `tests/models` names containing `qwen3_8_flash_next` or `flash_next` | `full` |
| `src/models` | Architecture registry | `src/models/registry.h` | load-plan tests | `full` |
| `src/runtime/engine` | The generation loop, scheduler, context cache | `src/runtime/engine/engine_core.h` | `tests/runtime` | `quick`; real tests in `full` |
| `src/runtime` | Engine contracts and model construction | `src/runtime/engine/model_instance.h` | `tests/runtime` | `quick` |
| `src/serve` | HTTP, OpenAI and Anthropic parsers, request log | `src/serve/serve_options.h` | `tests` names containing `openai` or `anthropic` | `quick` |
| `src/text` | Grammar and constrained decoding | `src/text` | `tests/text` | `full` (`grammar`) |
| `src/artifact` | `.ninfer` container read and bind | `src/artifact` | `tests/artifact` | `quick` |
| `src/product` | Logging, host options, prompt input, media acquire | `src/product/host_options.h` | serve and CLI option tests | `quick` |
| `src/media` | Image and video decode | `src/media` | media tests under `tests` | `quick` |
| `apps` | `ninfer-spo`, `ninfer-spo-serve`, `ninfer-spo-perplexity` | `apps/CMakeLists.txt` | the option tests | `quick` |
| `tools/gates` | Gate runner and probes | `tools/gates/run.py` | the probes' self-tests | the level you run |
| `tools/repo_health` | Size, layer, language, and doc checks | `tools/repo_health/health.py` | | `quick` |
| `docs/maintainer` | Format and op references that are still in use | | | `health.py lang` |

`include/ninfer/types.h` is the public type umbrella. The topics are under `include/ninfer/types/`.

<!-- health:sizes:start -->

| lines | file |
|---:|---|
| 2038 | `tools/bench/ttft/cases.py` |
| 1547 | `tools/bench/ttft/report.py` |
| 1329 | `tools/bench/run_serve_corpus.py` |
| 1204 | `src/models/qwen3_5/program/transactions/reclaim.cpp` |
| 1203 | `src/ops/launcher/hyperconnection.cu` |
| 1196 | `src/models/qwen3_8_flash_next/frontend/tokenizer.cpp` |
| 1175 | `src/program/storage/kv_address_space.h` |
| 1171 | `src/ops/sparse_moe/flash_next/flash_next_moe.cu` |
| 1151 | `tools/bench/run_serve_concurrency.py` |
| 1098 | `src/models/qwen3_5/frontend/processor.cpp` |
| 1095 | `src/models/qwen3_8_flash_next/frontend/processor.cpp` |
| 1091 | `src/runtime/engine/engine_core.h` |
| 1074 | `src/serve/anthropic_messages_request.cpp` |
| 1026 | `src/ops/wrapper/gdn_input_proj.cpp` |
| 1001 | `tools/convert/qwen3_5.py` |
| 1000 | `src/models/qwen3_5/frontend/tokenizer.cpp` |
| 994 | `src/serve/request_log.cpp` |
| 977 | `src/core/paged_kv_cache.cpp` |
| 975 | `src/models/qwen3_5/program/planning/startup.cpp` |
| 966 | `src/models/qwen3_5/frontend/frontend.cpp` |
| 879 | `tools/bench/ttft/render.py` |
| 863 | `src/serve/openai_chat_request.cpp` |
| 859 | `src/ops/sparse_moe/prefill/sparse_moe_prefill_gemm.cuh` |
| 847 | `src/models/qwen3_8_flash_next/program/v3_program.cpp` |
| 840 | `src/models/qwen3_5/program/decode.cpp` |
| 831 | `tools/bench/ttft/build_fixtures.py` |
| 818 | `src/ops/softmax_attention/qsa/qsa_attention.cu` |
| 789 | `src/models/qwen3_8_flash_next/impl/runtime/layouts.cpp` |
| 779 | `src/models/qwen3_5/program/prefill.cpp` |
| 778 | `src/models/qwen3_8_flash_next/impl/runtime/state_image_store.h` |
| 775 | `src/models/qwen3_5/frontend/output_session.cpp` |
| 767 | `src/ops/linear_attention/gated_delta_net/recurrent.cuh` |
| 766 | `src/models/qwen3_5/program/storage/state_store.h` |
| 760 | `src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu` |
| 758 | `src/models/qwen3_5/frontend/tool_call_parser.cpp` |
| 753 | `src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh` |
| 751 | `src/runtime/engine/context_cache/resource_sources.inl` |
| 728 | `tools/bench/run_serve_ttft_campaign.py` |
| 725 | `src/models/qwen3_5/execution/draft.cpp` |
| 719 | `src/ops/kernel/speculative_round.cuh` |

<!-- health:sizes:end -->
