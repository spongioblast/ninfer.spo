# Gates

`python tools/gates/run.py <level>` is the runner. GPU levels need `CUDA_VISIBLE_DEVICES` pointed at a Blackwell GPU before the run. Real tests read `NINFER_QWEN38_FLASH_NEXT_WEIGHTS` (the Flash-Next artifact). Perplexity and the probes read `NINFER_MODEL`, and perplexity needs `NINFER_PPL_CORPUS` on `eval/corpora/perplexity-1m/manifest.json` from a tree that has the corpus.

`perf` and `soak` talk to `NINFER_PORT`, which they set to 8013 when it is unset, and write their outputs to `NINFER_GATE_OUT` (default `<workspace>/tmp/ninfer`). `perf` needs a spo server already listening there with its request log at `<out>/reqlog.jsonl` (the cancel probe reads it), plus `NINFER_TOKENIZER`, `NINFER_CORPUS`, and Nsight Systems. Its cross-path probe restarts a dev server on that port and leaves it running. `serve_ctl.py` stops only the `ninfer-spo-serve.exe` that listens on `NINFER_PORT`, and refuses when another program owns the port. `soak` needs `psutil`, `pynvml`, and `transformers`. `full` and `perf` run `.cmd` probes and return 1 on Linux. The other script and gate variables are in `docs/configuration.md`.

Run the Flash-Next real tests on the fp8proj serving artifact. On the plain artifact the captured prefill graph contains a cuBLAS kernel node (`nvjet_*`) whose parameters the audit in `ninfer_qwen3_8_flash_next_prefill_graph_mtp_real_test` cannot read, so it reports every audited graph as a mismatch (2026-10-11).

| Level | What it runs |
|---|---|
| `dev` | Build. If you pass a CTest regex, those tests. Health on the files changed since `HEAD` and on untracked files. Format check on those files when `clang-format` is on `PATH`. |
| `quick` | Build. Every test whose name does not contain `_real`. `health.py all`. Format check. |
| `full` | Build. CTest regex `flash_next\|qsa\|rewrite_grouping\|grammar\|frontend`, timeout 900 seconds, four jobs. Then `tools/gates/probes/ppl_ab.cmd dev` for quick perplexity. |
| `perf` | Live probe, p8 TTFT and switch, MTP and decode profiles, cancel probe, cross-path probe. |
| `soak` | Two workers for 120 minutes. The slope verdict is the leak check: `leak` (rising away from a full pool), request errors, and missing monitor samples fail; `flat`, `falling`, and `saturation` pass. The 95% interval is widened for the autocorrelation of the per-minute floors. |
| `linux` | CTest the `build-spo-linux` tree in the CUDA 13.3 image, on GPU 1, skipping `_real`. A regex argument limits the tests. Without that build or image, it prints the toolchain blocker and returns 1. The host WSL install is still CUDA 12.4 without CMake. |

`quick` and `full` were run on 2026-10-11. `quick` passed 143 tests. `full` passed 15, and `ninfer_qwen3_5_grammar_real_test` skipped because no Qwen3.5 artifact is on this machine. Quick perplexity was 3.554 overall.

A test that contains `_real` loads the artifact and talks to the GPU. A skip of the Qwen3.5 real test is expected without that artifact. A failure of a Flash-Next real test is not.

## Health

`python tools/repo_health/health.py` takes `sizes`, `lang`, `layers`, `docs`, `dupes`, `report`, or `all`.

- `sizes` fails on a hard-cap breach that is not in `tools/repo_health/allowlist.txt`. Soft overages print.
- `lang` fails on CJK text outside the allowlist.
- `layers` fails on a new include or `getenv` violation that is not in `tools/repo_health/layer_baseline.txt`.
- `docs` fails when a backtick path under `src`, `include`, `docs`, `tools`, `apps`, `tests`, or `scripts` does not exist.
- `report` rewrites the size table in `docs/code-map.md`.

`health.py all` does not take an inventory file. Pass `--inventory` and `--ownership` when those CSVs should be checked. An inventory row fails only when its status is exactly `todo`. An ownership row fails when its action is not `keep`, `rewrite`, or `delete`, or when a `keep` row names no user.

## Updating a baseline

A new hard-cap file is split, not added to the allowlist, unless the split is already scheduled and the line says so. A new layer violation is fixed. `layer_baseline.txt` records violations that already existed and are not to be copied. After a real change in the size table, run `python tools/repo_health/health.py report` and keep the generated block.
