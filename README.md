# ninfer.spo

A fast low latency Windows and Linux CUDA inference engine for Qwen3.8 Flash-Next, Qwen3.5, and Gemma 4 31B on Blackwell (`sm_120`, RTX 6000 MaxQ and others). One process, one GPU, one resident model. The artifact is a V3 `.ninfer` file. The namespace stays `ninfer`.

The binaries are `ninfer-spo`, `ninfer-spo-serve`, and `ninfer-spo-perplexity`. They speak the local CLI and OpenAI- and Anthropic-compatible HTTP. Build with `scripts/build.cmd` on Windows or `scripts/build.sh` on Linux. The steps, the CUDA architecture, and FFmpeg are in `docs/building.md`. Flags and `NINFER_*` switches are in `docs/configuration.md`.

The architectures the registry accepts are `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`, `Qwen3_8FlashNextForCausalLM`, and `Gemma4ForCausalLM` (`src/models/registry.h`). Gemma 4 31B-it is the ported size: text, the MTP assistant, and the vision tower.

Every number below is from one RTX PRO 6000 Blackwell Max-Q (96 GB, 300 W cap, PCIe 5.0 x4) on Windows, one request at a time unless noted. Decode is generated tokens per second. TTFT is the time to the first token.

## Qwen3.8 Flash-Next, the Pi server

`scripts/serve.cmd` starts the server from the release build beside this repo. It loads the artifact in `NINFER_MODEL`, listens on `NINFER_PORT` (8003 if unset), and serves 262,144 context, bf16 KV, MTP with `auto` drafts and the proposal head, vision, preserve-thinking, and 16 device state slots. Set `NINFER_MODEL` to `qwen3_8_flash_next_125b_a6b_nvfp4_fp8proj_nvfp4mtp.ninfer`. The Pi wiring is in `docs/pi.md`. The engine is ready in 45 to 60 s, with 70.5 GiB of weights on the GPU and the 47.7 GiB PLE table locked in host RAM. Quick perplexity is 3.554.

### Build the artifact

That file is not published. Build it from [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594) at revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`:

```bash
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --local-dir /path/to/Qwen3.8-Flash-Next-NVFP4
```

The converter is in the NInfer 6000 tree (`ninfer`, beside this repo), not in this one. Run it from that tree with Python 3.11 and PyTorch. The output name is fixed; another name is rejected. RadixArk's PLE table is already FP8, so leave `--ple-format` unset. Routed experts stay the checkpoint's NVFP4 words. `--projection-format` stores the attention and GDN projections as FP8 E4M3 with one BF16 scale per row, and `--mtp-expert-format nvfp4` quantizes the drafter experts. The result is 126,890,325,248 bytes. On the 3090 this took about 13 minutes. Keep the entry file and every `.part-NNNN` file together.

```bash
python -m tools.convert.qwen3_8_flash_next_125b_a6b.convert \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --projection-format fp8_e4m3fn_row_bf16 \
  --mtp-expert-format nvfp4 \
  --device cuda \
  --out qwen3_8_flash_next_125b_a6b_nvfp4_fp8proj_nvfp4mtp.ninfer
```

### Speed

Replies by kind, from the release server (MTP `auto`). Pi sampling is what Pi sends: no temperature, top_k, or top_p.

| Reply | Greedy | Pi sampling | TTFT |
|---|---:|---:|---|
| Short Pi-style turns | 286 | 266 | 40–140 ms warm |
| Long code edit (up to 12k tokens) | 398 (358–483) | 291 | about 1.2 s cold, 50 ms on a repeat |
| Long English prose | 229 | 180 | |
| Long Japanese prose | 228 | 184 | |
| Long agent loop | 265 | 211 | |

Cold prefill of a short prompt, MTP `auto`, is 30 ms at 32 tokens, 48 ms at 128, 80 ms at 512, 187 ms at 2,048, and 354 ms at 4,096.

Long prompts, `ninfer_bench` on a mixed corpus of books, Wikipedia, and code, 256 new tokens. Measured 2026-10-29 on the current build:

| Prompt | Prefill | Prefill time | Decode, MTP off | Decode, MTP `auto` |
|---:|---:|---:|---:|---:|
| 512 | 6,400 tok/s | 80 ms | 162 | 244 |
| 8,192 | 12,200 tok/s | 0.67 s | 162 | 274 |
| 32,768 | 11,900 tok/s | 2.8 s | 146 | 211 |
| 65,536 | 11,600 tok/s | 5.6 s | 145 | 316 |
| 131,072 | 11,100 tok/s | 12 s | 140 | 251 |
| 261,632 | 10,800 tok/s | 24 s | 139 | 413 |

Long greedy continuations start repeating the corpus, so MTP accepts more drafts there (5.9 tokens per step at 262k, against 2.3 at 512 and 2.8 at 8k).

Pi-shaped chats on the server (system, tools, tool results). Cold is a new history. Extend is a few hundred new tokens. Repeat is the same request again:

| Prompt | Cold TTFT | Warm extend | Exact repeat |
|---:|---|---|---|
| 22k | 2.9 s | 0.19–0.21 s | 44 ms |
| 63k | 9.0 s | 0.21–0.35 s | 127 ms |
| 124k | 17 s | 0.25–0.34 s | 106 ms |
| 256k | 36 s | 0.29–0.41 s | 131 ms |

Four 60k sessions switched round-robin stay on their checkpoints: 0 of 12 turns recomputed, switch TTFT 0.22–0.31 s. Three 240k sessions do not fit together in the 447,040-token KV, so every switch recomputes the prompt (33–40 s).

Code gets the most tokens per verify step (5.3 on the edit prompts) and prose the fewest (2.4). A short request that arrives during a 117k prefill gets its first token in about 0.75 s instead of 0.11 s. A Pi history with many tool calls prefills slower than the bench, because every earlier assistant message adds a rewrite boundary (about 70 ms each).

### Smaller cards

The weights alone are 70.5 GiB of VRAM, so the normal setup needs the 96 GB card. With `--expert-vram-gib N`, only the hottest routed experts live in N GiB of VRAM. The rest stay pinned in host RAM: decode computes those misses on the CPU, and prefill copies them into VRAM first. The artifact is the same file.

What you need:

- **VRAM:** about 12 GiB for everything except the experts (32k bf16 KV, one request, no vision or proposal head), plus N. On a 32 GB card that leaves N = 19. A 19 GiB run here used 49 GiB in total with the default automatic KV size; set `--kv-capacity` to keep it small. Below 13 GiB is not measured.
- **Host RAM:** the 47.7 GiB PLE table plus the cold experts (44 GiB at N = 19, 50 GiB at 13), both page-locked, so about 128 GB. Windows pins about half the RAM minus 8 GiB, and startup stops if the pin fails.
- **CPU:** AVX-512 VNNI or AVX2 VNNI; there is a scalar fallback. `NINFER_EXPERT_CPU_THREADS` sets the workers (default 8).

Run the server directly rather than through `scripts/serve.cmd`, which asks for 262k context, vision, and 16 state slots:

```
ninfer-spo-serve qwen3_8_flash_next_125b_a6b_nvfp4_fp8proj_nvfp4mtp.ninfer \
    --port 8003 --model-id Qwen/Qwen3.8-Flash-Next \
    --max-context 32768 --kv-capacity 32768 --kv-dtype bf16 --max-concurrency 1 \
    --spec mtp --draft-tokens 3 --expert-vram-gib 19
```

Set `--kv-capacity` explicitly: on Windows `cudaMemGetInfo` reports too little free VRAM once the cold experts are mapped, so `auto` sizes from a wrong number. Startup prints `expert tier: vram budget …, N slots of 24576 experts, pinned cold … GiB`. A routing profile (`NINFER_EXPERT_PROFILE`) picks which experts stay in VRAM; without one the lowest expert ids stay. How to write one is in `docs/serving.md` under "Smaller cards, same artifact".

Short greedy replies on this card with the budget limited, `--draft-tokens 3`:

| Run | Prompts | tok/s | TTFT | Greedy text |
|---|---|---:|---|---|
| Every expert in VRAM | 16 short | 286 | 70–90 ms | reference |
| Every expert in VRAM | first 4 of those | 239 | about 80 ms | reference |
| CPU misses, 13 GiB resident, 8 threads | those 4 | 18 | 0.7 s | diverges |
| CPU misses, 19 GiB resident, 8 threads | those 4 | 15–19 | 0.2–1.0 s | |
| CPU misses, 21 GiB resident, 8 threads | those 4 | 13–18 | 0.2–1.0 s | |
| PCIe zero-copy misses, 13 GiB (earlier decode path) | those 4 | 7.4 | 1.9–2.4 s | matches |
| PCIe zero-copy misses, 21 GiB (earlier decode path) | 12 short | 11 | 1.1–1.2 s | matches |

Expect about 15–19 tok/s and a first token in up to a second, against 240–290 tok/s with every expert in VRAM. The 19 and 21 GiB CPU rows are the same four prompts as the 13 GiB row, measured 2026-10-29. Raising the budget from 13 to 21 GiB does not make those short replies faster. The PCIe rows are the earlier decode path, kept to compare the two. The CPU dot is fp32 against the GPU's FP4 tensor core, so greedy text can differ from a full-VRAM run.

## Gemma 4 31B, voice chat

Convert `gemma4_31b_fp8` with `--components text,vision,mtp` and `--override gemma4_fp8_a8` (`docs/weight-conversion.md`). That artifact has the FP8 A8 text trunk, one FP8 vocabulary table, the MTP assistant, and the vision tower. The text-only FP8 artifact has no vision. The NVFP4 recipes missed the quality gate.

```
ninfer-spo-serve <artifact> --port 8080 --model-id local --max-context 16384 --kv-dtype bf16 \
    --spec mtp --draft-tokens 4 --vision --no-thinking \
    --device-state-slots 4 --host-context-mib 8192
```

One request at a time. A client that omits sampling gets temperature 1.0, top_k 20, top_p 0.95; top_k above 20 is rejected. Four drafts is fastest for replies of 20 to 100 tokens; `--draft-tokens auto` (6) is faster once replies reach a few hundred tokens. The server uses 44.4 GiB at 16k context and 45.7 GiB at 32k. The checkpoint behaviour, the prewarm request, and the sizing are in `docs/serving.md` under "Gemma 4 for voice chat".

### Speed

Cold prefill on the voice artifact, measured 2026-10-29. MTP changes this by only a few milliseconds:

| Prompt | 32 | 96 | 512 | 4,096 | 16,384 | 65,536 |
|---|---:|---:|---:|---:|---:|---:|
| TTFT | 27 ms | 34 ms | 85 ms | 650 ms | 3.2 s | 24 s |

Decode, 256 greedy tokens:

| Prompt | MTP off | 4 drafts | MTP `auto` |
|---|---:|---:|---:|
| 32–512 tokens, code and docs | 45–46 | 180–193 | 220–252 |
| 4,096, code and docs | 44 | 147 | 155 |
| 4,096, one ShareGPT prompt | 44 | 169 | 204 |
| 16,384, code and docs | 42 | | 129 |
| 65,536, code and docs | 34 | | 66 |

The Gemma bench corpus is 65,536 tokens, so 128k and 262k were not measured. On chat prompts `auto` accepts more drafts than on the code and docs corpus (6.4 against 4.9 at 4k).

| Case | MTP off | MTP |
|---|---:|---:|
| Voice replies of 20–100 tokens, recorded session | 43 | 87 at 4 drafts, 80 at `auto` |
| 256-token chat, temperature 1.0 | 46 | 134 median at `auto` |

On a recorded 62-request voice session with 14 tools, the median TTFT is 104 ms at the client and 67 ms at the server. The p90 is 337 ms: those are the turns that add a tool result or a long user message. Only the first request is cold (about 1.1 s). The session computed 26k of its 484k prompt tokens; the rest came from checkpoints.

The QAT `q4_g32` artifact (19.9 GB) decodes 68 tok/s without MTP and 226 with MTP on code and docs, but on chat prompts MTP accepts 1.95 tokens per step and drops to 79 tok/s, so it is not the voice setup.

## License

This engine is derived from NInfer (https://github.com/Neroued/ninfer), including the Windows build notes checked against the headpiece747 ninfer-5090-windows tree. Those projects remain under their own copyrights. The license is Apache-2.0. See `NOTICE`.
