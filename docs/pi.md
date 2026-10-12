# Pi

Pi talks to this engine over HTTP as an OpenAI-compatible server. The model id it sends is `Qwen/Qwen3.8-Flash-Next`. `scripts/serve.cmd` is the Windows launcher that advertises that id.

The script reads `NINFER_MODEL` (the `.ninfer` artifact), `NINFER_PORT` (8003 if unset), and `NINFER_BUILD` (the build folder, relative to the repo root or absolute; `..\build-spo-release` if unset). `NINFER_LM_HEAD_DRAFT=0` leaves out `--lm-head-draft`. It selects GPU 1 by PCI bus order (it does not check the device name) and starts:

```bat
ninfer-spo-serve.exe <artifact> --host 127.0.0.1 --port %NINFER_PORT% --model-id Qwen/Qwen3.8-Flash-Next ^
  --device 0 --max-context 262144 --kv-capacity auto --kv-headroom-mib 4096 --kv-dtype bf16 ^
  --max-concurrency 2 --spec mtp --draft-tokens auto --lm-head-draft --vision --preserve-thinking ^
  --prefill-chunk 8192 --context-cost-presets scripts\presets\pro6000_maxq_x4.json
```

`--device 0` is the visible device after `CUDA_VISIBLE_DEVICES=1`. Preserve-thinking is on, so a client that wants the next turn to reuse the assistant prefix sends `reasoning_content` back. The structural prefix (tools, then the leading system message) is what a second chat with the same tools and system prompt reuses.

The binary comes from `NINFER_BUILD`, which defaults to `build-spo-release` beside this repo. Build that preset with `scripts\build.cmd release` before starting the script.
