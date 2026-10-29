# Pi

Pi talks to this engine over HTTP as an OpenAI-compatible server. The model id it sends is `Qwen/Qwen3.8-Flash-Next`.

The Windows launcher is `scripts/serve.cmd`. It reads `NINFER_MODEL` (the `.ninfer` artifact), `NINFER_PORT` (8003 if unset), and `NINFER_BUILD` (the build folder, relative to the repo root or absolute; `..\build-spo-release` if unset). `NINFER_LM_HEAD_DRAFT=0` leaves out `--lm-head-draft`. The script selects GPU 1 by PCI bus order (it does not check the device name), so `--device 0` is that visible device. The arguments, including 16 device state slots, are in the script.

Preserve-thinking is on, so a client that wants the next turn to reuse the assistant prefix sends `reasoning_content` back. The structural prefix (tools, then the leading system message) is what a second chat with the same tools and system prompt reuses.

Build the release preset with `scripts\build.cmd release` before starting the script. The binary comes from `NINFER_BUILD`.
