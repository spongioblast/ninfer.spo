#!/bin/sh
# Start ninfer-spo-serve. Model path and port come from the environment.
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BUILD=${NINFER_BUILD:-$(CDPATH= cd -- "$ROOT/.." && pwd)/build-spo-release}
SERVE="$BUILD/apps/ninfer-spo-serve"
MODEL=${NINFER_MODEL:?set NINFER_MODEL to the .ninfer artifact}
PRESETS=${NINFER_PRESETS:-$(dirname "$0")/presets/pro6000_maxq_x4.json}
exec "$SERVE" "$MODEL" --host 127.0.0.1 --port "${NINFER_PORT:-8003}" \
  --model-id "${NINFER_MODEL_ID:-Qwen/Qwen3.8-Flash-Next}" \
  --context-cost-presets "$PRESETS" "$@"
