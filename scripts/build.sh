#!/bin/sh
# Configure and build from the repo root. The build directory sits next to the repo.
set -e
SRC=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
PRESET=${1:-dev}
BUILD=$(CDPATH= cd -- "$SRC/.." && pwd)/build-spo-$PRESET
EXTRA="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
if [ "$PRESET" = "dev" ]; then
  EXTRA="$EXTRA -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON"
fi
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-120a}" -DNINFER_BUILD_APPS=ON $EXTRA
cmake --build "$BUILD" --parallel "${JOBS:-8}"
echo "BUILD_OK $BUILD"
