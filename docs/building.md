# Building

The build compiles only `sm_120a`. CMake is 3.28 or newer, the host language is C++20, and the CUDA language standard is C++20. CMake does not pin a CUDA toolkit version. Ninja is the generator. There is no install step. Run the binaries from the build tree.

Windows and Linux both need the FFmpeg libraries `avcodec`, `avformat`, `avutil`, `swscale`, and `swresample`. On Windows those DLLs are taken from `ffmpeg/bin` and copied next to the binaries. On Linux the toolchain supplies them.

## Windows

`scripts/build.cmd` loads the Visual Studio 2022 x64 environment, configures, and builds. The build directory is `build-spo-dev` or `build-spo-release` next to this repo, not inside it.

```bat
scripts\build.cmd dev
scripts\build.cmd release
scripts\build.cmd dev --target ninfer-spo-serve
```

`dev` turns on tests and benchmarks. `trace` is `dev` plus `NINFER_PERFORMANCE_TRACE`. `release` builds the three apps only. A later argument is passed to `cmake --build`, which is how `--target` works.

The compiler database is written into the build directory (`CMAKE_EXPORT_COMPILE_COMMANDS`).

## GitHub

`.github/workflows/build.yml` builds the Windows release on each push to `main`. The runner installs CUDA 13.3 and an FFmpeg 9.0 LGPL shared build, then configures the same way as `scripts/build.cmd release`. It has no GPU, so the run only compiles. The `windows-release` artifact holds `ninfer-spo`, `ninfer-spo-serve`, `ninfer-spo-perplexity`, and the five FFmpeg DLLs those programs load.

## Linux

`scripts/build.sh` configures the same way. `CUDA_ARCH` overrides the architecture and must stay `120a` or CMake rejects it. `JOBS` is the parallel level (default 8).

```sh
scripts/build.sh dev
scripts/build.sh release
```

Linux here is compile and unit-test coverage. GPU runs stay on the Windows machine with the Blackwell GPU.

## CMake presets

`CMakePresets.json` has `dev` and `release`. Those presets write to `build/` inside the source tree. The scripts above are the ones this tree uses, because they keep the build beside the repo and pass `sm_120a` explicitly.
