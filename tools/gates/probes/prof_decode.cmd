@echo off
setlocal
rem usage: prof_decode.cmd <output-name> [bench flags...]   (dev ninfer_bench under nsys, CUDA trace, round stats)
rem Exit code is the bench's, or the export's when the bench passed. GPU and NUMA node follow the caller when set.
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=1"
set "CUDA_MODULE_LOADING=EAGER"
if not defined NINFER_NUMA_NODE set "NINFER_NUMA_NODE=1"
set "NINFER_DECODE_WAIT=spin"
set "NINFER_DEBUG_ROUND_STATS=100"
if not defined NINFER_MODEL (echo set NINFER_MODEL to the .ninfer artifact & exit /b 1)
if not defined NINFER_CORPUS (echo set NINFER_CORPUS to common_mix.ids & exit /b 1)
if not defined NINFER_NSYS set "NINFER_NSYS=C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe"
set "NSYS=%NINFER_NSYS%"
set "OUT=%~dp0..\..\..\..\..\tmp\ninfer"
if not exist "%OUT%\prof" mkdir "%OUT%\prof"
set "BENCH=%~dp0..\..\..\..\build-spo-dev\bench\ninfer_bench.exe"
cd /d "%~dp0..\..\.."
set "ARGS=--weights %NINFER_MODEL% --corpus %NINFER_CORPUS% -pg 8192,512 --max-ctx 9216 --prefill-chunk 8192 --warmup 1 -r 1 -o json --output-file %OUT%\%1.json"
if "%NOPROF%"=="1" (
  "%BENCH%" %ARGS% %2 %3 %4 %5 %6 > "%OUT%\%1.log" 2>&1
  exit /b %ERRORLEVEL%
)
rem spin polls cudaEventQuery millions of times and floods CUPTI; sleep has the same wake p50 (LOG Phase 5)
set "NINFER_DECODE_WAIT=sleep"
"%NSYS%" profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true --output "%OUT%\prof\%1" ^
  "%BENCH%" %ARGS% %2 %3 %4 %5 %6 > "%OUT%\%1.log" 2>&1
if errorlevel 1 (
  echo profiled bench failed, see %OUT%\%1.log
  exit /b 1
)
"%NSYS%" export --type sqlite --force-overwrite true --output "%OUT%\prof\%1.sqlite" "%OUT%\prof\%1.nsys-rep" > "%OUT%\prof\%1_export.log" 2>&1
exit /b %ERRORLEVEL%
