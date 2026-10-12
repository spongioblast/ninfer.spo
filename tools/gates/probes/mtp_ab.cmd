@echo off
setlocal
rem usage: mtp_ab.cmd <tag>   MTP draft-count A/B on the common corpus, 8K and 64K, dev bench
rem Exit code is 1 when any arm fails. GPU and NUMA node follow the caller when already set.
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=1"
set "CUDA_MODULE_LOADING=EAGER"
if not defined NINFER_NUMA_NODE set "NINFER_NUMA_NODE=1"
set "NINFER_DECODE_WAIT=spin"
if not defined NINFER_MODEL (echo set NINFER_MODEL to the .ninfer artifact & exit /b 1)
if not defined NINFER_CORPUS (echo set NINFER_CORPUS to common_mix.ids & exit /b 1)
set "OUT=%~dp0..\..\..\..\..\tmp\ninfer"
if not exist "%OUT%" mkdir "%OUT%"
set "BENCH=%~dp0..\..\..\..\build-spo-dev\bench\ninfer_bench.exe"
cd /d "%~dp0..\..\.."
set "FAILED=0"
for %%D in (auto 2 3 4) do (
  "%BENCH%" --weights "%NINFER_MODEL%" --corpus "%NINFER_CORPUS%" ^
    -pg "8192,512;65536,512" --max-ctx 66560 --prefill-chunk 8192 --warmup 1 -r 2 ^
    --spec mtp --draft-tokens %%D --lm-head-draft -o json --output-file "%OUT%\%1_d%%D.json" > "%OUT%\%1_d%%D.log" 2>&1
  if errorlevel 1 (
    echo arm d%%D failed, see %OUT%\%1_d%%D.log
    set "FAILED=1"
  )
)
exit /b %FAILED%
