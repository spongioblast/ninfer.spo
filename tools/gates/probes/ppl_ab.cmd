@echo off
setlocal
rem usage: ppl_ab.cmd <dev|release> <output-name>
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=1"
set "CUDA_MODULE_LOADING=EAGER"
if not defined NINFER_NUMA_NODE set "NINFER_NUMA_NODE=1"
set "NINFER_DECODE_WAIT=spin"
if not defined NINFER_MODEL (echo set NINFER_MODEL to the .ninfer artifact & exit /b 1)
if not defined NINFER_PPL_CORPUS (echo set NINFER_PPL_CORPUS to the perplexity manifest & exit /b 1)
set "ROOT_OUT=%~dp0..\..\..\..\..\tmp\ninfer"
if not exist "%ROOT_OUT%" mkdir "%ROOT_OUT%"
set "OUT=%ROOT_OUT%\%2"
set "PPL=%~dp0..\..\..\..\build-spo-%1\apps\ninfer-spo-perplexity.exe"
if not exist "%PPL%" (echo %PPL% not found & exit /b 1)
"%PPL%" "%NINFER_MODEL%" --corpus "%NINFER_PPL_CORPUS%" --quick --kv-dtype bf16 --output "%OUT%" > "%OUT%.out" 2> "%OUT%.err"
exit /b %ERRORLEVEL%
