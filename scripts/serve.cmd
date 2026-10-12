@echo off
setlocal
rem Repo root is the parent of scripts\. The binary name is the R1b rename.
set "ROOT=%~dp0.."
if not defined NINFER_BUILD set "NINFER_BUILD=..\build-spo-release"
rem NINFER_BUILD is relative to the repo root unless it is a drive or UNC path.
set "BUILD_DIR=%ROOT%\%NINFER_BUILD%"
if "%NINFER_BUILD:~1,1%"==":" set "BUILD_DIR=%NINFER_BUILD%"
if "%NINFER_BUILD:~0,2%"=="\\" set "BUILD_DIR=%NINFER_BUILD%"
set "SERVE=%BUILD_DIR%\apps\ninfer-spo-serve.exe"
rem NINFER_LM_HEAD_DRAFT=0 serves without the LM-head draft (for the served-draft A/B).
set "LM_HEAD_DRAFT=--lm-head-draft"
if "%NINFER_LM_HEAD_DRAFT%"=="0" set "LM_HEAD_DRAFT="
if not defined NINFER_MODEL (
  echo [ERROR] set NINFER_MODEL to the .ninfer artifact
  exit /b 1
)
set "MODEL=%NINFER_MODEL%"
set "PRESETS=%~dp0presets\pro6000_maxq_x4.json"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=1"
set "CUDA_MODULE_LOADING=EAGER"
if not defined NINFER_PORT set "NINFER_PORT=8003"
if not exist "%SERVE%" (echo [ERROR] %SERVE% not found. Build with scripts\build.cmd release & exit /b 1)
if not exist "%MODEL%" (echo [ERROR] model not found: %MODEL% & exit /b 1)
set "IFEO_ROOT=HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options"
reg query "%IFEO_ROOT%\ninfer-spo-serve.exe" >nul 2>&1 && if not defined NINFER_ALLOW_DEBUG_HEAP (
  echo [ERROR] ninfer-spo-serve.exe has an Image File Execution Options entry ^(debug heap^). Remove it or set NINFER_ALLOW_DEBUG_HEAP=1.
  exit /b 1
)
netstat -ano | findstr /r /c:":%NINFER_PORT% .*LISTENING" >nul && (echo [ERROR] port %NINFER_PORT% is in use & exit /b 1)
"%SERVE%" "%MODEL%" --host 127.0.0.1 --port %NINFER_PORT% --model-id Qwen/Qwen3.8-Flash-Next ^
  --device 0 --max-context 262144 --kv-capacity auto --kv-headroom-mib 4096 --kv-dtype bf16 ^
  --max-concurrency 2 --spec mtp --draft-tokens auto %LM_HEAD_DRAFT% --vision --preserve-thinking ^
  --prefill-chunk 8192 --context-cost-presets "%PRESETS%" %*
exit /b %ERRORLEVEL%
