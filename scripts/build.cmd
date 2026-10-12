@echo off
setlocal
rem Source tree is the parent of this scripts directory. Builds land next to the repo.
set "SRC=%~dp0.."
set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=dev"
set "BUILD=%SRC%\..\build-spo-%PRESET%"
rem Compiler temp files stay beside the build (NINFER_BUILD_TEMP overrides), not on the system drive.
if not defined NINFER_BUILD_TEMP set "NINFER_BUILD_TEMP=%BUILD%\.tmp"
if not exist "%NINFER_BUILD_TEMP%" mkdir "%NINFER_BUILD_TEMP%"
set "TEMP=%NINFER_BUILD_TEMP%"
set "TMP=%NINFER_BUILD_TEMP%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
where cmake || exit /b 1
where ninja || exit /b 1
set "EXTRA=-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
if /i "%PRESET%"=="dev" set "EXTRA=%EXTRA% -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON"
if /i "%PRESET%"=="trace" set "EXTRA=%EXTRA% -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_PERFORMANCE_TRACE=ON"
cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_BUILD_APPS=ON %EXTRA% || exit /b 1
set "BUILD_ARGS="
:collect_args
shift
if "%~1"=="" goto args_collected
set "BUILD_ARGS=%BUILD_ARGS% %1"
goto collect_args
:args_collected
cmake --build "%BUILD%" --parallel 16 %BUILD_ARGS% || exit /b 1
for %%D in (avcodec avformat avutil swscale swresample) do xcopy /y /q "%SRC%\ffmpeg\bin\%%D-*.dll" "%BUILD%\apps\" >nul
for %%D in (avcodec avformat avutil swscale swresample) do if exist "%BUILD%\bench" xcopy /y /q "%SRC%\ffmpeg\bin\%%D-*.dll" "%BUILD%\bench\" >nul
for %%D in (avcodec avformat avutil swscale swresample) do if exist "%BUILD%\tests" xcopy /y /q "%SRC%\ffmpeg\bin\%%D-*.dll" "%BUILD%\tests\" >nul
echo BUILD_OK %BUILD%
