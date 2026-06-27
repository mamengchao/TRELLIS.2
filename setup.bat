@echo off
setlocal enabledelayedexpansion

call "E:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

REM 设置CUDA相关环境变量
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\lib\x64;%CUDA_HOME%\extras\CUPTI\lib64;%CUDA_HOME%\include;%CUDA_HOME%\libnvvp;%PATH%
set DISTUTILS_USE_SDK=1

REM 创建Python虚拟环境（使用系统默认python，若需指定版本可改为py -3.10）
python310.exe -m venv .venv
if errorlevel 1 (
    echo 错误：创建虚拟环境失败，请确认Python已安装且版本兼容。
    pause
    exit /b 1
)

REM 安装基础依赖
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -r requirements.txt

echo 所有操作已完成！
pause