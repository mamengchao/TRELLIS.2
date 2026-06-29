@echo off
setlocal enabledelayedexpansion

call "E:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

REM 设置CUDA相关环境变量
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\lib\x64;%CUDA_HOME%\extras\CUPTI\lib64;%CUDA_HOME%\include;%CUDA_HOME%\libnvvp;%PATH%
set DISTUTILS_USE_SDK=1

if not exist ".\.venv\Scripts\python.exe" (
  echo virutalenv not exists, begin create...
  python310.exe -m venv .venv
  if errorlevel 1 (
      echo error：create virtualenv failed。
      pause
      exit /b 1
  )
)

set "pip_install=.\.venv\Scripts\python.exe -m pip install --no-build-isolation"

@REM %pip_install% --upgrade pip setuptools wheel

%pip_install% typing_extensions==4.15.0

%pip_install% torch==2.6.0 torchvision==0.21.0 --index-url https://mirror.nju.edu.cn/pytorch/whl/cu124

%pip_install% einops triton-windows==3.2.0.post21 wheel


%pip_install% -r requirements.txt

%pip_install% git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8#egg=utils3d


%pip_install% https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4%%2Bcu124torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl


%pip_install% https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0+cu124torch2.6-cp310-cp310-win_amd64.whl

%pip_install% git+https://github.com/JeffreyXiang/nvdiffrec.git@renderutils#egg=nvdiffrec_render

%pip_install% https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh-latest/cumesh-0.0.1+cu124torch2.6-cp310-cp310-win_amd64.whl

%pip_install% https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm-latest/flex_gemm-1.0.0+cu124torch2.6-cp310-cp310-win_amd64.whl

%pip_install% https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel-latest/o_voxel-0.0.1+cu124torch2.6-cp310-cp310-win_amd64.whl


echo setup complete
pause