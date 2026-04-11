@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if not defined HF_HOME set "HF_HOME=C:\ollama"
if not defined HUGGINGFACE_HUB_CACHE set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
if not exist "venv\Scripts\activate.bat" (
  echo Creating venv...
  python -m venv venv
)
call "venv\Scripts\activate.bat"
echo.
echo ========== Python dependencies ==========
echo   pip install -r requirements.txt
echo   lxml: large or non-standard XML feeds
echo.
pip install -r requirements.txt -q
if errorlevel 1 (
  echo [FAIL] Dependencies failed. Check requirements.txt and internet.
  pause
  exit /b 1
)
python -c "import lxml.etree" 2>nul
if errorlevel 1 (
  echo [WARN] lxml not importable. Installing...
  pip install "lxml>=5.0.0" -q
  if errorlevel 1 (
    echo [FAIL] Could not install lxml. Feed tab may fail on large XML.
  ) else (
    echo [OK] lxml installed.
  )
) else (
  echo [OK] lxml
)
python -c "import torch" 2>nul
if errorlevel 1 (
  echo [WARN] torch not found. Installing PyTorch...
  where nvidia-smi >nul 2>&1
  if errorlevel 1 (
    pip install torch -q
  ) else (
    echo Detected NVIDIA driver. Installing PyTorch with CUDA...
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
  )
  if errorlevel 1 (
    echo [FAIL] Could not install torch.
    pause
    exit /b 1
  )
  echo [OK] torch installed.
) else (
  python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
  if errorlevel 1 (
    where nvidia-smi >nul 2>&1
    if not errorlevel 1 (
      echo [WARN] PyTorch without CUDA. For LoRA training install PyTorch with CUDA:
      echo   pip uninstall torch -y ^&^& pip install torch --index-url https://download.pytorch.org/whl/cu121
      echo.
    )
  )
)
echo.

echo ========== Checking environment ==========
python --version >nul 2>&1
if errorlevel 1 (
  echo [FAIL] Python not found. Install Python and add to PATH.
  pause
  exit /b 1
)
echo [OK] Python

python -c "import gradio" 2>nul
if errorlevel 1 (
  echo [FAIL] Gradio not installed. Run: pip install -r requirements.txt
  pause
  exit /b 1
)
echo [OK] Dependencies (gradio, etc.)
python -c "import torch" 2>nul
if errorlevel 1 (
  echo [FAIL] torch not installed. Run: pip install torch
  pause
  exit /b 1
)
echo [OK] torch

python -c "import unsloth" 2>nul
if errorlevel 1 (
  echo [WARN] unsloth not found. Installing for LoRA training...
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
  if errorlevel 1 (
    echo [WARN] unsloth install failed. LoRA training will not work.
  ) else (
    echo [OK] unsloth installed.
    pip install torchao==0.12.0 torchvision --index-url https://download.pytorch.org/whl/cu121 --no-deps -q 2>nul
    if not errorlevel 1 echo [OK] torchao/torchvision for compatibility.
  )
) else (
  echo [OK] unsloth
)
python -c "import flash_linear_attention" 2>nul
if errorlevel 1 (
  echo [INFO] flash-linear-attention not installed. Qwen3.5 MoE will use slower path. OK.
) else (
  echo [OK] flash-linear-attention
)
python -c "import torchvision" 2>nul
if errorlevel 1 (
  where nvidia-smi >nul 2>&1
  if not errorlevel 1 (
    pip install torchvision --index-url https://download.pytorch.org/whl/cu121 -q 2>nul
    pip install torchao==0.12.0 --no-deps -q 2>nul
  )
)

set OLLAMA_EXE=
where ollama >nul 2>&1 && set "OLLAMA_EXE=ollama"
if not defined OLLAMA_EXE if exist "C:\ollama\ollama.exe" set "OLLAMA_EXE=C:\ollama\ollama.exe"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
if not defined OLLAMA_EXE if exist "%ProgramFiles%\Ollama\ollama.exe" set OLLAMA_EXE=%ProgramFiles%\Ollama\ollama.exe
if not defined OLLAMA_EXE if exist "%ProgramFiles(x86)%\Ollama\ollama.exe" set OLLAMA_EXE=%ProgramFiles(x86)%\Ollama\ollama.exe
if not defined OLLAMA_EXE if exist "%USERPROFILE%\AppData\Local\Ollama\ollama.exe" set OLLAMA_EXE=%USERPROFILE%\AppData\Local\Ollama\ollama.exe
if exist "C:\ollama" set "OLLAMA_MODELS=C:\ollama"

if not defined OLLAMA_EXE (
  echo [WARN] Ollama not found. Trying winget install...
  echo.
  winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements 2>nul
  if errorlevel 1 (
    echo Winget failed. Trying to download installer...
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $url = 'https://ollama.com/download/OllamaSetup.exe'; $out = \"$env:TEMP\OllamaSetup.exe\"; try { Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing; Start-Process -FilePath $out -Wait -Verb RunAs } catch { Write-Host $_.Exception.Message; exit 1 }" 2>nul
  )
  timeout /t 3 /nobreak >nul
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
  if not defined OLLAMA_EXE if exist "%ProgramFiles%\Ollama\ollama.exe" set OLLAMA_EXE=%ProgramFiles%\Ollama\ollama.exe
  if not defined OLLAMA_EXE (
    echo.
    echo Could not install Ollama. Install from https://ollama.com/download
    echo Then run run.bat again.
    echo.
    pause
  )
)
if defined OLLAMA_EXE (
  echo [OK] Ollama: %OLLAMA_EXE%
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434', timeout=2)" 2>nul
  if errorlevel 1 (
    echo        Starting Ollama server...
    if "%OLLAMA_EXE%"=="ollama" (start "Ollama" ollama serve) else (start "Ollama" "%OLLAMA_EXE%" serve)
    echo        Waiting for Ollama...
    timeout /t 6 /nobreak >nul
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434', timeout=3)" 2>nul
    if errorlevel 1 (
      echo [WARN] Server starting. If analysis fails, wait 10s and retry.
    ) else (
      echo [OK] Ollama server ready.
    )
  ) else (
    echo [OK] Ollama already running on localhost:11434
  )
  echo.
  echo Ensuring model from app settings...
  python "%~dp0ensure_ollama_model.py"
)
echo ==========================================
echo.

if not exist "llama.cpp\convert_hf_to_gguf.py" if not exist "llama.cpp\convert-hf-to-gguf.py" (
  echo [OK] llama.cpp not found. Cloning for GGUF export...
  git clone --depth 1 https://github.com/ggerganov/llama.cpp
  if errorlevel 1 (
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
  )
  if errorlevel 1 (
    echo [WARN] git clone failed. Install Git from https://git-scm.com/download/win then run run.bat again.
  ) else (
    echo [OK] llama.cpp ready.
  )
  echo.
) else (
  echo [OK] llama.cpp present.
)
echo.

echo Starting Image analyzer...
python app.py
pause
