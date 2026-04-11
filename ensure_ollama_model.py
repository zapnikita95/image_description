#!/usr/bin/env python3
"""Run from run.bat: pull Ollama model from app_settings.json if server is up."""
import json
import os
import subprocess
import sys
from pathlib import Path


def _find_ollama_exe_on_windows() -> str | None:
    """Проверяем те же пути, что и run.bat."""
    paths = [
        os.environ.get("LOCALAPPDATA", "") + r"\Programs\Ollama\ollama.exe",
        os.environ.get("ProgramFiles", r"C:\Program Files") + r"\Ollama\ollama.exe",
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)") + r"\Ollama\ollama.exe",
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Ollama\ollama.exe"),
        r"C:\ollama\ollama.exe",
    ]
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def main():
    base = Path(__file__).resolve().parent
    settings_path = base / "app_settings.json"
    model = "qwen3.5:35b"
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                model = json.load(f).get("model", model)
        except Exception:
            pass
    ollama_exe = os.environ.get("OLLAMA_EXE", "").strip() or "ollama"
    if ollama_exe == "ollama" and sys.platform == "win32":
        found = _find_ollama_exe_on_windows()
        if found:
            ollama_exe = found
    try:
        r = subprocess.run([ollama_exe, "list"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout:
            names = [line.split()[0] for line in r.stdout.strip().splitlines()[1:] if line.strip()]
            model_base = model.split(":")[0] if ":" in model else model
            if any(n.split(":")[0] == model_base or n == model for n in names):
                print(f"Model {model} already present.")
                return
    except Exception:
        pass
    print(f"Pulling model: {model}")
    try:
        r = subprocess.run([ollama_exe, "pull", model], check=False, timeout=3600)
        if r.returncode == 0:
            print(f"Model {model} ready.")
        else:
            print(f"Pull failed (code {r.returncode}). Если модель создана локально (ollama create), игнорируйте.", file=sys.stderr)
    except FileNotFoundError:
        print("Pull skipped: ollama not found.", file=sys.stderr)
    except Exception as e:
        print(f"Pull failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
