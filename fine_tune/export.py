#!/usr/bin/env python3
"""
Export LoRA adapter to GGUF and register in Ollama via 'ollama create'.

Steps:
  1. Merge LoRA adapter with base model (unsloth)
  2. Convert to GGUF (llama.cpp)
  3. Write Modelfile
  4. Run: ollama create <model_name> -f Modelfile

Usage:
  python fine_tune/export.py --adapter fine_tune/lora_out/lora_adapter \
      --base-model Qwen/Qwen2.5-VL-3B-Instruct \
      --output fine_tune/gguf_out \
      --ollama-name clothes-detector-v1
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


SYSTEM_PROMPT = (
    "Ты — эксперт по анализу одежды. "
    "По фотографии определяй характеристики одежды и надписи в строгом JSON-формате."
)


def _sanitize_gguf_filename(name: str) -> str:
    """Windows forbids : and other chars in filenames. Ollama allows 'name:tag'."""
    if not name or not name.strip():
        return "model"
    s = name.strip().replace(":", "-").replace("/", "-").replace("\\", "-")
    s = "".join(c for c in s if c not in '<>|?*"')
    return s.strip() or "model"


def _path_for_ollama(p: Path) -> str:
    """Path for Ollama CLI on Windows: forward slashes so 'filename/dir' syntax is not broken."""
    return p.resolve().as_posix()


_ANSI_ESCAPE = re.compile(r"\x1b\[[?0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07?")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text).strip()


def _do_merge(model, tok, processor, merged_dir, _log):
    """Merge LoRA adapter into base weights and save as 16-bit safetensors."""
    merged_dir = Path(merged_dir)

    def _sets_to_lists(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _sets_to_lists(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_sets_to_lists(x) for x in obj)
        return obj

    try:
        if hasattr(model, "config") and model.config is not None:
            d = getattr(model.config, "to_dict", lambda: model.config.__dict__)
            safe = _sets_to_lists(d() if callable(d) else model.config.__dict__)
            if isinstance(safe, dict):
                for k, v in safe.items():
                    setattr(model.config, k, v)
    except Exception:
        pass

    if hasattr(model, "save_pretrained_merged"):
        try:
            model.save_pretrained_merged(str(merged_dir), tok, save_method="merged_16bit", maximum_memory_usage=0.7)
        except TypeError as te:
            if "set" in str(te).lower() or "unhashable" in str(te).lower():
                model.save_pretrained_merged(str(merged_dir), tok, save_method="merged_16bit")
            else:
                raise
    else:
        model.save_pretrained(str(merged_dir))
        tok.save_pretrained(str(merged_dir))
    if processor is not None and processor is not tok and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(merged_dir))
    _log(f"  Merged model saved to {merged_dir}")


def _get_convert_script(_log) -> Path:
    """Find or download llama.cpp convert-hf-to-gguf.py."""
    llama_root = Path(__file__).parent.parent / "llama.cpp"
    for name in ("convert-hf-to-gguf.py", "convert_hf_to_gguf.py"):
        candidate = llama_root / name
        if candidate.exists():
            return candidate
    cwd_script = Path("convert_hf_to_gguf.py")
    if cwd_script.exists():
        return cwd_script.resolve()
    cache_dir = Path(__file__).parent.parent / ".llama_convert_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    script_path = cache_dir / "convert-hf-to-gguf.py"
    if script_path.exists():
        return script_path
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert-hf-to-gguf.py"
        _log(f"  Скачиваю скрипт конвертации: {url}")
        urllib.request.urlretrieve(url, script_path)
        return script_path
    except Exception as e:
        raise FileNotFoundError(
            f"llama.cpp convert script не найден и не удалось скачать: {e}. "
            "Склонируйте: git clone https://github.com/ggerganov/llama.cpp"
        ) from e


def _gguf_via_llamacpp(merged_dir, gguf_path, _log):
    """Convert merged safetensors to GGUF using llama.cpp convert script."""
    merged_dir = Path(merged_dir)
    gguf_path = Path(gguf_path)
    convert_script = _get_convert_script(_log)
    result = subprocess.run(
        [sys.executable, str(convert_script), str(merged_dir), "--outfile", str(gguf_path), "--outtype", "q8_0"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed:\n{result.stderr}")
    _log(f"  GGUF saved to {gguf_path}")


def export_to_gguf(
    adapter_dir: str | Path,
    base_model: str,
    output_dir: str | Path,
    ollama_model_name: str = "clothes-detector",
    progress_callback=None,
) -> dict:
    """
    Merge adapter, export GGUF, register in Ollama.
    Returns {success, gguf_path, ollama_name, error}.
    """
    adapter_dir = Path(adapter_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not adapter_dir.exists():
        return {"success": False, "error": f"Adapter not found: {adapter_dir}"}

    def _log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    gguf_basename = _sanitize_gguf_filename(ollama_model_name) + ".gguf"
    gguf_path = output_dir / gguf_basename

    # ── Step 1: Load adapter, merge to 16-bit, convert to GGUF ───────────────
    # save_pretrained_gguf in Unsloth requires Git/winget to clone llama.cpp; we use merge + convert script instead.
    _log("Step 1/3: Loading adapter and merging to 16-bit...")
    try:
        is_vl = True
        try:
            from unsloth import FastVisionModel
            model, processor = FastVisionModel.from_pretrained(
                model_name=str(adapter_dir),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,
            )
            tok = processor
        except Exception:
            is_vl = False
            from unsloth import FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(
                model_name=str(adapter_dir),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,
            )
            processor = None

        _log(f"  Model loaded ({'VL/vision' if is_vl else 'text-only'}). Merge → GGUF...")
        merged_dir = output_dir / "merged"
        _do_merge(model, tok, processor, merged_dir, _log)
        _log("Step 2/3: Converting to GGUF...")
        _gguf_via_llamacpp(merged_dir, gguf_path, _log)

    except ImportError as e:
        return {"success": False, "error": f"unsloth not installed: {e}"}
    except Exception as e:
        return {"success": False, "error": f"GGUF export failed: {e}"}

    # ── Step 3: Create Modelfile and register in Ollama ───────────────────
    # Windows: path with backslashes and/or colon in filename breaks ollama create.
    # Use only relative path in Modelfile and run ollama from output_dir.
    _log("Step 3/3: Registering model in Ollama...")
    modelfile_path = output_dir / "Modelfile"
    gguf_relative = gguf_path.name  # already sanitized, no colons
    modelfile_content = f"""FROM ./{gguf_relative}

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 512
"""
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    if not gguf_path.exists():
        return {
            "success": False,
            "error": f"GGUF файл не найден: {gguf_path}. Конвертация в GGUF могла упасть выше.",
        }

    # Windows: pass Modelfile path with forward slashes and run from output_dir so FROM ./file.gguf resolves
    modelfile_arg = _path_for_ollama(modelfile_path)
    _log(f"  Запуск: ollama create ... -f {modelfile_arg} (из папки {output_dir})")

    try:
        result = subprocess.run(
            ["ollama", "create", ollama_model_name, "-f", modelfile_arg],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(output_dir),
        )
        if result.returncode != 0:
            err_parts = [result.stderr or "", result.stdout or ""]
            err_body = _strip_ansi("\n".join(p.strip() for p in err_parts if p.strip()))
            return {
                "success": False,
                "error": f"ollama create failed:\n{err_body}\n\nModelfile: {modelfile_path}\nПапка: {output_dir}",
            }
        _log(f"  Model '{ollama_model_name}' registered in Ollama")
    except FileNotFoundError:
        return {
            "success": False,
            "error": "ollama command not found — install Ollama first",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "ollama create timed out"}

    return {
        "success": True,
        "gguf_path": str(gguf_path),
        "ollama_name": ollama_model_name,
        "modelfile": str(modelfile_path),
        "error": None,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output", default="fine_tune/gguf_out")
    p.add_argument("--ollama-name", default="clothes-detector-v1")
    args = p.parse_args()

    result = export_to_gguf(
        adapter_dir=args.adapter,
        base_model=args.base_model,
        output_dir=args.output,
        ollama_model_name=args.ollama_name,
        progress_callback=print,
    )
    if result["success"]:
        print(f"\nSuccess! Model '{result['ollama_name']}' is now available in Ollama.")
        print(f"Run: ollama run {result['ollama_name']}")
    else:
        print(f"\nFailed: {result['error']}", file=sys.stderr)
        sys.exit(1)
