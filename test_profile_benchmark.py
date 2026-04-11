#!/usr/bin/env python3
"""
Бенчмарк analyze_offer на Ollama (по умолчанию qwen3.5:9b если есть).
Пишет в stdout и в benchmark_results.txt.

Важно на Windows: используйте http://127.0.0.1:11434 (curl к localhost часто висит).
Картинка — локальный JPEG (не внешние CDN).

  python test_profile_benchmark.py
  python test_profile_benchmark.py --skip-second --timeout 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULT_FILE = _ROOT / "benchmark_results.txt"
JSON_FILE = _ROOT / "benchmark_results.json"
LOCAL_JPEG = _ROOT / "_benchmark_local.jpg"


def _log(msg: str) -> None:
    line = msg.rstrip()
    print(line, flush=True)
    try:
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def ensure_local_jpeg() -> Path:
    if LOCAL_JPEG.exists() and LOCAL_JPEG.stat().st_size > 100:
        return LOCAL_JPEG
    try:
        from PIL import Image

        img = Image.new("RGB", (512, 512), color=(180, 90, 45))
        img.save(LOCAL_JPEG, "JPEG", quality=85)
    except ImportError:
        # 1x1 pixel JPEG (минимальный валидный)
        LOCAL_JPEG.write_bytes(
            bytes.fromhex(
                "ffd8ffe000104a46494600010100000100010000ffdb004300"
                "080606070605080707070909080a0c140d0c0b0b0c1912130f"
                "141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c3031"
                "3434341f27393d38323c2e333432ffdb0043010909090c0b0c"
                "180d0d1832211c213232323232323232323232323232323232"
                "32323232323232323232323232323232323232323232323232"
                "323232ffc00011080001000103011100021101031101ffc400"
                "14000100000000000000000000000000000008ffc400141001"
                "000000000000000000000000000000ffda000c030100021003"
                "10003f00d2cf20ffd9"
            )
        )
    return LOCAL_JPEG


def check_ollama(base: str) -> tuple[bool, str, list[str]]:
    import requests

    base = base.rstrip("/")
    try:
        r = requests.get(f"{base}/", timeout=(2, 5))
        r.raise_for_status()
    except Exception as e:
        return False, str(e), []

    try:
        r = requests.get(f"{base}/api/tags", timeout=(2, 8))
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models") or []:
            n = m.get("name") or m.get("model")
            if n:
                models.append(n)
        return True, "", sorted(models)
    except Exception as e:
        return True, f"tags: {e}", []


def check_loaded_models(base: str) -> list[dict]:
    """Проверка, какие модели уже в памяти Ollama."""
    import requests
    from attribute_detector import ollama_loaded_models
    try:
        return ollama_loaded_models(base, timeout=5)
    except Exception:
        return []


def pick_model(models: list[str], want: str) -> str:
    if want in models:
        return want
    low = [m.lower() for m in models]
    if want.lower() in low:
        return models[low.index(want.lower())]
    for m in models:
        if "9b" in m.lower() and "qwen" in m.lower():
            return m
    for m in models:
        if "qwen" in m.lower():
            return m
    return models[0] if models else want


def run_once(
    analyze_offer,
    model: str,
    ollama_url: str,
    max_parallel: int,
    image_path: Path,
    timeout: int,
    image_max_size: int = 512,  # Меньше для быстрее
) -> dict:
    config = {
        "model": model,
        "ollama_url": ollama_url,
        "image_max_size": image_max_size,
        "max_parallel_vision": max_parallel,
        "dynamic_clothing_attributes": False,
        "extract_inscriptions": False,
        "directions": [
            {
                "id": "clothing",
                "name": "Одежда",
                "text_enabled": False,
                "attributes": [
                    {"key": "sleeve_length", "label": "Рукав", "options": ["short", "long"]},
                    {"key": "material", "label": "Материал", "options": ["cotton", "polyester"]},
                    {"key": "color", "label": "Цвет", "options": []},
                ],
            }
        ],
    }
    offer = {
        "offer_id": "bench_1",
        "name": "Bench item",
        "picture_urls": [str(image_path.resolve())],
        "category": "test",
    }
    t0 = time.perf_counter()
    result = analyze_offer(offer, config, timeout=timeout)
    wall_ms = (time.perf_counter() - t0) * 1000
    prof = dict(result.get("_profile") or {})
    prof["wall_including_python_ms"] = round(wall_ms, 2)
    prof["error"] = result.get("error")
    prof["model"] = model
    return prof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--skip-second", action="store_true")
    ap.add_argument("--image-size", type=int, default=512, help="image_max_size (512=fast, 1024=default, 2048=slow)")
    ap.add_argument("--skip-warmup", action="store_true", help="Skip model warmup (for testing cold start)")
    args = ap.parse_args()

    if RESULT_FILE.exists():
        RESULT_FILE.unlink()

    os.environ["IMAGE_DESC_PROFILE"] = "1"
    # Импорт только после установки env (чтобы профилирование точно включилось)
    from attribute_detector import analyze_offer as ao

    _log("=== benchmark start ===")
    _log(f"ollama_url={args.ollama_url}")

    ok, err, models = check_ollama(args.ollama_url)
    if not ok:
        _log(f"FAIL: Ollama not reachable: {err}")
        _log("Hint: start Ollama; use http://127.0.0.1:11434 on Windows")
        return 1

    _log(f"Ollama OK; models count={len(models)}")
    if models:
        _log("first models: " + ", ".join(models[:10]))

    model = pick_model(models, args.model)
    if model not in models and models:
        _log(f"WARN: {args.model!r} not in list; using {model!r}")
    else:
        _log(f"using model: {model}")

    # Проверка, загружена ли модель в память
    loaded = check_loaded_models(args.ollama_url)
    model_loaded = any((m.get("name") or m.get("model")) == model for m in loaded)
    if model_loaded:
        _log(f"Model {model} already in memory (VRAM)")
    else:
        _log(f"Model {model} NOT in memory — will load on first request (slow!)")

    # Прогрев модели перед бенчмарком
    if not args.skip_warmup:
        _log("Warming up model (first request may take 30-120s to load into VRAM)...")
        from attribute_detector import warmup_ollama_model
        warmup_config = {"model": model, "ollama_url": args.ollama_url}
        try:
            warmup_start = time.perf_counter()
            warmup_ollama_model(warmup_config, timeout=180)
            warmup_ms = (time.perf_counter() - warmup_start) * 1000
            _log(f"Warmup completed in {warmup_ms:.0f} ms ({warmup_ms/1000:.1f} sec)")
        except Exception as e:
            _log(f"Warmup failed (continuing anyway): {e}")
    else:
        _log("Skipping warmup (--skip-warmup)")

    img_path = ensure_local_jpeg()
    _log(f"local image: {img_path} ({img_path.stat().st_size} bytes)")

    all_rows: list[dict] = []

    for max_par in (0, 1):
        if max_par == 1 and args.skip_second:
            break
        _log(f"--- run max_parallel_vision={max_par} image_size={args.image_size} ---")
        try:
            prof = run_once(ao, model, args.ollama_url, max_par, img_path, args.timeout, args.image_size)
            all_rows.append(prof)
            vc = prof.get("vision_calls") or []
            _log(
                f"total_wall_ms={prof.get('total_wall_ms')} "
                f"image_prep_ms={prof.get('image_prep_ms')} "
                f"sum_vision_ms={prof.get('vision_calls_sum_ms')} "
                f"n_calls={len(vc)} "
                f"err={prof.get('error')!r}"
            )
            for c in vc:
                if isinstance(c, dict):
                    _log(f"  call {c.get('task')} [{c.get('backend')}]: {c.get('ms')} ms")
        except Exception as e:
            _log(f"FAIL run: {e}")
            all_rows.append({"error": str(e), "max_parallel_vision": max_par})

    JSON_FILE.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"wrote {JSON_FILE.name}")

    rows_ok = [r for r in all_rows if r.get("total_wall_ms") is not None]
    if len(rows_ok) >= 2:
        a, b = rows_ok[0], rows_ok[1]
        t0, t1 = float(a["total_wall_ms"]), float(b["total_wall_ms"])
        if t1 < t0:
            _log(f"parallel=1 faster than unlimited by ~{(1 - t1/t0)*100:.1f}% (wall total_wall_ms)")
        elif t0 < t1:
            _log(f"unlimited faster than parallel=1 by ~{(1 - t0/t1)*100:.1f}%")

    _log("=== benchmark end ===")
    return 0 if rows_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
