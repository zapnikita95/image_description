#!/usr/bin/env python3
"""
CLI: замер analyze_offer для одной картинки (URL или путь к файлу).

Примеры:
  set IMAGE_DESC_PROFILE=1
  python scripts/profile_one_image.py https://example.com/p.jpg
  python scripts/profile_one_image.py C:\\pics\\x.jpg --model qwen2.5-vl:7b

Без IMAGE_DESC_PROFILE в ответе всё равно будет _profile, если передать --profile.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# корень репозитория
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import project_manager as pm  # noqa: E402
from attribute_detector import analyze_offer, image_analysis_profiling_enabled  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Профиль analyze_offer для одного изображения.")
    ap.add_argument("image", help="URL (http...) или путь к локальному файлу")
    ap.add_argument("--model", default=None, help="Модель Ollama (по умолчанию из app_settings)")
    ap.add_argument("--ollama-url", default=None, help="Базовый URL Ollama")
    ap.add_argument("--name", default="CLI test offer", help="Название оффера (в промпт)")
    ap.add_argument("--profile", action="store_true", help="Включить _profile без env IMAGE_DESC_PROFILE")
    ap.add_argument("--max-parallel", type=int, default=None, help="max_parallel_vision (0=без лимита)")
    ap.add_argument("--json", action="store_true", help="Печать только JSON результата")
    args = ap.parse_args()

    if args.profile:
        os.environ["IMAGE_DESC_PROFILE"] = "1"

    g = pm.get_global_settings()
    raw = (args.image or "").strip()
    if Path(raw).exists():
        pic = Path(raw).resolve().as_uri()
    else:
        pic = raw

    config = {
        **g,
        "directions": [
            {
                "id": "clothing",
                "name": "Одежда",
                "text_enabled": False,
                "attributes": [{"key": "sleeve_length", "label": "Рукав", "options": ["short", "long"]}],
                "custom_prompt": "",
            }
        ],
        "extract_inscriptions": False,
        "image_cache_dir": None,
    }
    if args.model:
        config["model"] = args.model
    if args.ollama_url:
        config["ollama_url"] = args.ollama_url
    if args.max_parallel is not None:
        config["max_parallel_vision"] = int(args.max_parallel)

    offer = {
        "offer_id": "profile_cli",
        "name": args.name,
        "picture_urls": [pic],
        "category": "",
    }

    if not args.json:
        print("image_analysis_profiling_enabled:", image_analysis_profiling_enabled())
        print("picture:", pic)
        print("model:", config.get("model"))
        print("ollama_url:", config.get("ollama_url"))
        print("max_parallel_vision:", config.get("max_parallel_vision", g.get("max_parallel_vision", 0)))
        print("---")

    result = analyze_offer(offer, config, timeout=600)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        prof = result.get("_profile")
        if prof:
            print(json.dumps(prof, ensure_ascii=False, indent=2))
        else:
            print("Нет _profile: задайте IMAGE_DESC_PROFILE=1 или --profile")
        if result.get("error"):
            print("error:", result["error"], file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
