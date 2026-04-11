#!/usr/bin/env python3
"""Дымовой vision-прогон по URL (Ollama). Пример: python scripts/smoke_vision_urls.py <url>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import project_manager as pm
from attribute_detector import analyze_offer


def main() -> int:
    gs = pm.get_global_settings()
    model = (gs.get("model") or "qwen3.5:9b").strip()
    ollama_url = (gs.get("ollama_url") or "http://127.0.0.1:11435").strip()
    urls = [u.strip() for u in sys.argv[1:] if u.strip()]
    if not urls:
        urls = [
            "https://collect-static.tsum.com/sig/872f8cfa9b75bd10ebde712c5f8f07a9/height/1526/document-hub/01HA2J4G7QJK3KW2DA8WWNFP9M.jpg",
        ]
    cfg = {
        "model": model,
        "ollama_url": ollama_url,
        "image_max_size": int(gs.get("image_max_size") or 1024),
        "vertical": "Одежда",
        "directions": [d for d in pm.DEFAULT_DIRECTIONS if d.get("id") == "clothing"],
        "dynamic_clothing_attributes": False,
        "extract_inscriptions": True,
        "inscription_mode": "separate_call",
        "clothing_standard_keys_enabled": None,
        "attribute_value_llm_translate": False,
    }
    for i, url in enumerate(urls):
        offer = {
            "offer_id": f"smoke_{i}",
            "name": "Тестовый товар",
            "picture_urls": [url],
            "category": "Женское / Верхняя одежда / Пальто",
        }
        print(f"\n=== URL {i + 1} ===\n{url}\nмодель: {model}")
        result = analyze_offer(offer, cfg, timeout=180)
        if result.get("error"):
            print("error:", result["error"])
        da = result.get("direction_attributes") or {}
        for did, attrs in da.items():
            if not isinstance(attrs, dict):
                continue
            clean = {k: v for k, v in attrs.items() if k != "error"}
            print(json.dumps({did: clean}, ensure_ascii=False, indent=2))
        print("avg_confidence:", result.get("avg_confidence"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
