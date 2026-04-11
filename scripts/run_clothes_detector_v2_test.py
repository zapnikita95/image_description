#!/usr/bin/env python3
"""Проверка модели clothes-detector-v2: 2 картинки, полный вывод атрибутов.
Использует локальный адаптер (путь к lora_out_2/lora_adapter). Требуется unsloth в окружении."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import project_manager as pm
from attribute_detector import analyze_offer

def main():
    # Картинки из датасета (одежда)
    img_dir = ROOT / "data" / "deepfashion" / "img"
    if not img_dir.exists():
        print("Нет папки data/deepfashion/img")
        return 1
    samples = []
    for sub in sorted(img_dir.iterdir())[:4]:
        if sub.is_dir():
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    samples.append(str(f.resolve()))
                    break
    samples = samples[:2]
    if not samples:
        print("Нет картинок в data/deepfashion/img")
        return 1

    # v2 через локальный адаптер (Ollama не грузит Qwen3-VL)
    adapter_dir = ROOT / "projects" / "Befree" / "lora_out_2" / "lora_adapter"
    if not adapter_dir.is_dir():
        adapter_dir = ROOT / "fine_tune" / "lora_out" / "lora_adapter"
    model_or_path = str(adapter_dir) if adapter_dir.is_dir() else "clothes-detector-v2:latest"

    config = {
        "model": model_or_path,
        "ollama_url": "http://localhost:11434",
        "image_max_size": 1024,
        "directions": pm.DEFAULT_DIRECTIONS,
    }

    print("Модель (или путь к адаптеру):", config["model"])
    print("Картинок:", len(samples))
    for i, path in enumerate(samples):
        offer = {
            "offer_id": f"test-{i}",
            "name": Path(path).parent.name,
            "picture_urls": [path],
            "category": "",
        }
        print("\n--- Картинка", i + 1, path, "---")
        result = analyze_offer(offer, config, timeout=90)
        if result.get("error"):
            print("Ошибка:", result["error"])
        da = result.get("direction_attributes") or {}
        for dir_id, attrs in da.items():
            if not isinstance(attrs, dict):
                continue
            err = attrs.get("error")
            if err:
                print(dir_id, "error:", err)
                continue
            clean = {k: v for k, v in attrs.items() if k != "error" and isinstance(v, dict)}
            print(dir_id + ":", json.dumps({k: v.get("value") for k, v in clean.items()}, ensure_ascii=False))
        print("avg_confidence:", result.get("avg_confidence"))
    print("\n--- Готово ---")
    return 0

if __name__ == "__main__":
    sys.exit(main())
