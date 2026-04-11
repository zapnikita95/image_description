#!/usr/bin/env python3
"""
Тест: цвет металла украшения (не камней), модель по умолчанию qwen3.5:9b.
Два эталонных URL + опционально офферы из кэша фида.
Результат: generated/metal_color_test/last_run.txt
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import project_manager as pm
import feed_cache as fc
from attribute_detector import analyze_offer, normalize_ollama_url

PROJECT_NAME = "TestMetalColor"
MODEL = "qwen3.5:9b"

# Как в интерфейсе «Запуск» — задание + одно поле JSON
TASK_INSTRUCTION = (
    "Определить цвет металла изделия. НЕ нужно определять цвет камней, только металл! "
    "Смотри на оправу, цепь, швензу, застёжку — не на оттенок вставок. "
    "Если видимого металла нет (только камни/эмаль/без металла) — верни в поле metal_color значение «нет металла» и уверенность. "
    "Значение metal_color по-русски: например золото, серебро, розовое золото, платина, сталь, латунь, бронза, медь, родий/чёрный металл — что реально видно. "
    "Верни строго JSON: объект с ключами metal_color (строка) и confidence (0-100). Без markdown."
)

# Эталоны пользователя
FIXTURES = [
    {
        "offer_id": "fixture_silver_21",
        "name": "Эталон: ожидается серебро",
        "picture_urls": ["https://static.insales-cdn.com/images/products/1/4737/2739745409/21.jpg"],
        "category": "jewelry",
    },
    {
        "offer_id": "fixture_gold_73",
        "name": "Эталон: ожидается золото",
        "picture_urls": ["https://static.insales-cdn.com/images/products/1/4089/617730041/73_1.jpg"],
        "category": "jewelry",
    },
]


def _ensure_project():
    try:
        cfg = pm.create_project(
            PROJECT_NAME,
            feed_path="",
            vertical="Ювелирные изделия",
            task_instruction=TASK_INSTRUCTION,
            directions=[
                {
                    "id": "main",
                    "name": "Металл",
                    "text_enabled": False,
                    "attributes": [{"key": "metal_color", "label": "Цвет металла", "options": []}],
                    "custom_prompt": "",
                }
            ],
        )
        print("Создан проект:", PROJECT_NAME)
        return cfg
    except ValueError as e:
        if "already exists" not in str(e).lower():
            raise
        cfg = pm.load_project(PROJECT_NAME)
        cfg["vertical"] = "Ювелирные изделия"
        cfg["task_instruction"] = TASK_INSTRUCTION
        cfg["directions"] = [
            {
                "id": "main",
                "name": "Металл",
                "text_enabled": False,
                "attributes": [{"key": "metal_color", "label": "Цвет металла", "options": []}],
                "custom_prompt": "",
            }
        ]
        pm.save_project(cfg)
        print("Обновлён проект:", PROJECT_NAME)
        return cfg


def main():
    print("=== Тест: цвет металла (не камней), qwen3.5:9b ===\n")
    cfg = _ensure_project()

    g = pm.get_global_settings()
    cfg["model"] = g.get("model") or MODEL
    cfg["ollama_url"] = g.get("ollama_url") or "http://localhost:11434"
    cfg["image_max_size"] = g.get("image_max_size", 1024)
    cfg["image_cache_dir"] = str(pm.image_cache_dir(PROJECT_NAME))

    url = normalize_ollama_url(cfg["ollama_url"] or "")
    try:
        import requests
        r = requests.get(url.rstrip("/") + "/", timeout=3)
        r.raise_for_status()
    except Exception as e:
        print("Ollama недоступна:", e)
        return 1

    offers = list(FIXTURES)
    # Добавить до 3 офферов из кэша, если фид уже кэшировали для этого проекта
    db = pm.cache_db_path(PROJECT_NAME)
    if Path(db).exists():
        extra = fc.get_offers(db, limit=15)
        seen = {o["offer_id"] for o in offers}
        for o in extra:
            oid = o.get("offer_id")
            if oid and oid not in seen and o.get("picture_urls"):
                seen.add(oid)
                offers.append(o)
            if len(offers) >= 5:
                break

    print("Модель:", cfg["model"], "| офферов:", len(offers), flush=True)
    print("---\n", flush=True)

    lines = []
    for i, offer in enumerate(offers):
        oid = offer.get("offer_id", "")
        name = (offer.get("name") or "")[:70]
        print(f"[{i+1}/{len(offers)}] {oid} — {name}...", flush=True)
        result = analyze_offer(offer, cfg, timeout=120)
        if result.get("error"):
            print("  Ошибка:", result["error"][:200])
            lines.append(f"{oid}: ERROR {result['error'][:80]}")
            continue
        da = result.get("direction_attributes") or {}
        for _did, attrs in da.items():
            if not isinstance(attrs, dict):
                continue
            if attrs.get("error"):
                print("  dir error:", attrs["error"][:120])
                continue
            mc = attrs.get("metal_color") or attrs.get("color")
            if isinstance(mc, dict):
                val = mc.get("value", "?")
                conf = mc.get("confidence", 0)
                print(f"  metal_color: {val} ({conf}%)")
                lines.append(f"{oid}: metal_color={val} conf={conf}% | {name[:40]}")
        print("  avg_confidence:", result.get("avg_confidence", 0), "%\n")

    out_dir = ROOT / "generated" / "metal_color_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "last_run.txt").write_text("\n".join(lines) if lines else "no results", encoding="utf-8")
    print("Записано:", out_dir / "last_run.txt")
    print("=== Конец ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
