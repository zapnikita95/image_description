#!/usr/bin/env python3
"""
Тест: ювелирка, задание «определить цвет», модель 9B.
Создаёт проект TestJewelry, кэширует фид yml-feed.6573.global.xml, прогоняет 5 офферов.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import project_manager as pm
import feed_cache as fc
from attribute_detector import analyze_offer, ensure_image_cached, normalize_ollama_url

FEED_JEWELRY = r"C:\Users\1\Downloads\yml-feed.6573.global.xml"
PROJECT_NAME = "TestJewelry"
NUM_OFFERS = 6  # берём по паре Фисташка + Малина + ещё
MODEL = "qwen3.5:9b"  # fallback, если в настройках приложения не задана модель

TASK_INSTRUCTION = (
    "Определи цвет строго по изображению, не по названию товара. "
    "По фото определи основной цвет украшения (металл или камень) и назови его по-русски. "
    "Верни JSON с полем color (значение — цвет на русском) и confidence (0-100)."
)


def main():
    print("=== Тест: ювелирка, цвет, модель 9B ===\n")

    # Проект: вертикаль + задание + одно направление (color)
    try:
        cfg = pm.create_project(
            PROJECT_NAME,
            feed_path="",
            vertical="Ювелирные изделия",
            task_instruction=TASK_INSTRUCTION,
            directions=[
                {
                    "id": "main",
                    "name": "Цвет",
                    "text_enabled": False,
                    "attributes": [{"key": "color", "label": "Цвет", "options": []}],
                    "custom_prompt": "",
                }
            ],
        )
        print("Создан проект:", PROJECT_NAME)
    except ValueError as e:
        if "already exists" in str(e).lower():
            cfg = pm.load_project(PROJECT_NAME)
            cfg["vertical"] = "Ювелирные изделия"
            cfg["task_instruction"] = TASK_INSTRUCTION
            cfg["directions"] = [
                {
                    "id": "main",
                    "name": "Цвет",
                    "text_enabled": False,
                    "attributes": [{"key": "color", "label": "Цвет", "options": []}],
                    "custom_prompt": "",
                }
            ]
            pm.save_project(cfg)
            print("Обновлён проект:", PROJECT_NAME)
        else:
            raise

    feed_path = Path(FEED_JEWELRY)
    if not feed_path.exists():
        print("Фид не найден:", FEED_JEWELRY)
        return 1

    db = pm.cache_db_path(PROJECT_NAME)
    print("Кэшируем фид...")
    summary = fc.parse_feed_to_cache(str(feed_path), db)
    print("Офферов в кэше:", summary["total"], "категорий:", len(summary["categories"]))
    pm.save_project({**cfg, "feed_path": str(feed_path)})

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

    # Берём и Фисташку (зелёные), и Малину (розовые) — проверяем оба цвета
    all_offers = fc.get_offers(db, limit=2000)
    fis = [o for o in all_offers if "фисташк" in (o.get("name") or "").lower()][:1]
    mal = [o for o in all_offers if "малин" in (o.get("name") or "").lower()][:1]
    seen_ids = set()
    offers = []
    for o in fis + mal:
        oid = o.get("offer_id")
        if oid and oid not in seen_ids:
            seen_ids.add(oid)
            offers.append(o)
    if len(offers) < 2:
        offers = fc.get_offers(db, limit=NUM_OFFERS)
    if not offers:
        print("Нет офферов в кэше.")
        return 1

    print("\nМодель:", cfg["model"], "| Офферов к анализу:", len(offers), "(Фисташка + Малина)", flush=True)
    print("---\n", flush=True)

    results_log = []
    for i, offer in enumerate(offers):
        oid = offer.get("offer_id", "")
        name = (offer.get("name") or "")[:60]
        print(f"[{i+1}/{len(offers)}] {oid} — {name}...", flush=True)
        result = analyze_offer(offer, cfg, timeout=90)
        err = result.get("error")
        if err:
            print("  Ошибка:", err[:120])
            continue
        da = result.get("direction_attributes") or {}
        for _dir_id, attrs in da.items():
            if not isinstance(attrs, dict) or attrs.get("error"):
                continue
            for k, v in attrs.items():
                if k == "error" or not isinstance(v, dict):
                    continue
                val = v.get("value", "?")
                conf = v.get("confidence", 0)
                print(f"  {k}: {val} (confidence {conf}%)")
                results_log.append(f"{name[:40]}: color={val} conf={conf}%")
        print("  avg_confidence:", result.get("avg_confidence", 0), "%")
        print()

    out_dir = ROOT / "generated" / "jewelry_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "last_run.txt").write_text("\n".join(results_log) if results_log else "no results", encoding="utf-8")
    print("=== Конец теста ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
