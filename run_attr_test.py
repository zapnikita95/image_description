#!/usr/bin/env python3
"""Тестовый прогон атрибутов: 2 оффера, полный вывод."""
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_manager as pm
import feed_cache as fc
from attribute_detector import analyze_offer

def main():
    name = "Befree"
    db = pm.cache_db_path(name)
    if not db.exists():
        print("Cache not found:", db)
        return 1
    rdb = pm.results_db_path(name)
    offers = fc.get_offers(db, ["Женская / Домашние майки"], limit=0)
    existing = set()
    if rdb.exists():
        import sqlite3
        con = sqlite3.connect(str(rdb))
        existing = {r[0] for r in con.execute("SELECT offer_id FROM results").fetchall() if r[0]}
        con.close()
    offers = [o for o in offers if o.get("offer_id") not in existing]
    random.shuffle(offers)
    offers = offers[:2]
    cfg = pm.load_project(name)
    cfg["image_cache_dir"] = str(pm.image_cache_dir(name))
    cfg["model"] = cfg.get("model") or "clothes-detector-v1:latest"
    cfg["ollama_url"] = cfg.get("ollama_url") or "http://localhost:11434"

    print("Тестовый прогон атрибутов, модель:", cfg["model"])
    print("Офферов:", len(offers))
    for i, offer in enumerate(offers):
        oid = offer.get("offer_id")
        print("\n--- Оффер", i + 1, "offer_id=" + oid, "---")
        result = analyze_offer(offer, cfg, timeout=60)
        err = result.get("error")
        if err:
            print("Ошибка:", err)
        da = result.get("direction_attributes") or {}
        for dir_id, attrs in da.items():
            if not isinstance(attrs, dict):
                continue
            clean = {k: v for k, v in attrs.items() if k != "error"}
            print(dir_id + ":", json.dumps(clean, ensure_ascii=False, indent=2))
        print("avg_confidence:", result.get("avg_confidence"))
    print("\n--- Конец прогона ---")
    return 0

if __name__ == "__main__":
    sys.exit(main())
