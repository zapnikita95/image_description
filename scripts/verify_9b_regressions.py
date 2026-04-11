#!/usr/bin/env python3
"""
Прогон qwen3.5:9b по URL из экспортов (Tsum «кнопки», Zolla «пуговицы») и регрессии промптов.

  set OLLAMA_VISION_MODEL=qwen3.5:9b   # по умолчанию так
  python scripts/verify_9b_regressions.py
  python scripts/verify_9b_regressions.py --only=0,1   # только кейсы с индексами 0 и 1
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy

import project_manager as pm
from attribute_detector import analyze_offer


def _pick(attrs: dict, key: str) -> str:
    b = (attrs or {}).get(key)
    if isinstance(b, dict):
        return (b.get("value") or "").strip()
    return ""


CASES: list[dict] = [
    # Tsum Collect: в CSV застёжка была «кнопки» — часто должны быть швейные пуговицы
    {
        "label": "TSUM рубашка (в экспорте: кнопки)",
        "url": "https://collect-static.tsum.com/sig/da7666699ef8dd66911acda34e6c2404/height/1526/document-hub/01HA9R3CBNK506Z6FH28MVYACV.jpg",
        "name": "Хлопковая рубашка",
        "category": "Женское / Блузки и рубашки / Рубашки",
        "old_fastener": "кнопки",
    },
    {
        "label": "TSUM пальто (в экспорте: кнопки)",
        "url": "https://collect-static.tsum.com/sig/6395624f90678249b62b19c67cd727e8/height/1526/document-hub/01HAKZPCTH49TS35JVHD6PC5BM.jpg",
        "name": "Пальто",
        "category": "Женское / Верхняя одежда / Пальто",
        "old_fastener": "кнопки",
    },
    {
        "label": "TSUM кардиган кашемир (в экспорте: кнопки)",
        "url": "https://collect-static.tsum.com/sig/c53303c14b0ba7e4162892f8dc760b9c/height/1526/document-hub/01HAHAE4HD64M63P0AJ7HTMT3K.jpg",
        "name": "Кашемировый кардиган",
        "category": "Женское / Трикотаж / Кардиганы",
        "old_fastener": "кнопки",
    },
    {
        "label": "TSUM пуховик (в экспорте: кнопки)",
        "url": "https://collect-static.tsum.com/sig/4855662b19084b86660b2449a21b25e4/height/1526/document-hub/01HAPK2K7J9AJVZYAY61J9FB4N.jpg",
        "name": "Пуховая куртка",
        "category": "Женское / Верхняя одежда / Куртки",
        "old_fastener": "кнопки",
    },
    # Zolla: в названии «пуговицы» — ожидаем пуговицы, не пресс-кнопки
    {
        "label": "Zolla кардиган на пуговицах",
        "url": "https://zolla.com/upload/iplm/images/goods/013436463033_80N0.jpg",
        "name": "Вязаный кардиган на пуговицах с воротником и карманами",
        "category": "Женская / Трикотаж / Кардиганы",
        "old_fastener": "(из фида: пуговицы)",
    },
    {
        "label": "Zolla кардиган меланж + в названии пуговицы",
        "url": "https://zolla.com/upload/iplm/images/goods/013436463033_91N0.jpg",
        "name": "Вязаный кардиган на пуговицах",
        "category": "Женская / Трикотаж / Кардиганы",
        "old_fastener": "(из фида: пуговицы)",
    },
    # В фиде «на кнопках» — может быть реально snaps; смотрим что скажет 9b
    {
        "label": "Zolla кардиган «на кнопках» (фид)",
        "url": "https://zolla.com/upload/iplm/images/goods/222326442023_91N0.jpg",
        "name": "Вязаный кардиган с застёжкой на кнопках и поясом на талии",
        "category": "Женская / Трикотаж / Кардиганы",
        "old_fastener": "(фид: кнопки)",
    },
    # Ёлочка vs гусиная лапка
    {
        "label": "TSUM пальто ёлочка (реф.)",
        "url": "https://collect-static.tsum.com/sig/872f8cfa9b75bd10ebde712c5f8f07a9/height/1526/document-hub/01HA2J4G7QJK3KW2DA8WWNFP9M.jpg",
        "name": "Пальто",
        "category": "Женское / Верхняя одежда / Пальто",
        "old_fastener": "",
    },
    # Завязки
    {
        "label": "TSUM жилет на завязках",
        "url": "https://collect-static.tsum.com/sig/dd10ecbf51ae65459ebf4f7be0bee493/height/1526/document-hub/01HAHAPKF99CE7SJG081V6P3FD.jpg",
        "name": "Жилет",
        "category": "Женское / Верхняя одежда / Жилеты",
        "old_fastener": "на завязках",
    },
    # Тедди / фактура
    {
        "label": "Zolla куртка тедди на пуговицах",
        "url": "https://zolla.com/upload/iplm/images/goods/013335550014_1000.jpg",
        "name": "Куртка-рубашка тедди на пуговицах",
        "category": "Женская / Верхняя одежда / Куртки",
        "old_fastener": "(фид: пуговицы)",
    },
]


def main() -> int:
    only: list[int] | None = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--only="):
        part = sys.argv[1].split("=", 1)[1].strip()
        only = [int(x.strip()) for x in part.split(",") if x.strip().isdigit()]
    gs = pm.get_global_settings()
    model = (os.environ.get("OLLAMA_VISION_MODEL") or "qwen3.5:9b").strip()
    ollama_url = (gs.get("ollama_url") or "http://127.0.0.1:11435").strip()
    dirs = copy.deepcopy([d for d in pm.DEFAULT_DIRECTIONS if d.get("id") == "clothing"])
    for d in dirs:
        d["text_enabled"] = False
    cfg = {
        "model": model,
        "ollama_url": ollama_url,
        "image_max_size": min(int(gs.get("image_max_size") or 1024), 1024),
        "vertical": "Одежда",
        "directions": dirs,
        "dynamic_clothing_attributes": False,
        "extract_inscriptions": False,
        "inscription_mode": "separate_call",
        "clothing_standard_keys_enabled": None,
        "attribute_value_llm_translate": False,
    }
    cases = [CASES[i] for i in only] if only is not None else CASES
    idx_map = only if only is not None else list(range(len(CASES)))
    print("model:", model, "ollama:", ollama_url, "cases:", len(cases))
    glossary = pm.load_attribute_glossary()
    rows = []
    for j, c in enumerate(cases):
        i = idx_map[j]
        url = c["url"]
        print(f"\n--- case #{i + 1} ({j + 1}/{len(cases)}) {c['label']} ---\n{url}")
        offer = {
            "offer_id": f"v9b_{i}",
            "name": c["name"],
            "picture_urls": [url],
            "category": c["category"],
        }
        result = analyze_offer(offer, cfg, timeout=240)
        if result.get("error"):
            print("ERROR:", result["error"])
            rows.append({**c, "error": result["error"]})
            continue
        cloth = (result.get("direction_attributes") or {}).get("clothing") or {}
        raw_f = _pick(cloth, "fastener")
        f_ru = pm.translate_attribute_value(raw_f, glossary) if raw_f else ""
        pp = _pick(cloth, "print_pattern")
        pp_ru = pm.translate_attribute_value(pp, glossary) if pp else ""
        col = _pick(cloth, "collar")
        pok = _pick(cloth, "pockets")
        print(
            json.dumps(
                {
                    "old_export_or_feed": c.get("old_fastener"),
                    "fastener": f_ru or raw_f,
                    "print_pattern": pp_ru or pp,
                    "collar": col,
                    "pockets": pok,
                    "avg_confidence": result.get("avg_confidence"),
                },
                ensure_ascii=False,
            )
        )
        rows.append(
            {
                **c,
                "fastener": f_ru or raw_f,
                "print_pattern": pp_ru or pp,
                "collar": col,
                "pockets": pok,
                "avg_confidence": result.get("avg_confidence"),
            }
        )
    out = Path(__file__).resolve().parent / "verify_9b_regressions_last.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСводка записана: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
