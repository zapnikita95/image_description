#!/usr/bin/env python3
"""
Проверка сборки промпта (короткая задача + шаблон + ограничения + примеры) на qwen3.5:9b.
Эталонные URL + несколько «новых» картинок с picsum/placeholder при отсутствии фида.
Результат: generated/block_prompt_test/last_run.txt
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import project_manager as pm
from attribute_detector import analyze_offer, normalize_ollama_url

PROJECT_NAME = "TestBlockPromptJewelry"
MODEL = "qwen3.5:9b"

# Короткая формулировка пользователя — как в интерфейсе
TASK = "Определить цвет металла изделия."

USER_CONSTRAINTS = (
    "Не путать цвет металла с цветом камней и вставок. Смотреть на оправу, швензу, цепь."
)

USER_EXAMPLES = (
    "Пример: в названии «золотое кольцо с бриллиантом» цвет металла берём с картинки. "
    "Если на фото оправа серебряная, а в названии золото — в ответе цвет металла: серебро."
)

# Два проверенных URL; другие id с того же хоста часто отдают 403 без сессии.
FIXTURES = [
    ("silver_21", "Эталон серебро", "https://static.insales-cdn.com/images/products/1/4737/2739745409/21.jpg"),
    ("gold_73", "Эталон золото", "https://static.insales-cdn.com/images/products/1/4089/617730041/73_1.jpg"),
]


def main():
    print("=== Тест: блоки промпта + короткая задача ===\n")
    try:
        pm.create_project(
            PROJECT_NAME,
            feed_path="",
            vertical="Ювелирные изделия",
            task_instruction=TASK,
            task_constraints=USER_CONSTRAINTS,
            task_examples=USER_EXAMPLES,
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
        print("Создан проект", PROJECT_NAME)
    except ValueError:
        cfg = pm.load_project(PROJECT_NAME)
        cfg["vertical"] = "Ювелирные изделия"
        cfg["task_instruction"] = TASK
        cfg["task_constraints"] = USER_CONSTRAINTS
        cfg["task_examples"] = USER_EXAMPLES
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
        print("Обновлён проект", PROJECT_NAME)

    g = pm.get_global_settings()
    cfg = pm.load_project(PROJECT_NAME)
    cfg["model"] = g.get("model") or MODEL
    cfg["ollama_url"] = g.get("ollama_url") or "http://localhost:11434"
    cfg["image_max_size"] = g.get("image_max_size", 1024)
    cfg["image_cache_dir"] = str(pm.image_cache_dir(PROJECT_NAME))

    url = normalize_ollama_url(cfg["ollama_url"] or "")
    try:
        import requests
        requests.get(url.rstrip("/") + "/", timeout=3).raise_for_status()
    except Exception as e:
        print("Ollama недоступна:", e)
        return 1

    lines = []
    for oid, title, pic in FIXTURES:
        offer = {
            "offer_id": oid,
            "name": title,
            "picture_urls": [pic],
            "category": "jewelry",
        }
        print(f"{oid} {title}...", flush=True)
        r = analyze_offer(offer, cfg, timeout=120)
        if r.get("error"):
            print("  err:", r["error"][:120])
            lines.append(f"{oid}: ERROR")
            continue
        for _d, attrs in (r.get("direction_attributes") or {}).items():
            if not isinstance(attrs, dict) or attrs.get("error"):
                continue
            m = attrs.get("metal_color")
            if isinstance(m, dict):
                print(f"  metal_color={m.get('value')} ({m.get('confidence')}%)")
                lines.append(f"{oid}: {m.get('value')} conf={m.get('confidence')}%")
        print(f"  avg={r.get('avg_confidence')}%")
        print()

    out = ROOT / "generated" / "block_prompt_test" / "last_run.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Записано:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
