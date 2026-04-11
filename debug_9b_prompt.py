#!/usr/bin/env python3
"""
Отладка: что именно отправляется модели 9b и что она возвращает.
Проверяем промпт и ответ для определения цвета металла.
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ["IMAGE_DESC_PROFILE"] = "1"

import project_manager as pm
from attribute_detector import (
    analyze_offer,
    normalize_ollama_url,
    warmup_ollama_model,
    _build_attributes_prompt,
    compose_task_prompt_blocks,
    _ollama_chat,
)

# Реальный промпт из run_prompt_last.json
REAL_PROMPT = {
    "task_instruction": "Укажи цвет металла украшений на картинках.",
    "task_constraints": "Не указывай цвет драгоценных камней или цвет не металлических изделий, не относящихся к украшению.",
    "task_examples": "Если на картинке изображено серебтряное кольцо с огромным красным изумрудом, то нужно укаать цвет \"золотой\", так как цвет металла будет золотой. Цвета могут быть разные, относящиеся к металлам: золотой, серебряный, бронзовый, возможно чёрный, может ещё какие-то цвета. Выделяй только те цвета, которые явно характеризуют металлы и НЕ могут относиться, например, к одежде. Плохие примеры: РОЗОВЫЙ, СИНИЙ, БЕЛЫЙ, РОЗОВОЕ ЗОЛОТО, БЕЛОЕ ЗОЛОТО. Хорошие примеры: ЗОЛОТОЙ, СЕРЕБРЯНЫЙ, БРОНЗОВЫЙ.",
    "task_target_attribute": "цвет металла",
    "task_target_attributes": ["цвет металла"],
}

FIXTURE = {
    "offer_id": "fixture_silver_21",
    "name": "Эталон: ожидается серебро",
    "picture_urls": ["https://static.insales-cdn.com/images/products/1/4737/2739745409/21.jpg"],
    "category": "jewelry",
}


def debug_prompt_building():
    """Проверить, как строится промпт."""
    print("="*70)
    print("ОТЛАДКА: Построение промпта")
    print("="*70)
    
    config = {
        "model": "qwen3.5:9b",
        "ollama_url": "http://127.0.0.1:11434",
        "image_max_size": 512,
        "vertical": "Ювелирные изделия",
        "task_instruction": REAL_PROMPT["task_instruction"],
        "task_constraints": REAL_PROMPT["task_constraints"],
        "task_examples": REAL_PROMPT["task_examples"],
        "task_target_attribute": REAL_PROMPT["task_target_attribute"],
        "task_target_attributes": REAL_PROMPT["task_target_attributes"],
        "dynamic_clothing_attributes": False,
        "extract_inscriptions": False,
        "directions": [
            {
                "id": "metal",
                "name": "Металл",
                "text_enabled": False,
                "attributes": [{"key": "metal_color", "label": "Цвет металла", "options": []}],
                "custom_prompt": "",  # Пустой - должен использоваться task_instruction
            }
        ],
    }
    
    # Проверяем, как строится промпт
    direction = config["directions"][0]
    vertical = config.get("vertical", "")
    use_project_task = bool(config.get("task_instruction") and vertical != "Одежда")
    
    print(f"\n1. use_project_task = {use_project_task}")
    print(f"   vertical = {vertical!r}")
    print(f"   task_instruction = {config.get('task_instruction')!r}")
    
    custom = (direction.get("custom_prompt") or "").strip() or (config.get("task_instruction") if use_project_task else "")
    print(f"\n2. custom (промпт для направления) = {custom!r}")
    
    # Строим промпт как в _build_attributes_prompt
    if custom and custom.strip():
        ta_list = list(config.get("task_target_attributes") or [])
        if not ta_list and (config.get("task_target_attribute") or "").strip():
            ta_list = [x.strip() for x in (config.get("task_target_attribute") or "").split("\n") if x.strip()]
        
        full_prompt = compose_task_prompt_blocks(
            custom.strip(),
            vertical=vertical,
            direction_name=direction.get("name", ""),
            product_name=FIXTURE.get("name"),
            user_constraints=config.get("task_constraints") or "",
            user_examples=config.get("task_examples") or "",
            target_attribute="",
            target_attributes=ta_list if ta_list else None,
        )
        print(f"\n3. Полный промпт (compose_task_prompt_blocks):")
        print("-" * 70)
        print(full_prompt)
        print("-" * 70)
    else:
        print("\n3. ПРОБЛЕМА: custom пустой, используется упрощённый промпт!")
        keys_str = ", ".join([a.get("key", "") for a in direction.get("attributes", [])])
        simple_prompt = (
            f"Вертикаль: {vertical}. Учти специфику сферы партнёра. "
            f"Analyze clothing. Direction: {direction.get('name', '')}. "
            f"Return JSON with ALL keys: {keys_str}. "
            f"Use options when they fit, or your own value. English. JSON only, no markdown, no code blocks."
        )
        print(simple_prompt)
    
    return config


def debug_model_response(config):
    """Проверить, что возвращает модель."""
    print("\n" + "="*70)
    print("ОТЛАДКА: Ответ модели 9b")
    print("="*70)
    
    # Прогрев
    print("\nПрогрев qwen3.5:9b...")
    try:
        warmup_ollama_model(config, timeout=180)
        print("✅ Модель готова")
    except Exception as e:
        print(f"⚠️  Прогрев не выполнен: {e}")
    
    # Анализ
    print(f"\nАнализ оффера: {FIXTURE['offer_id']} — {FIXTURE['name']}")
    result = analyze_offer(FIXTURE, config, timeout=300)
    
    # Показываем результат
    print("\nРезультат analyze_offer:")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    
    # Показываем direction_attributes
    da = result.get("direction_attributes") or {}
    for did, attrs in da.items():
        print(f"\nНаправление {did}:")
        if isinstance(attrs, dict):
            if attrs.get("error"):
                print(f"  ❌ Ошибка: {attrs['error']}")
            else:
                for key, val in attrs.items():
                    if key != "error":
                        print(f"  {key}: {val}")
    
    # Пытаемся получить raw ответ через перехват _ollama_chat
    print("\n" + "="*70)
    print("Попытка получить raw ответ модели...")
    print("="*70)
    
    # Загружаем картинку
    from attribute_detector import _url_to_base64
    image_b64 = _url_to_base64(
        FIXTURE["picture_urls"][0],
        max_size=config.get("image_max_size", 512),
        timeout=30
    )
    
    if not image_b64:
        print("❌ Не удалось загрузить картинку")
        return
    
    # Строим промпт вручную
    direction = config["directions"][0]
    vertical = config.get("vertical", "")
    use_project_task = bool(config.get("task_instruction") and vertical != "Одежда")
    custom = (direction.get("custom_prompt") or "").strip() or (config.get("task_instruction") if use_project_task else "")
    
    if custom and custom.strip():
        ta_list = list(config.get("task_target_attributes") or [])
        if not ta_list and (config.get("task_target_attribute") or "").strip():
            ta_list = [x.strip() for x in (config.get("task_target_attribute") or "").split("\n") if x.strip()]
        
        prompt = compose_task_prompt_blocks(
            custom.strip(),
            vertical=vertical,
            direction_name=direction.get("name", ""),
            product_name=FIXTURE.get("name"),
            user_constraints=config.get("task_constraints") or "",
            user_examples=config.get("task_examples") or "",
            target_attribute="",
            target_attributes=ta_list if ta_list else None,
        )
    else:
        keys_str = ", ".join([a.get("key", "") for a in direction.get("attributes", [])])
        prompt = (
            f"Вертикаль: {vertical}. Учти специфику сферы партнёра. "
            f"Analyze clothing. Direction: {direction.get('name', '')}. "
            f"Return JSON with ALL keys: {keys_str}. "
            f"Use options when they fit, or your own value. English. JSON only, no markdown, no code blocks."
        )
    
    print(f"\nПромпт, отправляемый модели:")
    print("-" * 70)
    print(prompt)
    print("-" * 70)
    
    # Вызываем модель напрямую с детальной отладкой
    print("\nВызов _ollama_chat напрямую...")
    try:
        import requests
        base = normalize_ollama_url(config["ollama_url"]).rstrip("/")
        url = f"{base}/api/chat"
        
        # Используем тот же формат, что и в _ollama_chat
        messages = []
        msg = {"role": "user", "content": prompt}
        if image_b64:
            msg["images"] = [image_b64]  # Ollama использует отдельное поле "images", не внутри content
        messages.append(msg)
        
        payload = {
            "model": config["model"],
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": 2048,  # Увеличено для 9b с thinking mode
            },
        }
        
        print(f"\nОтправка запроса в Ollama:")
        print(f"  URL: {url}")
        print(f"  Model: {config['model']}")
        print(f"  Image present: {bool(image_b64)}")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Payload size: {len(json.dumps(payload))} bytes")
        
        r = requests.post(url, json=payload, timeout=300)
        if r.status_code != 200:
            print(f"\n❌ ОШИБКА {r.status_code}:")
            print(f"Response text: {r.text[:500]}")
            try:
                error_json = r.json()
                print(f"Response JSON: {json.dumps(error_json, ensure_ascii=False, indent=2)}")
            except:
                pass
            r.raise_for_status()
        response_json = r.json()
        
        print(f"\nСтатус ответа: {r.status_code}")
        print(f"Полный ответ Ollama:")
        print(json.dumps(response_json, ensure_ascii=False, indent=2, default=str))
        
        raw_response = (response_json.get("message") or {}).get("content") or ""
        
        print(f"\nRaw ответ модели (content):")
        print("-" * 70)
        print(repr(raw_response))
        print("-" * 70)
        if raw_response:
            print(f"\nRaw ответ модели (текст):")
            print("-" * 70)
            print(raw_response)
            print("-" * 70)
        else:
            print("\n⚠️  ВНИМАНИЕ: Модель вернула ПУСТОЙ ответ!")
            print("Возможные причины:")
            print("  1. Слишком длинный промпт")
            print("  2. Проблема с обработкой изображения")
            print("  3. Таймаут генерации")
            print("  4. Ошибка в модели")
        
        # Парсим JSON
        from attribute_detector import _extract_json, _extract_and_merge_all_json
        parsed = _extract_and_merge_all_json(raw_response)
        if not parsed:
            parsed = _extract_json(raw_response)
        print(f"\nРаспарсенный JSON:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"❌ Ошибка при вызове модели: {e}")
        import traceback
        traceback.print_exc()


def main():
    config = debug_prompt_building()
    debug_model_response(config)


if __name__ == "__main__":
    main()
