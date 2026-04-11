#!/usr/bin/env python3
"""
Тест: сравнение qwen3.5:2b vs qwen3.5:9b на реальном промпте про цвет металла украшений.
Проверяет, справляется ли 2b с vision-задачами и реальными промптами пользователя.
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ["IMAGE_DESC_PROFILE"] = "1"

import project_manager as pm
import feed_cache as fc
from attribute_detector import analyze_offer, normalize_ollama_url, warmup_ollama_model

# Реальный промпт из run_prompt_last.json
REAL_PROMPT = {
    "task_instruction": "Укажи цвет металла украшений на картинках.",
    "task_constraints": "Не указывай цвет драгоценных камней или цвет не металлических изделий, не относящихся к украшению.",
    "task_examples": "Если на картинке изображено серебтряное кольцо с огромным красным изумрудом, то нужно укаать цвет \"золотой\", так как цвет металла будет золотой. Цвета могут быть разные, относящиеся к металлам: золотой, серебряный, бронзовый, возможно чёрный, может ещё какие-то цвета. Выделяй только те цвета, которые явно характеризуют металлы и НЕ могут относиться, например, к одежде. Плохие примеры: РОЗОВЫЙ, СИНИЙ, БЕЛЫЙ, РОЗОВОЕ ЗОЛОТО, БЕЛОЕ ЗОЛОТО. Хорошие примеры: ЗОЛОТОЙ, СЕРЕБРЯНЫЙ, БРОНЗОВЫЙ.",
    "task_target_attribute": "цвет металла",
    "task_target_attributes": ["цвет металла"],
}

# Эталоны из run_metal_color_test.py + дополнительные реальные украшения
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
    # Используем только реальные фикстуры - Unsplash обрывает соединения
]


def get_test_offers() -> list[dict]:
    """Собрать офферы для теста: только реальные фикстуры (без Unsplash - обрывает соединения)."""
    # Используем только проверенные фикстуры из run_metal_color_test.py
    # Они реальные URL украшений, которые работают
    return FIXTURES[:2]  # Только 2 реальных фикстуры для надёжного теста


def run_model_test(model: str, offers: list[dict], ollama_url: str) -> dict:
    """Запустить тест на одной модели."""
    print(f"\n{'='*70}")
    print(f"МОДЕЛЬ: {model}")
    print(f"{'='*70}\n")
    
    config = {
        "model": model,
        "ollama_url": ollama_url,
        "image_max_size": 512,  # Оптимизированный размер
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
                "custom_prompt": "",
            }
        ],
    }
    
    # Прогрев
    print(f"Прогрев {model}...")
    try:
        warmup_ollama_model(config, timeout=180)
        print("✅ Модель готова\n")
    except Exception as e:
        print(f"⚠️  Прогрев не выполнен: {e}\n")
    
    results = []
    total_time = 0
    
    for i, offer in enumerate(offers, 1):
        oid = offer.get("offer_id", "")
        name = (offer.get("name") or "")[:60]
        print(f"[{i}/{len(offers)}] {oid} — {name}")
        
        result = analyze_offer(offer, config, timeout=300)
        prof = result.get("_profile") or {}
        vision_ms = prof.get("vision_calls_sum_ms", 0)
        total_time += vision_ms
        
        if result.get("error"):
            print(f"  ❌ Ошибка: {result['error'][:100]}")
            results.append({"offer_id": oid, "error": result["error"], "time_ms": 0})
            continue
        
        # Отладка: показать raw ответ модели
        da = result.get("direction_attributes") or {}
        for _did, attrs in da.items():
            if isinstance(attrs, dict) and attrs.get("_raw"):
                raw_preview = attrs["_raw"][:300] if len(attrs["_raw"]) > 300 else attrs["_raw"]
                print(f"  [DEBUG] Raw ответ: {raw_preview!r}")
                break
        
        # Извлечь metal_color
        da = result.get("direction_attributes") or {}
        metal_color = None
        confidence = 0
        for _did, attrs in da.items():
            if not isinstance(attrs, dict) or attrs.get("error"):
                continue
            mc = attrs.get("metal_color")
            if isinstance(mc, dict):
                metal_color = mc.get("value", "?")
                confidence = mc.get("confidence", 0)
                break
        
        if metal_color:
            status = "✅" if metal_color.lower() not in ("unknown", "нет металла", "?") else "⚠️"
            print(f"  {status} metal_color: {metal_color!r} (confidence: {confidence}%) | время: {vision_ms:.0f}ms")
            results.append({
                "offer_id": oid,
                "metal_color": metal_color,
                "confidence": confidence,
                "time_ms": round(vision_ms, 1),
                "avg_confidence": result.get("avg_confidence", 0),
            })
        else:
            print(f"  ⚠️  metal_color не найден | время: {vision_ms:.0f}ms")
            results.append({"offer_id": oid, "metal_color": None, "time_ms": round(vision_ms, 1)})
    
    avg_time = total_time / len(offers) if offers else 0
    successful = [r for r in results if r.get("metal_color") and r.get("metal_color") not in ("unknown", "?")]
    success_rate = len(successful) / len(results) * 100 if results else 0
    
    print(f"\n📊 Сводка {model}:")
    print(f"  Успешных ответов: {len(successful)}/{len(results)} ({success_rate:.0f}%)")
    print(f"  Среднее время: {avg_time:.0f}ms ({avg_time/1000:.1f} сек)")
    if successful:
        avg_conf = sum(r.get("confidence", 0) for r in successful) / len(successful)
        print(f"  Средняя уверенность: {avg_conf:.0f}%")
    
    return {
        "model": model,
        "results": results,
        "avg_time_ms": round(avg_time, 1),
        "success_rate": round(success_rate, 1),
        "successful_count": len(successful),
    }


def main():
    print("="*70)
    print("ТЕСТ: qwen3.5:2b vs qwen3.5:9b на реальном промпте про цвет металла")
    print("="*70)
    
    ollama_url = "http://127.0.0.1:11434"
    offers = get_test_offers()
    
    if len(offers) < 2:
        print(f"⚠️  Найдено только {len(offers)} офферов. Используем фикстуры.")
        offers = FIXTURES[:2]  # Минимум 2 для сравнения
    
    print(f"\n📦 Тестируем на {len(offers)} украшениях (реальные фикстуры из run_metal_color_test.py)\n")
    
    # Тест 2b
    result_2b = run_model_test("qwen3.5:2b", offers, ollama_url)
    
    # Тест 9b
    result_9b = run_model_test("qwen3.5:9b", offers, ollama_url)
    
    # Сравнение
    print(f"\n{'='*70}")
    print("СРАВНЕНИЕ:")
    print(f"{'='*70}")
    print(f"qwen3.5:2b: успех {result_2b['successful_count']}/{len(offers)} ({result_2b['success_rate']:.0f}%), время {result_2b['avg_time_ms']:.0f}ms ({result_2b['avg_time_ms']/1000:.1f}с)")
    print(f"qwen3.5:9b: успех {result_9b['successful_count']}/{len(offers)} ({result_9b['success_rate']:.0f}%), время {result_9b['avg_time_ms']:.0f}ms ({result_9b['avg_time_ms']/1000:.1f}с)")
    
    speedup = result_9b['avg_time_ms'] / result_2b['avg_time_ms'] if result_2b['avg_time_ms'] > 0 else 0
    if speedup > 1:
        print(f"\n✅ 2b быстрее в {speedup:.1f}x раз")
    elif speedup < 1 and speedup > 0:
        print(f"\n⚠️  9b быстрее в {1/speedup:.1f}x раз")
    
    quality_diff = result_9b['success_rate'] - result_2b['success_rate']
    if abs(quality_diff) < 5:
        print(f"✅ Качество примерно одинаковое (разница {quality_diff:+.1f}%)")
    elif quality_diff > 5:
        print(f"⚠️  9b лучше по качеству на {quality_diff:.1f}%")
    else:
        print(f"✅ 2b лучше по качеству на {abs(quality_diff):.1f}%")
    
    # Сохранить результаты
    out_file = ROOT / "generated" / "2b_vs_9b_jewelry_test.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps({"2b": result_2b, "9b": result_9b}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n📄 Результаты сохранены: {out_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
