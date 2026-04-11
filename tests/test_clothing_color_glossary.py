"""Глоссарий и перевод значений color / color_shade для одежды (EN → различимый RU)."""

import os

import pytest
import project_manager as pm


def _clothing_attr_options(key: str) -> list[str]:
    for d in pm.DEFAULT_DIRECTIONS:
        if d.get("id") != "clothing":
            continue
        for a in d.get("attributes") or []:
            if (a.get("key") or "").strip() == key:
                return [str(x) for x in (a.get("options") or [])]
    return []


def _has_cyrillic(s: str) -> bool:
    return any("\u0400" <= c <= "\u04ff" for c in (s or ""))


@pytest.mark.parametrize("attr_key", ["color", "color_shade"])
def test_clothing_color_options_all_translate_to_russian(attr_key):
    """Каждая опция из настроек одежды даёт value с кириллицей после translate_attribute_value."""
    g = pm.load_attribute_glossary()
    opts = _clothing_attr_options(attr_key)
    assert opts, f"no options for {attr_key}"
    bad: list[tuple[str, str]] = []
    for o in opts:
        ru = pm.translate_attribute_value(o, g)
        if not _has_cyrillic(ru):
            bad.append((o, ru))
    assert not bad, f"non-Russian translations: {bad[:15]}"


def test_nuance_colors_distinct_russian_not_collapsed():
    """Нюансы не схлопываются в один и тот же русский текст (пары из типичных коллизий)."""
    g = pm.load_attribute_glossary()
    pairs = [
        ("fuchsia", "magenta"),
        ("fuchsia", "pink"),
        ("dusty rose", "blush"),
        ("dusty rose", "hot pink"),
        ("crimson", "raspberry"),
        ("magenta", "raspberry"),
        ("purple", "violet"),
        ("pink", "rose"),
        ("baby pink", "pale pink"),
        ("cerulean", "azure"),
        ("navy", "royal blue"),
        ("teal", "turquoise"),
        ("washed", "acid wash"),
    ]
    for a, b in pairs:
        ra = pm.translate_attribute_value(a, g)
        rb = pm.translate_attribute_value(b, g)
        assert ra != rb, f"collapsed: {a!r} and {b!r} -> {ra!r}"


def test_rose_gold_not_broken_by_rose_key():
    """Длинная фраза «rose gold» не ломается укороченным ключом rose."""
    g = pm.load_attribute_glossary()
    assert pm.translate_attribute_value("rose gold", g) == "розовое золото"


def test_translate_direction_attribute_values_inplace_color_keys():
    """Как в проде: после analyze значения color проходят через translate inplace."""
    g = pm.load_attribute_glossary()
    da = {
        "clothing": {
            "color": {"value": "fuchsia", "confidence": 90},
            "color_shade": {"value": "dusty rose", "confidence": 85},
        }
    }
    pm.translate_direction_attribute_values_inplace(da, g)
    assert da["clothing"]["color"]["value"] == "фуксия"
    assert da["clothing"]["color_shade"]["value"] == "пыльная роза"


@pytest.mark.skipif(os.environ.get("RUN_OLLAMA_COLOR_SMOKE") != "1", reason="set RUN_OLLAMA_COLOR_SMOKE=1 to call Ollama")
def test_ollama_smoke_returns_color_json(tmp_path, monkeypatch):
    """
    Дымовой тест с реальной моделью: один квадрат цвета, проверка что в ответе есть ключи и RU после глоссария.
    """
    from PIL import Image
    import attribute_detector as ad

    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    img_path = tmp_path / "red.png"
    Image.new("RGB", (120, 120), (200, 40, 40)).save(img_path)

    cfg = {
        "model": os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b"),
        "ollama_url": os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
        "image_max_size": 256,
        "vertical": "Одежда",
        "directions": [
            {
                "id": "clothing",
                "name": "Одежда",
                "text_enabled": False,
                "attributes": [
                    a
                    for a in next(d for d in pm.DEFAULT_DIRECTIONS if d["id"] == "clothing")["attributes"]
                    if a.get("key") in ("color", "color_shade")
                ],
                "custom_prompt": "",
            },
        ],
        "dynamic_clothing_attributes": False,
        "clothing_standard_keys_enabled": ["color", "color_shade"],
    }
    offer = {
        "offer_id": "smoke-color",
        "name": "Красная футболка тест",
        "picture_urls": [str(img_path)],
        "category": "Женская / Футболки",
    }
    result = ad.analyze_offer(offer, cfg, timeout=120)
    assert not result.get("error"), result.get("error")
    cloth = (result.get("direction_attributes") or {}).get("clothing") or {}
    assert isinstance(cloth, dict) and not cloth.get("error")
    g = pm.load_attribute_glossary()
    color_val = (cloth.get("color") or {}).get("value") or ""
    shade_val = (cloth.get("color_shade") or {}).get("value") or ""
    # модель может вернуть уже по-русски или по-английски — нормализуем через глоссарий
    color_ru = pm.translate_attribute_value(str(color_val), g)
    shade_ru = pm.translate_attribute_value(str(shade_val), g) if shade_val else ""
    assert _has_cyrillic(color_ru), f"color value raw={color_val!r} translated={color_ru!r}"
    if shade_val:
        assert _has_cyrillic(shade_ru), f"shade raw={shade_val!r} translated={shade_ru!r}"
