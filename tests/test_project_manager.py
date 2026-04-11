"""Tests for project_manager."""
import json

import pytest
import project_manager as pm


def test_get_global_settings_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "APP_SETTINGS_PATH", tmp_path / "missing.json")
    g = pm.get_global_settings()
    assert g["model"] == "qwen3.5:35b"
    assert "ollama_url" in g
    assert g["image_max_size"] == 1024
    assert g.get("max_parallel_vision") == 0
    assert g.get("batch_offer_workers") == 1
    assert g.get("attribute_value_llm_translate") is False
    assert g.get("attribute_value_translate_model") == ""


def test_save_and_get_global_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "APP_SETTINGS_PATH", tmp_path / "app_settings.json")
    pm.save_global_settings(
        {
            "model": "test:4b",
            "ollama_url": "http://x:11434",
            "attribute_value_llm_translate": True,
            "attribute_value_translate_model": "tiny",
        }
    )
    g = pm.get_global_settings()
    assert g["model"] == "test:4b"
    assert g["ollama_url"] == "http://x:11434"
    assert g["attribute_value_llm_translate"] is True
    assert g["attribute_value_translate_model"] == "tiny"


def test_list_projects_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    assert pm.list_projects() == []


def test_create_and_load_project(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    cfg = pm.create_project("TestProj", feed_path="/some/feed.xml")
    assert cfg["name"] == "TestProj"
    assert "directions" in cfg
    loaded = pm.load_project("TestProj")
    assert loaded["name"] == "TestProj"
    assert loaded.get("feed_path") == "/some/feed.xml"


def test_load_project_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    with pytest.raises(ValueError, match="not found"):
        pm.load_project("NoSuch")


def test_create_project_empty_name(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    with pytest.raises(ValueError, match="empty"):
        pm.create_project("  ")


def test_save_project_merge_directions(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    pm.create_project("D", directions=[{"id": "clothing", "name": "Одежда", "attributes": []}])
    loaded = pm.load_project("D")
    dirs = {d["id"]: d for d in loaded["directions"]}
    assert "clothing" in dirs
    assert "other" in dirs
    ckeys = [a.get("key") for a in dirs["clothing"].get("attributes") or []]
    assert "color" in ckeys and "color_shade" in ckeys


def test_merge_clothing_preserves_saved_attrs_and_inserts_color(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    old_attrs = [
        {"key": "sleeve_length", "label": "Длина рукава", "options": ["short"]},
    ]
    pm.create_project(
        "P",
        directions=[
            {
                "id": "clothing",
                "name": "Одежда",
                "text_enabled": True,
                "attributes": old_attrs,
                "custom_prompt": "",
            },
            {"id": "other", "name": "Другое", "text_enabled": False, "attributes": [], "custom_prompt": ""},
        ],
    )
    loaded = pm.load_project("P")
    cloth = next(d for d in loaded["directions"] if d["id"] == "clothing")
    keys = [a["key"] for a in cloth["attributes"]]
    assert "color" in keys and "color_shade" in keys
    assert keys.index("color") < keys.index("material")
    sl = next(a for a in cloth["attributes"] if a["key"] == "sleeve_length")
    assert sl["options"] == ["short"]


def test_default_clothing_attribute_catalog_includes_color():
    keys = [k for _, k in pm.default_clothing_attribute_catalog()]
    assert "color" in keys
    assert "color_shade" in keys


def test_run_batch_fingerprint_includes_unique_pictures_mode():
    fp = pm.run_batch_fingerprint("P", ["a"], 10, False, False, "phash", "same_prompt")
    assert fp["unique_pictures_mode"] == "phash"
    assert fp["inscription_mode"] == "same_prompt"


def test_results_db_path(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    p = pm.results_db_path("X")
    assert p.name == "results.db"
    assert "X" in str(p)


def test_cache_db_path(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    p = pm.cache_db_path("X")
    assert p.name == "cache.db"


def test_image_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    p = pm.image_cache_dir("X")
    assert p.name == "image_cache"


def test_get_or_create_creates(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    cfg = pm.get_or_create_project("NewOne")
    assert cfg["name"] == "NewOne"


def test_get_or_create_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    pm.create_project("Exist")
    cfg = pm.get_or_create_project("Exist")
    assert cfg["name"] == "Exist"


def test_save_and_load_corrections(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    pm.create_project("C")
    pm.save_correction("C", {"offer_id": "1", "attributes": {"a": "b"}})
    corr = pm.load_corrections("C")
    assert len(corr) == 1
    assert corr[0]["offer_id"] == "1"


def test_get_all_attribute_definitions():
    config = {
        "directions": [
            {"id": "c", "attributes": [{"key": "x", "label": "X"}, {"key": "y", "label": "Y"}]},
        ]
    }
    defs = pm.get_all_attribute_definitions(config)
    assert len(defs) == 2
    keys = [d["key"] for d in defs]
    assert "x" in keys and "y" in keys


def test_save_project_empty_name(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    with pytest.raises(ValueError, match="empty"):
        pm.save_project({"name": ""})


def test_create_project_already_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    pm.create_project("Dup")
    with pytest.raises(ValueError, match="already exists"):
        pm.create_project("Dup")


def test_load_attribute_glossary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "ATTRIBUTE_GLOSSARY_PATH", tmp_path / "no_glossary.json")
    assert pm.load_attribute_glossary() == {}


def test_load_attribute_glossary_exists(tmp_path, monkeypatch):
    p = tmp_path / "glossary.json"
    p.write_text('{"short": "короткий", "Long": "длинный"}', encoding="utf-8")
    monkeypatch.setattr(pm, "ATTRIBUTE_GLOSSARY_PATH", p)
    g = pm.load_attribute_glossary()
    assert g["short"] == "короткий"
    assert g["long"] == "длинный"


def test_translate_attribute_value_hit():
    glossary = {"short": "короткий", "unknown": "неизвестно"}
    assert pm.translate_attribute_value("short", glossary) == "короткий"
    assert pm.translate_attribute_value("SHORT", glossary) == "короткий"
    assert pm.translate_attribute_value("Unknown", glossary) == "неизвестно"


def test_translate_attribute_value_miss():
    glossary = {"short": "короткий"}
    assert pm.translate_attribute_value("long", glossary) == "long"
    assert pm.translate_attribute_value("", glossary) == ""


def test_translate_attribute_value_no_clip_inside_word():
    """«short» из глоссария не должно ломать слово «shorts»."""
    glossary = {"short": "короткий"}
    assert pm.translate_attribute_value("shorts", glossary) == "shorts"


def test_translate_attribute_value_phrase_inside_sentence():
    glossary = {"cotton mesh": "хлопок и сетка", "upper": "верх"}
    assert pm.translate_attribute_value("cotton mesh upper", glossary) == "хлопок и сетка верх"


def test_translate_attribute_value_comma_parts():
    glossary = {"eyelet": "прошва", "ruffle": "оборка"}
    assert pm.translate_attribute_value("eyelet, ruffle", glossary) == "прошва, оборка"


def test_translate_plain_satin_sheen_from_glossary():
    g = pm.load_attribute_glossary()
    assert pm.translate_attribute_value("plain, satin sheen", g) == "однотонный, атлас"


def test_translate_plain_lurex_shimmer_from_glossary():
    g = pm.load_attribute_glossary()
    assert pm.translate_attribute_value("plain, lurex shimmer", g) == "однотонный, люрекс"


def test_translate_sequin_pattern_from_glossary():
    g = pm.load_attribute_glossary()
    assert pm.translate_attribute_value("plain, sequin pattern", g) == "однотонный, пайетки (узор)"


def test_translate_python_skin_from_glossary():
    g = pm.load_attribute_glossary()
    assert pm.translate_attribute_value("python skin", g) == "змеиная кожа"
    assert pm.translate_attribute_value("Python Skin", g) == "змеиная кожа"


def test_finetune_queue_and_correction_offer_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    pm.create_project("QProj", feed_path="/x.xml")
    assert pm.load_finetune_queue_offer_ids("QProj") == []
    added, total = pm.append_finetune_queue_offer_ids("QProj", ["a", "a", "b"])
    assert (added, total) == (2, 2)
    assert pm.load_finetune_queue_offer_ids("QProj") == ["a", "b"]
    pm.clear_finetune_queue_offer_ids("QProj")
    assert pm.load_finetune_queue_offer_ids("QProj") == []
    pm.save_correction(
        "QProj",
        {"offer_id": "o1", "picture_url": "", "corrected_attributes": {"color": "black"}},
    )
    assert pm.correction_offer_ids("QProj") == {"o1"}


def test_translate_direction_attribute_values_inplace():
    glossary = {"short": "короткий", "cotton": "хлопок"}
    da = {"clothing": {"sleeve_length": {"value": "short", "confidence": 90}, "material": {"value": "cotton", "confidence": 80}}}
    pm.translate_direction_attribute_values_inplace(da, glossary)
    assert da["clothing"]["sleeve_length"]["value"] == "короткий"
    assert da["clothing"]["material"]["value"] == "хлопок"


def test_fix_ru_spatial_length_words():
    assert pm.fix_ru_spatial_length_words("sleeve_length", "долгий") == "длинный"
    assert pm.fix_ru_spatial_length_words("length", "Долгая") == "Длинная"
    assert pm.fix_ru_spatial_length_words("material", "долгий") == "долгий"


def test_translate_direction_fixes_dolgij_before_glossary():
    da = {"clothing": {"sleeve_length": {"value": "долгий", "confidence": 90}}}
    pm.translate_direction_attribute_values_inplace(da, {})
    assert da["clothing"]["sleeve_length"]["value"] == "длинный"


def test_sanitize_print_pattern_value_removes_plain_with_pattern():
    assert pm.sanitize_print_pattern_value("однотонный, клетка") == "клетка"
    assert pm.sanitize_print_pattern_value("однотонный, двутонный") == "двутонный"
    assert pm.sanitize_print_pattern_value("однотонный") == "однотонный"


def test_translate_direction_sanitizes_print_pattern_after_glossary():
    g = pm.load_attribute_glossary()
    da = {"clothing": {"print_pattern": {"value": "plain, checked", "confidence": 90}}}
    pm.translate_direction_attribute_values_inplace(da, g)
    v = da["clothing"]["print_pattern"]["value"].lower()
    assert "однотонный" not in v
    assert "клетка" in v


def test_normalize_presence_like_attribute_values():
    da = {
        "clothing": {
            "pockets": {"value": "присутствуют", "confidence": 90},
            "fastener": {"value": "застёжка - отсутствие", "confidence": 80},
        }
    }
    pm.normalize_presence_like_attribute_values_inplace(da)
    assert da["clothing"]["pockets"]["value"] == "да"
    assert da["clothing"]["fastener"]["value"] == "нет"


def test_strip_forbidden_original_name_only_clothing():
    da = {
        "clothing": {"original_name": {"value": "x", "confidence": 90}, "color": {"value": "red", "confidence": 80}},
        "drugs": {"original_name": {"value": "Аспирин", "confidence": 90}},
    }
    pm.strip_forbidden_attribute_keys_inplace(da)
    assert "original_name" not in da["clothing"]
    assert "original_name" in da["drugs"]


def test_translate_attribute_value_uses_load_when_glossary_none(tmp_path, monkeypatch):
    p = tmp_path / "g.json"
    p.write_text('{"yes": "да"}', encoding="utf-8")
    monkeypatch.setattr(pm, "ATTRIBUTE_GLOSSARY_PATH", p)
    assert pm.translate_attribute_value("yes", None) == "да"


def test_fashion_colors_basic_from_glossary():
    """Базовые цвета должны переводиться из glossary."""
    g = pm.load_attribute_glossary()
    cases = [
        ("black", "чёрный"),
        ("white", "белый"),
        ("red", "красный"),
        ("blue", "синий"),
        ("green", "зелёный"),
        ("yellow", "жёлтый"),
        ("orange", "оранжевый"),
        ("purple", "фиолетовый"),
        ("pink", "розовый"),
        ("brown", "коричневый"),
        ("grey", "серый"),
        ("gray", "серый"),
        ("beige", "бежевый"),
        ("navy", "тёмно-синий"),
    ]
    for en, ru in cases:
        assert pm.translate_attribute_value(en, g) == ru, f"Color '{en}' failed"


def test_fashion_colors_nuanced_from_glossary():
    """Нюансовые fashion-цвета должны переводиться из glossary."""
    g = pm.load_attribute_glossary()
    cases = [
        ("burgundy", "бургундский"),
        ("dusty rose", "пыльная роза"),
        ("sage green", "шалфейный"),
        ("mustard", "горчичный"),
        ("teal", "сине-зелёный"),
        ("turquoise", "бирюзовый"),
        ("lavender", "лавандовый"),
        ("lilac", "лиловый"),
        ("mauve", "розовато-лиловый"),
        ("blush", "пудровый"),
        ("nude", "нюд"),
        ("ivory", "слоновая кость"),
        ("ecru", "экрю"),
        ("taupe", "тауп"),
        ("camel", "верблюжий"),
        ("khaki", "хаки"),
        ("olive", "оливковый"),
        ("emerald", "изумрудный"),
        ("coral", "коралловый"),
        ("terracotta", "терракотовый"),
        ("rust", "ржавчина"),
        ("charcoal", "антрацит"),
        ("fuchsia", "фуксия"),
        ("magenta", "пурпурный"),
        ("indigo", "индиго"),
        ("cobalt blue", "кобальтово-синий"),
        ("navy blue", "тёмно-синий"),
        ("royal blue", "королевский синий"),
        ("sky blue", "небесно-голубой"),
        ("midnight blue", "полуночно-синий"),
        ("champagne", "шампань"),
        ("pearl", "жемчужный"),
        ("mocha", "мокко"),
        ("espresso", "эспрессо"),
        ("caramel", "карамельный"),
        ("chocolate", "шоколадный"),
        ("copper", "медный"),
        ("multicolor", "многоцветный"),
        ("color block", "колорблок"),
    ]
    for en, ru in cases:
        assert pm.translate_attribute_value(en, g) == ru, f"Color '{en}' failed"


def test_fashion_colors_compound_comma_list():
    """Список цветов через запятую переводится покомпонентно."""
    g = pm.load_attribute_glossary()
    result = pm.translate_attribute_value("black, white", g)
    assert result == "чёрный, белый"

    result2 = pm.translate_attribute_value("navy, coral, ivory", g)
    assert result2 == "тёмно-синий, коралловый, слоновая кость"


def test_fashion_colors_case_insensitive():
    """Перевод цветов не зависит от регистра."""
    g = pm.load_attribute_glossary()
    assert pm.translate_attribute_value("BURGUNDY", g) == "бургундский"
    assert pm.translate_attribute_value("Dusty Rose", g) == "пыльная роза"
    assert pm.translate_attribute_value("NAVY BLUE", g) == "тёмно-синий"
    assert pm.translate_attribute_value("Mustard Yellow", g) == "горчичный"


def test_fashion_colors_no_clip_in_compound_words():
    """'navy' из глоссария не должен ломать составное слово с другим значением."""
    g = pm.load_attribute_glossary()
    # 'navy' само по себе переводится
    assert pm.translate_attribute_value("navy", g) == "тёмно-синий"
    # но 'navy blue' тоже должно корректно переводиться как фраза
    assert pm.translate_attribute_value("navy blue", g) == "тёмно-синий"


def test_strip_placeholder_attribute_values_inplace():
    """unknown / неизвестно / known — убираем ключ; валидный none (нет застёжки) не трогаем."""
    da = {
        "clothing": {
            "color": {"value": "неизвестно", "confidence": 99},
            "hood": {"value": "known", "confidence": 100},
            "material": {"value": "cotton", "confidence": 80},
            "fastener": {"value": "none", "confidence": 85},
        }
    }
    pm.strip_placeholder_attribute_values_inplace(da)
    assert "color" not in da["clothing"]
    assert "hood" not in da["clothing"]
    assert da["clothing"]["material"]["value"] == "cotton"
    assert da["clothing"]["fastener"]["value"] == "none"


def test_strip_placeholder_comma_list_drops_only_placeholders():
    da = {"clothing": {"color": {"value": "красный, unknown", "confidence": 90}}}
    pm.strip_placeholder_attribute_values_inplace(da)
    assert da["clothing"]["color"]["value"] == "красный"


def test_attribute_value_is_placeholder_noise_russian_prefixes():
    assert pm.attribute_value_is_placeholder_noise("неизвестный")
    assert pm.attribute_value_is_placeholder_noise("неузнаваемый")
    assert pm.attribute_value_is_placeholder_noise("известняк")
    assert pm.attribute_value_is_placeholder_noise("  неизвестно  ")
    assert not pm.attribute_value_is_placeholder_noise("пыльная роза")
    assert not pm.attribute_value_is_placeholder_noise("бежевый")


def test_strip_placeholder_removes_neuznavaemyj_and_izvestnyak():
    da = {
        "clothing": {
            "a": {"value": "неузнаваемый", "confidence": 80},
            "b": {"value": "известняк", "confidence": 80},
        }
    }
    pm.strip_placeholder_attribute_values_inplace(da)
    assert "a" not in da["clothing"] and "b" not in da["clothing"]


def test_parse_model_size_billions():
    assert pm.parse_model_size_billions("qwen3.5:9b") == 9.0
    assert pm.parse_model_size_billions("qwen2.5-vl:7b") == 7.0
    assert pm.parse_model_size_billions("qwen3.5:35b") == 35.0
    assert pm.parse_model_size_billions("") is None


def test_recommended_batch_offer_workers_24gb():
    assert pm.recommended_batch_offer_workers("qwen3.5:4b", vram_gb=24) == 4
    assert pm.recommended_batch_offer_workers("qwen3.5:9b", vram_gb=24) == 2
    assert pm.recommended_batch_offer_workers("qwen3.5:35b", vram_gb=24) == 1
    assert pm.recommended_batch_offer_workers("qwen2.5-vl:7b", vram_gb=24) == 3
    assert pm.recommended_batch_offer_workers("unknown-model", vram_gb=24) == 2


def test_load_project_auto_omit_offer_title_non_clothing_when_key_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    (tmp_path / "Pharm").mkdir()
    raw = {
        "name": "Pharm",
        "vertical": "Аптека",
        "confidence_threshold": 50,
        "feed_path": "",
        "output_dir": "",
        "directions": [
            {
                "id": "drugs",
                "name": "Препараты",
                "text_enabled": False,
                "attributes": [{"key": "original_name", "label": "N", "options": []}],
                "custom_prompt": "",
            }
        ],
        "dynamic_clothing_attributes": True,
        "extract_inscriptions": True,
        "process_unique_pictures_mode": "off",
        "inscription_mode": "separate_call",
        "inscription_model": "",
        "clothing_standard_keys_enabled": None,
    }
    (tmp_path / "Pharm" / "config.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    loaded = pm.load_project("Pharm")
    assert loaded["omit_offer_title_in_prompt"] is True


def test_load_project_explicit_omit_false_kept_for_non_clothing(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path)
    (tmp_path / "Pharm2").mkdir()
    raw = {
        "name": "Pharm2",
        "vertical": "Аптека",
        "omit_offer_title_in_prompt": False,
        "confidence_threshold": 50,
        "feed_path": "",
        "output_dir": "",
        "directions": [
            {
                "id": "drugs",
                "name": "Препараты",
                "text_enabled": False,
                "attributes": [],
                "custom_prompt": "x",
            }
        ],
        "dynamic_clothing_attributes": True,
        "extract_inscriptions": True,
        "process_unique_pictures_mode": "off",
        "inscription_mode": "separate_call",
        "inscription_model": "",
        "clothing_standard_keys_enabled": None,
    }
    (tmp_path / "Pharm2" / "config.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    loaded = pm.load_project("Pharm2")
    assert loaded["omit_offer_title_in_prompt"] is False
