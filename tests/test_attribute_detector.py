"""Tests for attribute_detector."""
import base64
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import attribute_detector as ad


def test_normalize_ollama_url_localhost():
    assert "127.0.0.1" in ad.normalize_ollama_url("http://localhost:11434")
    assert ":11434" in ad.normalize_ollama_url("http://localhost:11434")


def test_normalize_ollama_url_other():
    url = "http://192.168.1.1:11434"
    assert ad.normalize_ollama_url(url) == url


def test_ollama_root_health_timeout_pool_port():
    assert ad.ollama_root_health_timeout_s("http://127.0.0.1:11435") == 12.0
    assert ad.ollama_root_health_timeout_s("http://localhost:11435") == 12.0


def test_ollama_root_health_timeout_direct_ollama():
    assert ad.ollama_root_health_timeout_s("http://127.0.0.1:11434") == 3.0


def test_vision_profile_append_empty_list():
    buf: list = []
    ad._vision_profile_append(buf, {"task": "x", "ms": 1.0})
    assert len(buf) == 1 and buf[0]["task"] == "x"


def test_vision_profile_append_none_noop():
    ad._vision_profile_append(None, {"task": "x"})  # не падает


def test_strip_thinking_fence_before_json():
    raw = '```thinking\nрассуждаю\n```\n{"metal_color": {"value": "золотой", "confidence": 90}}'
    out = ad._parse_vision_json_response(raw, ["metal_color"])
    assert out["metal_color"]["value"] == "золотой"


def test_parse_vision_picks_object_with_expected_key():
    raw = '{"draft": true} nonsense {"metal_color": {"value": "серебряный", "confidence": 88}}'
    out = ad._parse_vision_json_response(raw, ["metal_color"])
    assert out["metal_color"]["value"] == "серебряный"


def test_extract_json_plain():
    out = ad._extract_json('{"a": 1}')
    assert out == {"a": 1}


def test_extract_json_with_markdown():
    out = ad._extract_json('```json\n{"b": 2}\n```')
    assert out == {"b": 2}


def test_extract_json_invalid():
    out = ad._extract_json("no json here")
    assert out == {}


def test_text_prompt_without_product():
    p = ad._text_prompt(None)
    assert "JSON" in p
    assert "text_found" in p


def test_text_prompt_with_product():
    p = ad._text_prompt("Топ вискозный")
    assert "Топ вискозный" in p
    assert "ТОЛЬКО" in p


def test_build_attributes_prompt_focus():
    p = ad._build_attributes_prompt("Одежда", [{"key": "x", "label": "X", "options": ["a"]}], product_name="Топ")
    assert "Топ" in p
    assert "ТОЛЬКО" in p
    assert '"x"' in p


def test_build_attributes_prompt_print_pattern_houndstooth_hint():
    attrs = [{"key": "print_pattern", "label": "Паттерн", "options": ["plain"]}]
    p = ad._build_attributes_prompt("Одежда", attrs)
    assert "гусиная лапка" in p
    assert "houndstooth" in p
    assert "фото" in p
    assert "однотонный" in p and "узор" in p
    assert "меланж" in p.lower() or "melange" in p
    assert "люрекс" in p.lower()
    assert "пайетк" in p.lower()
    assert "sequin pattern" in p or "sequin" in p


def test_build_attributes_prompt_details_decor_hint_with_print_pattern():
    attrs = [
        {"key": "print_pattern", "label": "Паттерн", "options": ["plain"]},
        {"key": "details_decor", "label": "Декор", "options": ["none"]},
    ]
    p = ad._build_attributes_prompt("Одежда", attrs)
    assert "details_decor" in p
    assert "sequins" in p
    assert "по-русски" in p


def test_build_attributes_prompt_color_multi_for_sets():
    attrs = [
        {"key": "color", "label": "Цвет (базовый)", "options": ["blue"]},
        {"key": "print_pattern", "label": "Паттерн", "options": ["plain"]},
    ]
    p = ad._build_attributes_prompt(
        "Одежда", attrs, product_name="Набор носков (5 пар в комплекте)"
    )
    assert "через запятую" in p
    assert "Набор" in p or "набор" in p.lower()


def test_compose_task_prompt_blocks_set_name_multi_color():
    p = ad.compose_task_prompt_blocks(
        "Опиши цвет.",
        product_name="Набор носков (5 пар в комплекте)",
    )
    assert "через запятую" in p
    assert "несколько" in p.lower()


def test_compose_task_prompt_blocks_task_priority_section():
    p = ad.compose_task_prompt_blocks("Только самая крупная надпись на упаковке.", product_name=None)
    assert "ПРИОРИТЕТ ФОРМУЛИРОВКИ" in p
    assert "наивысший приоритет" in p.lower()
    assert "крупная" in p.lower()
    assert "латиниц" in p.lower() or "латин" in p.lower()


def test_verbatim_from_image_mode_by_key_and_task():
    assert ad._verbatim_from_image_mode("", "", "", ["part_number"], None) is True
    assert ad._verbatim_from_image_mode("", "", "", ["color"], None) is False
    assert ad._verbatim_from_image_mode("Сними VIN с таблички", "", "", [], None) is True


def test_default_compose_no_hard_english_ban():
    """Универсальный блок формата: нет жёсткого «только русский / не английский»."""
    p = ad.compose_task_prompt_blocks(
        "Определи цвет металла по фото.",
        target_attribute="metal_color / цвет металла",
        required_json_keys=["metal_color"],
    )
    assert "Не заполняй value на английском" not in p
    assert "как на фото" in p.lower()


def test_compose_run_prompt_last_packaging_not_clothing_value_rules():
    """Сохранённое задание (Original_name + крупная надпись): не толкаем модель в «русский класс / принт в другой запрос»."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "run_prompt_last.json").read_text(encoding="utf-8"))
    p = ad.compose_task_prompt_blocks(
        data["task_instruction"],
        user_constraints=data["task_constraints"],
        user_examples=data["task_examples"],
        target_attributes=data.get("task_target_attributes"),
        required_json_keys=["Original_name"],
    )
    assert "не сюда, а в блок надписей" not in p
    _, fmt = p.split("=== ФОРМАТ ОТВЕТА ===", 1)
    assert "класс** признака" not in fmt
    assert "дословно" in p.lower()
    assert "не подменяй" in p.lower()
    assert "латин" in p.lower()


def test_compose_task_prompt_blocks_no_product_skips_focus_block():
    p = ad.compose_task_prompt_blocks("Прочитай этикетку.", product_name=None)
    assert "ФОКУС НА ТОВАРЕ" not in p
    assert "Нужен только товар из оффера" not in p
    assert "Нас интересует ТОЛЬКО товар" not in p
    assert "коробка с лекарством" in p


def test_build_attributes_prompt_custom_uses_compose_without_feed_title():
    p = ad._build_attributes_prompt(
        "Препараты",
        [{"key": "original_name", "label": "Имя", "options": []}],
        custom_prompt="Извлеки только самую крупную строку с упаковки.",
        product_name=None,
        vertical="Аптека",
    )
    assert "ПРИОРИТЕТ ФОРМУЛИРОВКИ" in p
    assert "ФОКУС НА ТОВАРЕ" not in p
    assert "Нас интересует ТОЛЬКО товар" not in p


@patch.object(ad, "_ollama_chat", return_value='{"k":{"value":"x","confidence":90}}')
def test_detect_full_prompt_appends_image_suffix_when_no_product(mock_chat):
    ad.detect_attributes_for_direction(
        "Zm9v",
        "d",
        "n",
        [{"key": "k", "label": "K", "options": []}],
        "",
        "m",
        "http://127.0.0.1:11434",
        5,
        product_name=None,
        full_prompt_override='Ответь JSON с ключом k. Плейсхолдер: «{product_name}»',
    )
    prompt = mock_chat.call_args[0][0]
    assert "Системное напоминание" in prompt
    assert "фото" in prompt


@patch.object(ad, "_ollama_chat", return_value='{"k":{"value":"x","confidence":90}}')
def test_detect_full_prompt_no_extra_suffix_when_product_passed(mock_chat):
    ad.detect_attributes_for_direction(
        "Zm9v",
        "d",
        "n",
        [{"key": "k", "label": "K", "options": []}],
        "",
        "m",
        "http://127.0.0.1:11434",
        5,
        product_name="Витамин C",
        full_prompt_override="JSON k. {product_name}",
    )
    prompt = mock_chat.call_args[0][0]
    assert "Системное напоминание" not in prompt
    assert "Витамин C" in prompt


def test_resize_image_bytes_small():
    try:
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (100, 100), color="red").save(buf, format="PNG")
        data = buf.getvalue()
    except ImportError:
        pytest.skip("PIL not available")
    out = ad._resize_image_bytes(data, max_size=500)
    assert isinstance(out, bytes)
    assert len(out) > 0


@patch("attribute_detector.requests.post")
def test_ollama_chat_qwen3_sends_think_false(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"message": {"content": "{}"}}
    mock_post.return_value.raise_for_status = MagicMock()
    ad._ollama_chat("x", None, "qwen3.5:9b", "http://127.0.0.1:11434", timeout=1)
    body = mock_post.call_args.kwargs.get("json") or {}
    assert body.get("think") is False


@patch("attribute_detector.requests.post")
def test_ollama_chat_success(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"message": {"content": "ok"}}
    mock_post.return_value.raise_for_status = MagicMock()
    result = ad._ollama_chat("hi", None, "model", "http://127.0.0.1:11434", timeout=1)
    assert result == "ok"


@patch("attribute_detector.requests.post")
def test_ollama_chat_with_image(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"message": {"content": "done"}}
    mock_post.return_value.raise_for_status = MagicMock()
    ad._ollama_chat("describe", "base64data", "m", "http://127.0.0.1:11434", timeout=1)
    body = mock_post.call_args.kwargs.get("json", {})
    assert "messages" in body
    assert body["messages"][0].get("images") == ["base64data"]


def test_ollama_error_message_connection():
    e = Exception("Connection refused")
    msg = ad._ollama_error_message(e)
    assert "Ollama" in msg or "11434" in msg


def test_ollama_error_message_with_response():
    e = Exception("400 Bad Request")
    msg = ad._ollama_error_message(e, '{"error": "invalid image"}')
    assert "invalid" in msg or "400" in msg


@patch("attribute_detector._ollama_chat")
def test_detect_text_success(mock_chat):
    mock_chat.return_value = '{"text_found": true, "texts": ["hello"], "confidence": 90}'
    out = ad.detect_text(None, product_name="Top")
    assert out["text_found"] is True
    assert out["texts"] == ["hello"]
    assert out["confidence"] == 90
    assert out.get("error") is None


@patch("attribute_detector._ollama_chat")
def test_detect_text_exception(mock_chat):
    mock_chat.side_effect = RuntimeError("fail")
    out = ad.detect_text(None)
    assert out["error"] == "fail"
    assert out["text_found"] is False


@patch("attribute_detector._ollama_chat")
def test_detect_attributes_for_direction_success(mock_chat):
    mock_chat.return_value = '{"sleeve_length": {"value": "short", "confidence": 80}}'
    out = ad.detect_attributes_for_direction(
        None, "c", "Одежда", [{"key": "sleeve_length", "label": "Рукав"}], "",
        "model", "http://127.0.0.1:11434", timeout=1
    )
    assert "sleeve_length" in out
    assert out["sleeve_length"]["value"] == "short"
    assert out["sleeve_length"]["confidence"] == 80


def test_normalize_parsed_keys_maps_synonym_by_token_overlap():
    """Синонимы ключей без хардкода по metal: пересечение токенов (color_metal → metal_color)."""
    attrs = [{"key": "metal_color", "label": "Цвет металла"}]
    parsed = {"color_metal": {"value": "silver", "confidence": 70}}
    out = ad._normalize_parsed_attribute_keys(parsed, attrs)
    assert "metal_color" in out
    assert out["metal_color"]["value"] == "silver"


def test_normalize_parsed_keys_russian_label_compact():
    """Склеенная форма подписи атрибута (цветметалла) → канонический key из конфига."""
    attrs = [{"key": "metal_color", "label": "Цвет металла"}]
    parsed = {"цветметалла": {"value": "золотой", "confidence": 80}}
    out = ad._normalize_parsed_attribute_keys(parsed, attrs)
    assert "metal_color" in out


def test_coerce_failed_extraction_unknown():
    v, c = ad._coerce_failed_extraction_value("unknown", 90)
    assert v == "" and c == 0


def test_infer_latin_keys_in_parentheses():
    assert ad.infer_latin_keys_in_parentheses("Определи цвет (metal_color) по фото.") == ["metal_color"]


def test_resolve_attributes_for_prompt_fills_empty_direction():
    attrs = ad.resolve_attributes_for_prompt(
        [],
        ["Цвет металла (metal_color)"],
        "Определи по фото.",
        "",
    )
    assert len(attrs) == 1 and attrs[0]["key"] == "metal_color"


def test_resolve_attributes_russian_line_without_parentheses():
    """Как в run_prompt_last: только «цвет металла» без (metal_color)."""
    attrs = ad.resolve_attributes_for_prompt(
        [],
        ["цвет металла"],
        "Укажи цвет металла.",
        "",
    )
    assert len(attrs) == 1 and attrs[0]["key"] == "metal_color"


def test_canonicalize_russian_attribute_line():
    assert ad.canonicalize_target_attribute_line("цвет металла") == "цвет металла (metal_color)"
    assert ad.canonicalize_target_attribute_line("Цвет металла (metal_color)") == "Цвет металла (metal_color)"


def test_prepare_visual_analysis_plan_jewelry_matches_runtime():
    """Не брать ключи с направления clothing: только metal_color из задания."""
    cfg = {
        "vertical": "Ювелирные изделия",
        "directions": [
            {"id": "clothing", "name": "Одежда", "text_enabled": False, "attributes": [{"key": "sleeve_length", "label": "Рукав", "options": []}], "custom_prompt": ""},
            {"id": "other", "name": "Другое", "text_enabled": False, "attributes": [], "custom_prompt": ""},
        ],
        "task_instruction": "Цвет металла.",
        "task_target_attributes": ["цвет металла"],
        "use_full_prompt_edit": False,
    }
    plan = ad.prepare_visual_analysis_plan(cfg)
    assert "clothing" not in [d.get("id") for d in plan["directions"]]
    keys = []
    for d in plan["directions"]:
        attrs = d.get("attributes") or []
        custom = (d.get("custom_prompt") or "").strip() or (cfg["task_instruction"].strip())
        for a in ad.resolve_attributes_for_prompt(attrs, plan["task_target_list"], custom, ""):
            if a.get("key"):
                keys.append(a["key"])
    assert keys == ["metal_color"]


def test_prepare_visual_analysis_plan_clothing_ignores_stale_jewelry_target_keys():
    """Одежда: «старые» ключи из run_prompt_last (metal_color) не сужают направления."""
    cfg = {
        "vertical": "Одежда",
        "directions": [
            {
                "id": "clothing",
                "name": "Одежда",
                "text_enabled": False,
                "attributes": [
                    {"key": "sleeve_length", "label": "Рукав", "options": []},
                    {"key": "hood", "label": "Капюшон", "options": []},
                ],
                "custom_prompt": "",
            },
        ],
        "task_instruction": "",
        "task_target_attributes": ["Цвет металла (metal_color)"],
        "use_full_prompt_edit": False,
    }
    plan = ad.prepare_visual_analysis_plan(cfg)
    clothing = [d for d in plan["directions"] if d.get("id") == "clothing"]
    assert len(clothing) == 1
    keys = [a["key"] for a in (clothing[0].get("attributes") or [])]
    assert "sleeve_length" in keys and "hood" in keys


def test_compose_includes_required_json_keys():
    p = ad.compose_task_prompt_blocks(
        "Задача.",
        vertical="Ювелирные изделия",
        required_json_keys=["metal_color"],
    )
    assert "metal_color" in p
    assert "ОБЯЗАТЕЛЬНО" in p


def test_hoist_nested_json():
    inner = {"metal_color": {"value": "x", "confidence": 70}}
    out = ad._hoist_nested_attribute_json({"result": inner}, ["metal_color"])
    assert out == inner


@patch("attribute_detector._ollama_chat")
def test_detect_nested_result_wrapper(mock_chat):
    mock_chat.return_value = '{"result": {"metal_color": {"value": "золотой", "confidence": 82}}}'
    out = ad.detect_attributes_for_direction(
        None,
        "other",
        "Другое",
        [{"key": "metal_color", "label": "Цвет металла"}],
        "",
        "m",
        "http://127.0.0.1:11434",
        timeout=1,
    )
    assert out["metal_color"]["value"] == "золотой"
    assert out["metal_color"]["confidence"] == 82


@patch("attribute_detector._url_to_base64")
@patch("attribute_detector._ollama_chat")
def test_analyze_offer_jewelry_uses_task_target_when_dir_attrs_empty(mock_chat, mock_b64):
    """Вертикаль не одежда: clothing отфильтрован, other с [] — ключи берём из task_target_attributes."""
    mock_b64.return_value = "img"
    mock_chat.return_value = '{"metal_color": {"value": "золотой", "confidence": 88}}'
    offer = {"offer_id": "1", "name": "Кольцо", "picture_urls": ["https://example.com/x.jpg"]}
    config = {
        "model": "m",
        "ollama_url": "http://127.0.0.1:11434",
        "vertical": "Ювелирные изделия",
        "directions": [
            {"id": "other", "name": "Другое", "text_enabled": False, "attributes": [], "custom_prompt": ""},
        ],
        "task_instruction": "Определи цвет металла по фото.",
        "dynamic_clothing_attributes": True,
        "task_target_attributes": ["Цвет металла (metal_color)"],
        "extract_inscriptions": False,
    }
    result = ad.analyze_offer(offer, config, timeout=1)
    assert result["direction_attributes"]["other"]["metal_color"]["value"] == "золотой"
    assert result.get("avg_confidence", 0) >= 80


@patch("attribute_detector._ollama_chat")
def test_detect_attributes_unknown_not_filled_from_product_name(mock_chat):
    """Название товара не подставляется вместо unknown (нет галлюцинации «золотой» из заголовка)."""
    mock_chat.return_value = '{"metal_color": {"value": "unknown", "confidence": 0}}'
    out = ad.detect_attributes_for_direction(
        None,
        "j",
        "Ювелирка",
        [{"key": "metal_color", "label": "Цвет металла"}],
        "",
        "model",
        "http://127.0.0.1:11434",
        timeout=1,
        product_name="Золотые серьги",
    )
    assert out["metal_color"]["value"] == ""
    assert out["metal_color"]["confidence"] == 0


@patch("attribute_detector._ollama_chat")
def test_analyze_offer_mock(mock_chat):
    mock_chat.return_value = '{"text_found": false, "texts": [], "confidence": 0}'
    offer = {"offer_id": "1", "name": "Top", "picture_urls": []}
    config = {"model": "m", "ollama_url": "http://127.0.0.1:11434", "directions": []}
    result = ad.analyze_offer(offer, config, timeout=1)
    assert result["offer_id"] == "1"
    assert "text_detection" in result
    assert "direction_attributes" in result
    assert "avg_confidence" in result


def test_warmup_ollama_model_no_raise():
    ad.warmup_ollama_model({"model": "x", "ollama_url": "http://127.0.0.1:99999"}, timeout=1)


def test_build_attributes_prompt_custom():
    p = ad._build_attributes_prompt("D", [], custom_prompt="Describe the item.")
    assert "Describe" in p
    assert "JSON" in p or "ФОРМАТ" in p
    assert "ИСТОЧНИК ИСТИНЫ" in p


def test_compose_task_prompt_blocks_user_sections():
    p = ad.compose_task_prompt_blocks(
        "Найти цвет металла.",
        vertical="Ювелирные изделия",
        direction_name="Металл",
        user_constraints="Не камни.",
        user_examples="Если фото и название расходятся — фото.",
        target_attribute="metal_color / цвет металла",
    )
    assert "ЗАДАЧА" in p
    assert "ИЗВЛЕКАЕМЫЕ АТРИБУТЫ" in p
    assert "metal_color" in p
    assert "ОГРАНИЧЕНИЯ (от пользователя)" in p
    assert "ПРИМЕРЫ" in p
    assert "Не камни" in p


def test_detect_attributes_empty_attributes():
    out = ad.detect_attributes_for_direction(
        None, "c", "O", [], "", "m", "http://127.0.0.1:11434", timeout=1
    )
    assert out == {"error": None}


@patch("attribute_detector.requests.get")
def test_ensure_image_cached_miss_then_save(mock_get, tmp_path):
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"\xff\xd8\xff"
    mock_get.return_value.raise_for_status = MagicMock()
    out = ad.ensure_image_cached("https://example.com/pic.jpg", tmp_path, max_size=2048, timeout=1)
    assert out is not None
    assert Path(out).exists()


def test_ensure_image_cached_empty_url():
    assert ad.ensure_image_cached("", Path("/tmp")) is None


@patch("attribute_detector.requests.get")
def test_url_to_base64_success(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"\xff\xd8\xff"
    mock_get.return_value.raise_for_status = MagicMock()
    out = ad._url_to_base64("https://example.com/x.jpg", timeout=1)
    assert out is not None
    assert isinstance(out, str)
    assert len(out) > 0


@patch("attribute_detector.requests.get")
def test_url_to_base64_fail(mock_get):
    mock_get.side_effect = Exception("network error")
    out = ad._url_to_base64("https://example.com/x.jpg", timeout=1)
    assert out is None


def test_image_cache_path():
    p = ad._image_cache_path(Path("/cache"), "https://example.com/a.jpg")
    assert p is not None
    assert "jpg" in str(p)


def test_path_to_base64(tmp_path):
    (tmp_path / "x.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    out = ad._path_to_base64(tmp_path / "x.jpg", max_size=1000)
    assert out is not None
    assert isinstance(out, str)


@patch("attribute_detector._time.sleep")
@patch("attribute_detector.requests.post")
def test_ollama_chat_retry_then_success(mock_post, mock_sleep):
    import requests as req
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"message": {"content": "ok"}}
    mock_post.side_effect = [
        req.RequestException("Connection refused"),
        req.RequestException("timeout"),
        resp,
    ]
    result = ad._ollama_chat("hi", None, "m", "http://127.0.0.1:11434", timeout=1)
    assert result == "ok"
    assert mock_post.call_count == 3


@patch("attribute_detector._ollama_chat")
def test_analyze_offer_direction_raises(mock_chat):
    mock_chat.side_effect = RuntimeError("model error")
    offer = {"offer_id": "1", "name": "X", "picture_urls": []}
    config = {
        "model": "m",
        "ollama_url": "http://127.0.0.1:11434",
        "directions": [{"id": "c", "name": "O", "attributes": [{"key": "x", "label": "X"}]}],
    }
    result = ad.analyze_offer(offer, config, timeout=1)
    assert result["direction_attributes"].get("c", {}).get("error") == "model error"


@patch("attribute_detector._ollama_chat")
def test_analyze_offer_includes_profile_when_env(mock_chat, monkeypatch):
    monkeypatch.setenv("IMAGE_DESC_PROFILE", "1")

    def _ollama_side_effect(*args, **kwargs):
        calls = kwargs.get("vision_profile_calls")
        if calls is not None:
            calls.append(
                {
                    "task": kwargs.get("vision_profile_task", "ollama"),
                    "backend": "ollama",
                    "ms": 1.0,
                }
            )
        return '{"sleeve_length": {"value": "short", "confidence": 70}}'

    mock_chat.side_effect = _ollama_side_effect
    offer = {"offer_id": "p1", "name": "Top", "picture_urls": []}
    config = {
        "model": "m",
        "ollama_url": "http://127.0.0.1:11434",
        "dynamic_clothing_attributes": False,
        "directions": [
            {
                "id": "clothing",
                "name": "Одежда",
                "text_enabled": False,
                "attributes": [{"key": "sleeve_length", "label": "Рукав", "options": ["short"]}],
            },
        ],
    }
    result = ad.analyze_offer(offer, config, timeout=1)
    prof = result.get("_profile")
    assert prof is not None
    assert "total_wall_ms" in prof
    assert "image_prep_ms" in prof
    assert "vision_calls" in prof
    assert prof["max_parallel_vision"] >= 1
    assert any(c.get("task", "").startswith("direction:") for c in prof["vision_calls"])


@patch("attribute_detector._ollama_chat")
def test_analyze_offer_max_parallel_vision(mock_chat):
    """При max_parallel_vision=1 пул из одного потока — оба направления всё равно отрабатывают (2 вызова)."""
    mock_chat.side_effect = [
        '{"x": {"value": "a", "confidence": 80}}',
        '{"y": {"value": "b", "confidence": 80}}',
    ]
    offer = {"offer_id": "p1", "name": "Top", "picture_urls": []}
    config = {
        "model": "m",
        "ollama_url": "http://127.0.0.1:11434",
        "max_parallel_vision": 1,
        "directions": [
            {"id": "da", "name": "A", "text_enabled": False, "attributes": [{"key": "x", "label": "X"}]},
            {"id": "db", "name": "B", "text_enabled": False, "attributes": [{"key": "y", "label": "Y"}]},
        ],
    }
    ad.analyze_offer(offer, config, timeout=1)
    assert mock_chat.call_count == 2


@patch("attribute_detector._ollama_chat")
def test_analyze_offer_with_directions(mock_chat):
    def ret(prompt, *args, **kwargs):
        if "text_found" in (prompt or ""):
            return '{"text_found": false, "texts": [], "confidence": 0}'
        return '{"sleeve_length": {"value": "short", "confidence": 70}}'
    mock_chat.side_effect = ret
    offer = {"offer_id": "1", "name": "Top", "picture_urls": []}
    config = {
        "model": "m",
        "ollama_url": "http://127.0.0.1:11434",
        "directions": [
            {"id": "clothing", "name": "Одежда", "text_enabled": True, "attributes": [{"key": "sleeve_length", "label": "Рукав", "options": ["short"]}]},
        ],
    }
    result = ad.analyze_offer(offer, config, timeout=1)
    assert result["offer_id"] == "1"
    assert "clothing" in result["direction_attributes"]
    assert result["direction_attributes"]["clothing"].get("sleeve_length", {}).get("value") == "короткий"


def test_attribute_value_needs_llm_translate():
    assert ad.attribute_value_needs_llm_translate("mesh")
    assert ad.attribute_value_needs_llm_translate("хлопок mesh")
    assert not ad.attribute_value_needs_llm_translate("хлопок")
    assert not ad.attribute_value_needs_llm_translate("ab")


def test_extract_json_array_from_text():
    raw = '```json\n["один", "два"]\n```'
    assert ad._extract_json_array_from_text(raw) == ["один", "два"]
    assert ad._extract_json_array_from_text('["x"]') == ["x"]
    assert ad._extract_json_array_from_text("") is None


def test_apply_llm_translate_remaining_latin_inplace_mock():
    dr = {
        "clothing": {
            "a": {"value": "plain Russian", "confidence": 90},
            "b": {"value": "acid wash", "confidence": 80},
        }
    }
    with patch.object(
        ad,
        "_batch_translate_attribute_values_llm",
        return_value=["plain Russian", "кислотная варка"],
    ):
        ad.apply_llm_translate_remaining_latin_inplace(
            dr,
            enabled=True,
            translate_model="m",
            vision_model="m",
            ollama_url="http://localhost:11434",
            timeout=60,
            vision_profile_calls=None,
        )
    assert dr["clothing"]["a"]["value"] == "plain Russian"
    assert dr["clothing"]["b"]["value"] == "кислотная варка"


def test_inscription_mode_same_prompt_legacy_russian_label():
    assert ad.inscription_mode_is_same_prompt({"inscription_mode": "same_prompt"})
    assert ad.inscription_mode_is_same_prompt(
        {"inscription_mode": "Та же модель, тот же JSON, что и атрибуты (один запрос на направление)"}
    )
    assert not ad.inscription_mode_is_same_prompt({"inscription_mode": "separate_call"})


def test_inapplicable_clothing_attribute_keys_jeans_and_socks():
    j = ad.inapplicable_clothing_attribute_keys("Женская / Джинсы")
    assert "sleeve_length" in j and "length" in j and "hood" in j
    n = ad.inapplicable_clothing_attribute_keys("Женская / Носки")
    assert "sleeve_length" in n and "length" in n


def test_inapplicable_clothing_attribute_keys_dress_keeps_sleeve():
    d = ad.inapplicable_clothing_attribute_keys("Женская / Платья")
    assert "sleeve_length" not in d and "length" not in d


def test_strip_inapplicable_clothing_attributes():
    dr = {
        "sleeve_length": {"value": "короткий", "confidence": 90},
        "material": {"value": "хлопок", "confidence": 85},
    }
    out = ad.strip_inapplicable_clothing_attributes("Женская / Джинсы", "Одежда", "clothing", dr)
    assert "sleeve_length" not in out
    assert "material" in out


def test_strip_skips_non_clothing_direction():
    dr = {"x": {"value": "1", "confidence": 50}}
    assert ad.strip_inapplicable_clothing_attributes("Женская / Джинсы", "Одежда", "other", dr) is dr


def test_filter_clothing_standard_attributes_for_extraction():
    attrs = [
        {"key": "hood", "label": "Капюшон"},
        {"key": "print_pattern", "label": "Принт"},
        {"key": "colour", "label": "Цвет"},
    ]
    assert len(ad.filter_clothing_standard_attributes_for_extraction(attrs, {"clothing_standard_keys_enabled": None})) == 3

    out = ad.filter_clothing_standard_attributes_for_extraction(
        attrs, {"clothing_standard_keys_enabled": ["print_pattern"]}
    )
    assert [a["key"] for a in out] == ["print_pattern", "colour"]

    out2 = ad.filter_clothing_standard_attributes_for_extraction(
        attrs, {"clothing_standard_keys_enabled": []}
    )
    assert [a["key"] for a in out2] == ["colour"]
