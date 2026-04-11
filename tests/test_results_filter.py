"""Фильтр результатов по надписям и вспомогательные функции app.py."""

from app import (
    RESULTS_SCHEMA,
    _RESULT_MODEL_UNKNOWN_LABEL,
    _collect_attr_keys_from_results,
    _confidence_for_attribute,
    _filter_results_list,
    _merge_result_filter_attr_choices,
    _normalize_attr_value_filter_pick,
    _result_matches_attr_value_needle,
    _result_matches_model_filter,
    _results_model_choices,
    _results_tab_filtered_list,
    _save_result,
    _selection_covers_all_on_page,
    _text_detection_has_inscription,
    _text_detection_joined_for_export,
)


def test_text_detection_has_inscription_empty():
    assert not _text_detection_has_inscription({})
    assert not _text_detection_has_inscription({"confidence": 95, "text_found": False, "texts": []})
    assert not _text_detection_has_inscription({"error": "x"})


def test_text_detection_has_inscription_positive():
    assert _text_detection_has_inscription({"text_found": True, "texts": []})
    assert _text_detection_has_inscription({"text_found": False, "texts": ["  hi  "]})
    assert _text_detection_has_inscription({"text": "LOGO"})


def test_confidence_text_only_when_inscription():
    r = {"text_detection": {"confidence": 95, "text_found": False, "texts": []}}
    assert _confidence_for_attribute(r, "__text__") is None

    r2 = {"text_detection": {"confidence": 88, "text_found": True, "texts": []}}
    assert _confidence_for_attribute(r2, "__text__") == 88

    r3 = {"text_detection": {"text_found": False, "texts": ["sale"]}}
    assert _confidence_for_attribute(r3, "__text__") == 0


def test_text_detection_joined_for_export():
    assert _text_detection_joined_for_export(None) is None
    assert _text_detection_joined_for_export({"error": "x"}) is None
    assert (
        _text_detection_joined_for_export({"texts": ["M", " M "], "confidence": 75})
        == "M, M"
    )
    assert _text_detection_joined_for_export({"text": "  I Love Me  "}) == "I Love Me"


def test_filter_results_list_text_strict():
    rows = [
        {"offer_id": "1", "avg_confidence": 90, "text_detection": {"confidence": 90, "texts": ["A"]}},
        {"offer_id": "2", "avg_confidence": 90, "text_detection": {"confidence": 90, "texts": []}},
    ]
    out = _filter_results_list(rows, "__text__", 0, 100)
    assert [r["offer_id"] for r in out] == ["1"]

    out2 = _filter_results_list(rows, "__text__", 85, 95)
    assert [r["offer_id"] for r in out2] == ["1"]


def test_normalize_attr_value_filter_pick():
    assert _normalize_attr_value_filter_pick(None) == ""
    assert _normalize_attr_value_filter_pick("") == ""
    assert _normalize_attr_value_filter_pick("  — любое значение —  ") == ""
    assert _normalize_attr_value_filter_pick("gold") == "gold"


def test_result_matches_attr_value_needle_substring():
    g = {"yellow gold": "жёлтое золото"}
    r = {
        "direction_attributes": {
            "jewelry": {"metal_color": {"value": "Yellow Gold", "confidence": 90}},
        },
    }
    assert _result_matches_attr_value_needle(r, "metal_color", "gold", g)
    assert _result_matches_attr_value_needle(r, "metal_color", "жёлто", g)
    assert not _result_matches_attr_value_needle(r, "metal_color", "silver", g)


def test_result_matches_attr_value_needle_text():
    r = {"text_detection": {"texts": ["Brand LOGO Here"]}}
    assert _result_matches_attr_value_needle(r, "__text__", "logo", {})
    assert not _result_matches_attr_value_needle(r, "__text__", "missing", {})


def test_collect_attr_keys_from_results():
    rows = [
        {
            "direction_attributes": {
                "jewelry": {"metal_color": {"value": "gold", "confidence": 95}},
            },
        },
        {
            "direction_attributes": {
                "jewelry": {"metal_color": {"value": "silver", "confidence": 90}},
            },
        },
    ]
    keys = _collect_attr_keys_from_results(rows)
    assert keys == ["metal_color"]


def test_selection_covers_all_on_page():
    page = [{"offer_id": "a"}, {"offer_id": "b"}]
    assert not _selection_covers_all_on_page(["a"], page)
    assert _selection_covers_all_on_page(["a", "b"], page)
    assert not _selection_covers_all_on_page([], page)
    assert not _selection_covers_all_on_page(["a"], [])


def test_result_matches_model_filter():
    assert _result_matches_model_filter({"model": "m1"}, "Все")
    assert _result_matches_model_filter({"model": "m1"}, "")
    assert _result_matches_model_filter({"model": "m1"}, "m1")
    assert not _result_matches_model_filter({"model": "m1"}, "m2")
    assert _result_matches_model_filter({"model": ""}, _RESULT_MODEL_UNKNOWN_LABEL)
    assert _result_matches_model_filter({"model_name": ""}, _RESULT_MODEL_UNKNOWN_LABEL)
    assert not _result_matches_model_filter({"model": "x"}, _RESULT_MODEL_UNKNOWN_LABEL)


def test_results_model_choices(tmp_path):
    db = tmp_path / "results.sqlite"
    import sqlite3

    con = sqlite3.connect(db)
    con.executescript(RESULTS_SCHEMA)
    con.commit()
    con.close()
    _save_result(
        db,
        {
            "offer_id": "a",
            "name": "",
            "category": "Cat1",
            "picture_url": "",
            "avg_confidence": 50,
            "text_detection": {},
            "direction_attributes": {},
            "model": "llama:x",
        },
    )
    _save_result(
        db,
        {
            "offer_id": "b",
            "name": "",
            "category": "Cat1",
            "picture_url": "",
            "avg_confidence": 50,
            "text_detection": {},
            "direction_attributes": {},
            "model": "",
        },
    )
    _save_result(
        db,
        {
            "offer_id": "c",
            "name": "",
            "category": "Cat2",
            "picture_url": "",
            "avg_confidence": 50,
            "text_detection": {},
            "direction_attributes": {},
            "model": "qwen:y",
        },
    )
    all_m = _results_model_choices(db, "Все")
    assert all_m[0] == "Все"
    assert _RESULT_MODEL_UNKNOWN_LABEL in all_m
    assert "llama:x" in all_m and "qwen:y" in all_m
    cat1 = _results_model_choices(db, "Cat1")
    assert "llama:x" in cat1 and "qwen:y" not in cat1
    assert _RESULT_MODEL_UNKNOWN_LABEL in cat1


def test_results_tab_filtered_list_by_model(tmp_path):
    db = tmp_path / "results.sqlite"
    import sqlite3

    con = sqlite3.connect(db)
    con.executescript(RESULTS_SCHEMA)
    con.commit()
    con.close()
    proj = {"directions": [{"attributes": [{"key": "k", "label": "K"}]}]}
    _save_result(
        db,
        {
            "offer_id": "1",
            "category": "Все",
            "picture_url": "",
            "avg_confidence": 80,
            "text_detection": {},
            "direction_attributes": {"clothing": {"k": {"value": "v", "confidence": 80}}},
            "model": "alpha",
        },
    )
    _save_result(
        db,
        {
            "offer_id": "2",
            "category": "Все",
            "picture_url": "",
            "avg_confidence": 80,
            "text_detection": {},
            "direction_attributes": {"clothing": {"k": {"value": "v", "confidence": 80}}},
            "model": "beta",
        },
    )
    rows, *_rest = _results_tab_filtered_list(
        db, proj, 0, 100, "Все", False, None, None, "alpha"
    )
    assert [r["offer_id"] for r in rows] == ["1"]


def test_merge_result_filter_attr_choices_adds_db_only_key():
    proj = {
        "directions": [
            {
                "attributes": [
                    {"key": "hood", "label": "Капюшон"},
                ],
            },
        ],
    }
    v, lab = _merge_result_filter_attr_choices(proj, ["metal_color", "hood"])
    assert "hood" in v
    assert "metal_color" in v
    assert "__text__" in v
    assert v.index("metal_color") < v.index("__text__")
    assert any("metal_color" in x for x in lab)
