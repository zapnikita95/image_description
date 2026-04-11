"""Правка результата распространяется на дубликаты по нормализованному URL картинки."""

import pytest

import project_manager as pm


@pytest.fixture()
def tmp_results_db(tmp_path, monkeypatch):
    import app

    db = tmp_path / "results.db"
    app._init_results_db(db)

    def row(oid, url, pat="plain"):
        return {
            "offer_id": oid,
            "name": f"n{oid}",
            "category": "c",
            "picture_url": url,
            "avg_confidence": 90,
            "text_detection": {"texts": [], "text_found": False, "confidence": 0},
            "direction_attributes": {
                "clothing": {"print_pattern": {"value": pat, "confidence": 90}},
            },
            "error": "",
            "model": "m",
        }

    app._save_result(db, row("1", "https://cdn.example.com/a.jpg?w=100", "plain"))
    app._save_result(db, row("2", "https://cdn.example.com/a.jpg?w=200", "plain"))
    app._save_result(db, row("3", "https://other.com/b.jpg", "plain"))
    return db


def test_offer_ids_sharing_normalized_picture(tmp_results_db):
    from app import _offer_ids_sharing_normalized_picture

    ids = _offer_ids_sharing_normalized_picture(tmp_results_db, "https://cdn.example.com/a.jpg?x=1")
    assert set(ids) == {"1", "2"}
    assert _offer_ids_sharing_normalized_picture(tmp_results_db, "https://other.com/b.jpg") == ["3"]


def test_apply_correction_updates_attrs(tmp_results_db):
    from app import _apply_correction_to_stored_result, _load_result_by_offer_id

    base = _load_result_by_offer_id(tmp_results_db, "1")
    assert base
    cfg = {"directions": pm.DEFAULT_DIRECTIONS}
    g = pm.load_attribute_glossary()
    out = _apply_correction_to_stored_result(
        base,
        {"print_pattern": "houndstooth"},
        None,
        cfg,
        g,
    )
    assert out["offer_id"] == "1"
    v = out["direction_attributes"]["clothing"]["print_pattern"]["value"]
    assert "гусин" in v.lower() or "houndstooth" in v.lower()


def test_save_result_propagates_via_helpers(tmp_results_db):
    import app

    cfg = {"directions": pm.DEFAULT_DIRECTIONS}
    g = pm.load_attribute_glossary()
    dup_ids = app._offer_ids_sharing_normalized_picture(tmp_results_db, "https://cdn.example.com/a.jpg")
    assert len(dup_ids) == 2
    for dup_oid in dup_ids:
        base = app._load_result_by_offer_id(tmp_results_db, dup_oid)
        updated = app._apply_correction_to_stored_result(
            base,
            {"print_pattern": "houndstooth"},
            "HELLO",
            cfg,
            g,
        )
        app._save_result(tmp_results_db, updated)

    for oid in ("1", "2"):
        r = app._load_result_by_offer_id(tmp_results_db, oid)
        assert r["text_detection"].get("texts") == ["HELLO"]
        v = r["direction_attributes"]["clothing"]["print_pattern"]["value"]
        assert "гусин" in v.lower() or "houndstooth" in v.lower()
    r3 = app._load_result_by_offer_id(tmp_results_db, "3")
    assert r3["text_detection"].get("texts") == []
