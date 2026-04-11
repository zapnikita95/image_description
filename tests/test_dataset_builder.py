"""Tests for fine_tune.dataset_builder."""
import json
from pathlib import Path

import pytest


def test_build_from_external_missing_file(tmp_path):
    from fine_tune.dataset_builder import build_from_external
    result = build_from_external(tmp_path / "missing.jsonl", tmp_path)
    assert result.get("error")
    assert result.get("valid_examples") == 0


def test_build_from_external_valid_jsonl(tmp_path):
    from fine_tune.dataset_builder import build_from_external, SYSTEM_PROMPT_EN
    jsonl = tmp_path / "ext.jsonl"
    jsonl.write_text(
        '{"image_path": "/nonexistent/img.jpg", "attributes": {"sleeve_length": "short", "collar": "v-neck"}}\n'
        '{"image_path": "/also/none", "attributes": {"hood": "yes"}}\n',
        encoding="utf-8",
    )
    out = tmp_path / "out"
    result = build_from_external(jsonl, out, include_images=False)
    assert result.get("error") is None
    assert result["valid_examples"] == 2
    out_file = Path(result["output_jsonl"])
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "conversations" in first
    roles = [c["role"] for c in first["conversations"]]
    assert roles == ["system", "user", "assistant"]
    assert first["conversations"][0]["value"] == SYSTEM_PROMPT_EN
    asst = json.loads(first["conversations"][2]["value"])
    assert asst["sleeve_length"]["value"] == "short"
    assert asst["collar"]["value"] == "v-neck"


def test_build_from_external_empty_attributes_skipped(tmp_path):
    from fine_tune.dataset_builder import build_from_external
    jsonl = tmp_path / "ext.jsonl"
    jsonl.write_text(
        '{"image_path": "x", "attributes": {}}\n'
        '{"image_path": "y", "attributes": {"a": "b"}}\n',
        encoding="utf-8",
    )
    result = build_from_external(jsonl, tmp_path / "out", include_images=False)
    assert result["valid_examples"] == 1


def test_merge_datasets(tmp_path):
    from fine_tune.dataset_builder import merge_datasets
    (tmp_path / "a.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"id": 2}\n{"id": 3}\n', encoding="utf-8")
    out = tmp_path / "merged"
    result = merge_datasets(tmp_path / "a.jsonl", tmp_path / "b.jsonl", out)
    assert result["valid_examples"] == 3
    train = out / "train.jsonl"
    assert train.exists()
    lines = train.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_merge_datasets_extra_jsonl(tmp_path):
    from fine_tune.dataset_builder import merge_datasets
    (tmp_path / "a.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    (tmp_path / "c.jsonl").write_text('{"y": 2}\n', encoding="utf-8")
    out = tmp_path / "merged2"
    r = merge_datasets(tmp_path / "a.jsonl", None, out, extra_jsonl_paths=[tmp_path / "c.jsonl"])
    assert r["valid_examples"] == 2


def test_result_dict_to_sharegpt_no_image():
    from fine_tune.dataset_builder import result_dict_to_sharegpt
    r = {
        "offer_id": "1",
        "picture_url": "",
        "direction_attributes": {
            "clothing": {"color": {"value": "чёрный", "confidence": 92}},
        },
        "text_detection": {},
    }
    ex = result_dict_to_sharegpt(r, include_image=False)
    assert ex is not None
    asst = json.loads(ex["conversations"][2]["value"])
    assert asst["color"]["value"] == "чёрный"


def test_dedupe_sharegpt_examples_by_url():
    from fine_tune.dataset_builder import dedupe_sharegpt_examples
    a = {"_meta": {"picture_url": "https://cdn.example.com/a.jpg?w=100"}}
    b = {"_meta": {"picture_url": "https://cdn.example.com/a.jpg?w=200"}}
    c = {"_meta": {"picture_url": "https://cdn.example.com/b.jpg"}}
    out = dedupe_sharegpt_examples([a, b, c])
    assert len(out) == 2
    assert out[0] is a and out[1] is c


def test_dedupe_train_jsonl_file(tmp_path):
    from fine_tune.dataset_builder import dedupe_train_jsonl_file
    p = tmp_path / "t.jsonl"
    rows = [
        {"_meta": {"picture_url": "https://x.com/i.jpg?q=1"}},
        {"_meta": {"picture_url": "https://x.com/i.jpg?q=2"}},
        {"_meta": {"offer_id": "1"}},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    r = dedupe_train_jsonl_file(p)
    assert r["dropped"] == 1
    assert r["kept"] == 2


def test_export_low_confidence_review(tmp_path):
    import sqlite3

    from fine_tune.dataset_builder import export_low_confidence_review

    db = tmp_path / "r.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE results (offer_id TEXT PRIMARY KEY, name TEXT, category TEXT, picture_url TEXT, "
        "avg_confidence INTEGER, text_json TEXT, attributes_json TEXT, error TEXT, run_ts REAL, model_name TEXT)"
    )
    con.execute(
        "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("low1", "n", "", "http://p", 50, "{}", "{}", "", 0.0, ""),
    )
    con.execute(
        "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("skip", "n", "", "http://p2", 40, "{}", "{}", "", 0.0, ""),
    )
    con.commit()
    con.close()
    outp = tmp_path / "low.json"
    r = export_low_confidence_review(db, outp, max_confidence=80, limit=10, skip_offer_ids={"skip"})
    assert r["count"] == 1
    data = json.loads(outp.read_text(encoding="utf-8"))
    assert data["items"][0]["offer_id"] == "low1"


def test_build_train_results_jsonl_from_sqlite(tmp_path):
    from fine_tune.dataset_builder import build_train_results_jsonl
    import sqlite3

    db = tmp_path / "r.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE results (offer_id TEXT PRIMARY KEY, name TEXT, category TEXT, picture_url TEXT, "
        "avg_confidence INTEGER, text_json TEXT, attributes_json TEXT, error TEXT, run_ts REAL, model_name TEXT)"
    )
    attrs = {"clothing": {"hood": {"value": "да", "confidence": 90}}}
    con.execute(
        "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("o1", "n", "c", "", 95, "{}", json.dumps(attrs, ensure_ascii=False), "", 0.0, ""),
    )
    con.commit()
    con.close()
    out_j = tmp_path / "tr.jsonl"
    r = build_train_results_jsonl(db, out_j, include_images=False, queued_offer_ids=["o1"])
    assert r["valid_examples"] == 1
    line = json.loads(out_j.read_text(encoding="utf-8").strip())
    assert "conversations" in line
