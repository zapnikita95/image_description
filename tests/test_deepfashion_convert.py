"""Tests for scripts/convert_deepfashion_to_jsonl (DeepFashion parsing)."""
import json
from pathlib import Path

import pytest


def test_parse_list_attr_cloth_with_count(tmp_path):
    """list_attr_cloth.txt: first line = count, then names."""
    from scripts.convert_deepfashion_to_jsonl import parse_list_attr_cloth
    p = tmp_path / "list_attr_cloth.txt"
    p.write_text("3\nsleeve_length\ncollar\nhood\n", encoding="utf-8")
    names = parse_list_attr_cloth(p)
    assert names == ["sleeve_length", "collar", "hood"]


def test_parse_list_attr_cloth_no_count(tmp_path):
    """When first line is not a number, treat all lines as names."""
    from scripts.convert_deepfashion_to_jsonl import parse_list_attr_cloth
    p = tmp_path / "list_attr_cloth.txt"
    p.write_text("sleeve_length\ncollar\n", encoding="utf-8")
    names = parse_list_attr_cloth(p)
    assert "sleeve_length" in names and "collar" in names


def test_parse_list_attr_img(tmp_path):
    from scripts.convert_deepfashion_to_jsonl import parse_list_attr_img
    root = tmp_path
    anno = root / "Anno"
    anno.mkdir()
    (anno / "list_attr_img.txt").write_text(
        "2\n"
        "img1.jpg 1 -1 0\n"
        "img2.jpg -1 1 0\n",
        encoding="utf-8",
    )
    attr_names = ["a", "b", "c"]
    examples = parse_list_attr_img(anno / "list_attr_img.txt", attr_names, root)
    assert len(examples) == 2
    assert examples[0]["attributes"].get("a") == "yes"
    assert examples[0]["attributes"].get("b") == "no"
    assert examples[1]["attributes"].get("a") == "no"
    assert examples[1]["attributes"].get("b") == "yes"


def test_convert_deepfashion_to_jsonl_integration(tmp_path):
    """Run full conversion with mock Anno files."""
    root = tmp_path
    anno = root / "Anno"
    anno.mkdir()
    (anno / "list_attr_cloth.txt").write_text("2\nsleeve_length\ncollar\n", encoding="utf-8")
    (anno / "list_attr_img.txt").write_text(
        "2\n"
        "f1.jpg 1 -1\n"
        "f2.jpg -1 1\n",
        encoding="utf-8",
    )
    out_jsonl = tmp_path / "out.jsonl"
    import scripts.convert_deepfashion_to_jsonl as conv
    attr_names = conv.parse_list_attr_cloth(anno / "list_attr_cloth.txt")
    examples = conv.parse_list_attr_img(anno / "list_attr_img.txt", attr_names, root)
    assert len(examples) == 2
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert "image_path" in row and "attributes" in row
    assert row["attributes"].get("sleeve_length") == "yes"
