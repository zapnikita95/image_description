"""Tests for scripts/extract_attribute_keys_from_sharegpt_jsonl (key extraction from ShareGPT JSONL)."""
import json
from pathlib import Path

import pytest


def test_extract_keys_from_sharegpt_jsonl(tmp_path):
    """Extract unique attribute keys from ShareGPT-style JSONL."""
    from scripts.extract_attribute_keys_from_sharegpt_jsonl import (
        extract_keys_from_sharegpt_jsonl,
        _extract_assistant_json_value,
    )

    # One line: assistant returns JSON with keys a, b, c
    row = {
        "conversations": [
            {"role": "system", "value": "You are expert."},
            {"role": "user", "value": [{"type": "text", "text": "Describe."}]},
            {
                "role": "assistant",
                "value": '{"a": {"value": "yes", "confidence": 95}, "b": {"value": "no", "confidence": 95}, "c": {"value": "yes", "confidence": 95}}',
            },
        ]
    }
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    keys = extract_keys_from_sharegpt_jsonl(jsonl)
    assert keys == {"a", "b", "c"}


def test_extract_assistant_json_value_dict_value():
    """Assistant value can be a dict (e.g. some datasets store structured)."""
    from scripts.extract_attribute_keys_from_sharegpt_jsonl import _extract_assistant_json_value

    row = {
        "conversations": [
            {"role": "assistant", "value": {"sleeve_length": "short", "collar": "v-neck"}},
        ]
    }
    out = _extract_assistant_json_value(row)
    assert out is not None
    assert "sleeve_length" in out or "collar" in out


def test_extract_keys_multiple_lines(tmp_path):
    """Multiple lines: keys are merged."""
    from scripts.extract_attribute_keys_from_sharegpt_jsonl import extract_keys_from_sharegpt_jsonl

    lines = [
        {"conversations": [{"role": "assistant", "value": '{"x": {"value": "yes"}, "y": {"value": "no"}}'}]},
        {"conversations": [{"role": "assistant", "value": '{"y": {"value": "yes"}, "z": {"value": "no"}}'}]},
    ]
    jsonl = tmp_path / "train.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    keys = extract_keys_from_sharegpt_jsonl(jsonl)
    assert keys == {"x", "y", "z"}


def test_extract_keys_skip_invalid_lines(tmp_path):
    """Invalid JSON or missing assistant value are skipped."""
    from scripts.extract_attribute_keys_from_sharegpt_jsonl import extract_keys_from_sharegpt_jsonl

    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(
        '{"conversations": [{"role": "user", "value": "hi"}]}\n'
        'not json\n'
        '{"conversations": [{"role": "assistant", "value": "{\\"k\\": {}}"}]}\n',
        encoding="utf-8",
    )
    keys = extract_keys_from_sharegpt_jsonl(jsonl)
    assert keys == {"k"}
