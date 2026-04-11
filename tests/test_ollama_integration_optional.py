"""
Опциональный прогон vision на реальной картинке (лекарственная упаковка).

Не гоняется в CI по умолчанию. Локально:
  set RUN_OLLAMA_INTEGRATION=1
  set OLLAMA_TEST_MODEL=qwen2.5:9b
  pytest tests/test_ollama_integration_optional.py -v --no-cov
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import requests

import attribute_detector as ad

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "drug_packaging_sample.png"
_ROOT = Path(__file__).resolve().parents[1]
_RUN_PROMPT = _ROOT / "run_prompt_last.json"


@pytest.mark.skipif(os.environ.get("RUN_OLLAMA_INTEGRATION") != "1", reason="set RUN_OLLAMA_INTEGRATION=1")
def test_ollama_original_name_matches_dominant_print_on_fixture():
    if not _FIXTURE.is_file():
        pytest.skip(f"missing fixture {_FIXTURE}")
    if not _RUN_PROMPT.is_file():
        pytest.skip("run_prompt_last.json missing")
    url = (os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
    try:
        r = requests.get(f"{url}/api/tags", timeout=3)
        r.raise_for_status()
    except OSError:
        pytest.skip("Ollama not reachable")

    b64 = ad._path_to_base64(_FIXTURE)
    assert b64
    data = json.loads(_RUN_PROMPT.read_text(encoding="utf-8"))
    model = (os.environ.get("OLLAMA_TEST_MODEL") or "qwen2.5:9b").strip()

    out = ad.detect_attributes_for_direction(
        b64,
        "test_dir",
        "Тест",
        [{"key": "Original_name", "label": "Original_name", "options": []}],
        data["task_instruction"],
        model,
        url,
        timeout=int(os.environ.get("OLLAMA_TEST_TIMEOUT") or "180"),
        product_name=None,
        vertical="Тест",
        task_constraints=data["task_constraints"],
        task_examples=data["task_examples"],
        task_target_attributes=data.get("task_target_attributes"),
    )
    err = out.get("error")
    if err:
        if "not found" in str(err).lower():
            pytest.skip(str(err))
        pytest.fail(str(out))
    block = out.get("Original_name") or out.get("original_name") or {}
    val = (block.get("value") or "").strip()
    assert val, f"empty value, full out={out!r}"
    low = val.lower()
    assert "глюкоза" in low or "glucose" in low, f"expected dominant print, got {val!r}"
