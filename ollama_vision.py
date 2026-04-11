#!/usr/bin/env python3
"""Send image + prompt to Ollama vision API and return text response."""

import base64
import json
import sys
from pathlib import Path

import requests

from ollama_pool_trace import pool_trace_headers

# Пул :11435 → Ollama :11434 (см. project_manager / ollama-queue-proxy).
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11435"
DEFAULT_MODEL = "qwen3.5:2b"


def image_to_base64(image_path: str | Path) -> str:
    """Read image file and return base64 string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def describe_image(
    image_path: str | Path | None = None,
    image_url: str | None = None,
    image_base64: str | None = None,
    prompt: str = "",
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
) -> str:
    """
    Send image and prompt to Ollama chat API; return assistant message text.

    Image can be provided as: local path, URL (will be downloaded and sent as base64), or base64 string.
    """
    if not prompt:
        return ""

    content = []
    if image_path or image_url or image_base64:
        img_b64 = image_base64
        if image_path:
            img_b64 = image_to_base64(image_path)
        elif image_url:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            img_b64 = base64.b64encode(resp.content).decode("ascii")
        if img_b64:
            content.append({"type": "image", "image": img_b64})
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }
    url = f"{ollama_url.rstrip('/')}/api/chat"
    try:
        r = requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers=pool_trace_headers(ollama_url, project="image_description", label="describe_image"),
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content") or ""
    except requests.RequestException as e:
        return f"[ERROR: {e}]"


def describe_clothing_from_image(
    image_path: str | Path | None = None,
    image_url: str | None = None,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
) -> tuple[str, str]:
    """
    Get text on clothing and adjective description for one image.
    Returns (text_on_clothing, description_adjectives).
    """
    prompt = """По фото предмета одежды ответь строго в два пункта на русском языке:
1. Текст на одежде: прочитай и выпиши весь видимый текст (надписи на футболке, принты). Если текста нет — напиши «нет».
2. Описание: 3–7 прилагательных через запятую (стиль, цвет, вид, принт). Только прилагательные, без предложений.

Формат ответа:
Текст на одежде: ...
Описание: ..."""

    raw = describe_image(
        image_path=image_path,
        image_url=image_url,
        prompt=prompt,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
    )

    text_on = ""
    desc = ""
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("текст на одежде"):
            text_on = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("описание"):
            desc = line.split(":", 1)[-1].strip()
    if not text_on and not desc:
        text_on = raw[:500] if raw else ""
    return text_on or "", desc or ""
