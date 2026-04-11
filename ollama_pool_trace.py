"""Заголовки для прокси :11435 — виджет пула показывает http.inflight (кто держит слот к Ollama)."""
from __future__ import annotations


def pool_trace_headers(ollama_url: str, *, project: str, label: str) -> dict[str, str]:
    u = (ollama_url or "").strip()
    if ":11435" not in u:
        return {}
    return {
        "X-Ollama-Pool-Project": (project or "image_description")[:100],
        "X-Ollama-Pool-Label": (label or "ollama")[:200],
    }
