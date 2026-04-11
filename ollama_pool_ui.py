"""
Статус пула Ollama (прокси :11435 → настоящий Ollama :11434). См. Desktop/ollama-queue-proxy.
"""
from __future__ import annotations

import html
import json
import urllib.error
import urllib.request


def fetch_ollama_pool_status_json(ollama_base_url: str, *, timeout_s: float = 2.5) -> dict | None:
    base = (ollama_base_url or "").strip().rstrip("/")
    if not base:
        return None
    url = f"{base}/_ollama_queue/status"
    try:
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def format_ollama_pool_status_html(ollama_base_url: str) -> str:
    d = fetch_ollama_pool_status_json(ollama_base_url)
    if not d:
        return (
            "<div style='padding:6px 12px;border-radius:6px;background:#fef3c7;border:1px solid #fcd34d;"
            "color:#92400e;font-size:13px'><b>Пул Ollama</b> — нет ответа от <code>/_ollama_queue/status</code>. "
            "Запусти <code>ollama_pool_service</code> на <b>11435</b> или проверь URL в Настройках.</div>"
        )
    h = d.get("http") or {}
    j = d.get("jobs") or {}
    failed = int(j.get("failed") or 0)
    failed_part = f", failed <b>{failed}</b>" if failed else ""
    hint = d.get("hint_jobs_vs_ollama") or d.get("hint") or ""
    hint_html = ""
    if hint:
        hint_html = (
            "<br><small style='opacity:0.92;line-height:1.35'>"
            + html.escape(str(hint))
            + "</small>"
        )
    return (
        "<div style='padding:6px 12px;border-radius:6px;background:#e0f2fe;border:1px solid #7dd3fc;"
        "color:#0c4a6e;font-size:13px'><b>Пул Ollama</b> — активно слотов "
        f"<b>{h.get('active', '?')}/{h.get('capacity', '?')}</b>, в очереди HTTP: <b>{h.get('waiting', 0)}</b> · "
        f"jobs pending <b>{j.get('pending', 0)}</b>, running <b>{j.get('running', 0)}</b>"
        f"{failed_part}{hint_html}</div>"
    )
