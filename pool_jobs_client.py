"""
Запросы к API очереди пула Ollama (хост из настроек «Ollama URL», обычно :11435).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def _base(ollama_url: str) -> str:
    return (ollama_url or "").strip().rstrip("/")


def push_pool_http_capacity(
    ollama_url: str,
    capacity: int,
    *,
    timeout_s: float = 8.0,
) -> tuple[bool, str]:
    """
    POST /_ollama_queue/http_capacity — применить число слотов к работающему пулу.
    Для прямого Ollama (:11434) вернёт ошибку (нет маршрута).
    """
    b = _base(ollama_url)
    if not b:
        return False, "пустой Ollama URL"
    try:
        n = int(capacity)
    except (TypeError, ValueError):
        return False, "некорректное число слотов"
    n = max(1, min(n, 32))
    body = json.dumps({"capacity": n}, ensure_ascii=False).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{b}/_ollama_queue/http_capacity",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            data = json.loads(r.read().decode())
        if data.get("ok") and "capacity" in data:
            return True, f"пул: слотов HTTP = {data.get('capacity')}"
        return False, json.dumps(data, ensure_ascii=False)[:400]
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode(errors="replace")
        except Exception:
            raw = str(e)
        return False, f"HTTP {e.code}: {raw[:300]}"
    except Exception as e:
        return False, str(e)[:400]


def fetch_pool_status(ollama_url: str, *, timeout_s: float = 2.5) -> dict[str, Any] | None:
    b = _base(ollama_url)
    if not b:
        return None
    try:
        req = urllib.request.Request(
            f"{b}/_ollama_queue/status",
            method="GET",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def pool_backlog_severe(
    ollama_url: str,
    *,
    min_waiting: int = 4,
    min_pending_jobs: int = 8,
) -> bool:
    st = fetch_pool_status(ollama_url)
    if not st:
        return False
    h = st.get("http") or {}
    j = st.get("jobs") or {}
    try:
        w = int(h.get("waiting") or 0)
        p = int(j.get("pending") or 0)
        r = int(j.get("running") or 0)
    except (TypeError, ValueError):
        return False
    return w >= min_waiting or p + r >= min_pending_jobs


def enqueue_cli(
    ollama_url: str,
    *,
    title: str,
    project: str,
    cwd: str,
    argv: list[str],
    timeout_s: float = 30.0,
) -> tuple[bool, str]:
    b = _base(ollama_url)
    if not b:
        return False, "empty ollama_url"
    body = json.dumps(
        {"title": title, "project": project, "cwd": cwd, "argv": argv},
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{b}/_ollama_queue/jobs",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            data = json.loads(r.read().decode())
        jid = data.get("id")
        return (True, str(jid)) if jid else (False, json.dumps(data)[:400])
    except urllib.error.HTTPError as e:
        try:
            return False, (e.read().decode(errors="replace") or str(e))[:500]
        except Exception:
            return False, str(e)
    except Exception as e:
        return False, str(e)
