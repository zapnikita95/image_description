"""
Дедуп картинок перед vision: по URL или по лёгкому dHash файла в локальном кэше.
Без тяжёлых зависимостей — только Pillow; в памяти держим маленькую серую матрицу на файл.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def dedupe_mode_from_config(cfg: dict | None) -> str:
    """off | url | phash; старая галка process_unique_pictures_only → url."""
    if not cfg:
        return "off"
    raw = cfg.get("process_unique_pictures_mode")
    m = (raw or "").strip().lower()
    if m in ("off", "url", "phash"):
        return m
    # Старый баг UI Gradio: в JSON попадала русская подпись радиокнопки вместо ключа.
    if "phash" in m or "dhash" in m or "содержим" in m:
        return "phash"
    if "первой картин" in m or "по url" in m:
        return "url"
    if "выключ" in m:
        return "off"
    if cfg.get("process_unique_pictures_only"):
        return "url"
    return "off"


def first_picture_url(offer: dict) -> str:
    pics = offer.get("picture_urls") or []
    return (pics[0] or "").strip() if pics else ""


def normalize_picture_url(url: str) -> str:
    """
    Ключ дедупа по «одной и той же картинке»: убираем пробелы, fragment и query —
    у CDN часто один JPEG с разными ?w= / трекингом.
    Схема и host+path сохраняются (разные пути = разные файлы).
    """
    u = (url or "").strip()
    if not u:
        return ""
    try:
        p = urlparse(u)
        # scheme netloc path — без params/query/fragment
        clean = urlunparse((p.scheme.lower(), p.netloc.lower(), p.path or "", "", "", ""))
        return clean.rstrip("/") or u
    except Exception:
        return u


def dhash64_file(path: Path, memo: dict[str, str] | None = None) -> str | None:
    """
    dHash 8×8 по уменьшенному серому 9×8. Совпадение строки hex ⇒ считаем один снимок.
    Не fuzzy: чуть разные JPEG могут разойтись — тогда сработает fallback по URL.
    """
    try:
        r = path.resolve()
        sk = str(r)
        if memo is not None and sk in memo:
            return memo[sk]
        from PIL import Image

        with Image.open(r) as im:
            g = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            px = list(g.getdata())
        bits = 0
        for row in range(8):
            o = row * 9
            for col in range(8):
                if px[o + col] > px[o + col + 1]:
                    bits |= 1 << (row * 8 + col)
        h = f"{bits:016x}"
        if memo is not None:
            memo[sk] = h
        return h
    except Exception:
        return None


def dhash256_file(path: Path, memo: dict[str, str] | None = None) -> str | None:
    """
    dHash 16×16 бит: серое изображение 17×16, сравнение соседних пикселей по горизонтали.
    Точнее 64-битного варианта для почти одинаковых обложек; всё ещё ~272 байта пикселей в RAM.
    """
    try:
        r = path.resolve()
        sk = str(r)
        if memo is not None and sk in memo:
            return memo[sk]
        from PIL import Image

        with Image.open(r) as im:
            g = im.convert("L").resize((17, 16), Image.Resampling.LANCZOS)
            px = list(g.getdata())
        bits = 0
        for row in range(16):
            o = row * 17
            for col in range(16):
                if px[o + col] > px[o + col + 1]:
                    bits |= 1 << (row * 16 + col)
        h = f"{bits:064x}"
        if memo is not None:
            memo[sk] = h
        return h
    except Exception:
        return None


def group_offers_by_picture_dedupe(
    offers: list[dict],
    mode: str,
    cache_dir: Path,
    max_size: int,
    ensure_cached: Callable[..., Path | str | None],
    stop_event: threading.Event | None = None,
) -> list[list[dict]]:
    """
    mode=url — как раньше, ключ первый URL.
    mode=phash — ключ по dHash скачанного в cache_dir файла; нет файла / ошибка — fallback на URL.
    stop_event — если установлен (например «Остановить»), прерываем группировку: уже собранные группы
    сохраняются, оставшиеся офферы идут по одному в группе (vision вызовется отдельно, но цикл не блокируется).
    """
    m = (mode or "off").strip().lower()
    if m not in ("url", "phash"):
        return [[o] for o in offers]
    order: list[str] = []
    groups: dict[str, list[dict]] = {}
    memo: dict[str, str] = {}
    for idx, o in enumerate(offers):
        if stop_event is not None and stop_event.is_set():
            built = [groups[k] for k in order]
            for rest in offers[idx:]:
                built.append([rest])
            return built
        url = first_picture_url(o)
        url_key = normalize_picture_url(url)
        if m == "url":
            key = url_key if url_key else f"__nopicture__:{o.get('offer_id', '')}"
        else:
            key = None
            if url:
                p = ensure_cached(url, cache_dir, max_size)
                if p:
                    pp = Path(p)
                    if pp.is_file():
                        hp = dhash256_file(pp, memo)
                        if hp:
                            key = f"__ph__:{hp}"
            if key is None:
                key = f"__urlfb__:{url_key}" if url_key else f"__nopicture__:{o.get('offer_id', '')}"
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(o)
    return [groups[k] for k in order]
