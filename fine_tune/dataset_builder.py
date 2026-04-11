#!/usr/bin/env python3
"""
Build LoRA training dataset from human-corrected predictions.

Input:  corrections.json  (list of correction dicts)
Output: fine_tune/dataset/<project>/train.jsonl  (ShareGPT format for unsloth)
        fine_tune/dataset/<project>/train_export.json  (human-readable)
"""

import base64
import json
import sqlite3
from pathlib import Path
from typing import Callable, Optional

import requests

from picture_dedupe import normalize_picture_url


ATTRIBUTE_LABELS = {
    "sleeve_length": "Длина рукава",
    "fastener": "Застёжка",
    "hood": "Капюшон",
    "collar": "Воротник",
    "pockets": "Карманы",
    "gender_target": "Целевой пол",
}

SYSTEM_PROMPT = (
    "Ты — эксперт по анализу одежды. "
    "По фотографии определяй характеристики одежды и надписи в строгом JSON-формате."
)

SYSTEM_PROMPT_EN = (
    "You are an expert at analyzing clothing. "
    "From a photo, identify clothing attributes and text on clothing. Reply with strict JSON only."
)


def _download_image_b64(url: str, timeout: int = 30) -> str | None:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return base64.b64encode(r.content).decode("ascii")
    except Exception:
        return None


def _correction_to_sharegpt(correction: dict, include_image: bool = False) -> dict | None:
    """Convert one correction entry to ShareGPT format."""
    offer_id = correction.get("offer_id", "")
    picture_url = correction.get("picture_url", "")
    corrected_attrs = correction.get("corrected_attributes", {})
    corrected_text = correction.get("corrected_text", {})

    if not corrected_attrs and not corrected_text:
        return None

    # Build expected answer JSON
    answer: dict = {}
    if corrected_attrs:
        for key, val in corrected_attrs.items():
            answer[key] = {"value": val, "confidence": 95}
    if corrected_text:
        answer["text_on_clothing"] = corrected_text

    user_content = []
    if include_image and picture_url:
        img_b64 = _download_image_b64(picture_url)
        if img_b64:
            user_content.append({"type": "image", "image": img_b64})

    user_content.append({
        "type": "text",
        "text": "Определи характеристики одежды и надписи на ней. Ответь строго JSON.",
    })

    return {
        "conversations": [
            {"role": "system", "value": SYSTEM_PROMPT},
            {"role": "user", "value": user_content},
            {"role": "assistant", "value": json.dumps(answer, ensure_ascii=False)},
        ],
        "_meta": {"offer_id": offer_id, "picture_url": picture_url},
    }


def direction_attributes_to_flat_answer(direction_attributes: dict | None) -> dict:
    """Слить блоки направлений в плоский {attr_key: {value, confidence}}."""
    out: dict = {}
    for _did, block in (direction_attributes or {}).items():
        if not isinstance(block, dict) or block.get("error"):
            continue
        for k, v in block.items():
            if k == "error" or not isinstance(v, dict):
                continue
            val = v.get("value")
            if val is None or (isinstance(val, str) and not val.strip()):
                continue
            try:
                conf = int(v.get("confidence") or 85)
            except (TypeError, ValueError):
                conf = 85
            out[k] = {"value": val, "confidence": conf}
    return out


def _text_detection_for_answer(text_detection: dict | None) -> dict | None:
    if not isinstance(text_detection, dict) or text_detection.get("error"):
        return None
    texts = text_detection.get("texts")
    if isinstance(texts, list) and texts:
        return {
            "texts": [str(t) for t in texts if t is not None],
            "text_found": bool(text_detection.get("text_found", True)),
        }
    return None


def result_dict_to_sharegpt(result: dict, include_image: bool = False) -> dict | None:
    """Один результат анализа (как из SQLite) → ShareGPT; target — текущие атрибуты из БД (часто RU после translate)."""
    offer_id = str(result.get("offer_id") or "").strip()
    picture_url = (result.get("picture_url") or "").strip()
    answer = direction_attributes_to_flat_answer(result.get("direction_attributes"))
    td_ans = _text_detection_for_answer(result.get("text_detection"))
    if td_ans:
        answer["text_on_clothing"] = td_ans
    if not answer:
        return None
    user_content = []
    if include_image and picture_url:
        img_b64 = _download_image_b64(picture_url)
        if img_b64:
            user_content.append({"type": "image", "image": img_b64})
    user_content.append({
        "type": "text",
        "text": "Определи характеристики одежды и надписи на ней. Ответь строго JSON.",
    })
    return {
        "conversations": [
            {"role": "system", "value": SYSTEM_PROMPT},
            {"role": "user", "value": user_content},
            {"role": "assistant", "value": json.dumps(answer, ensure_ascii=False)},
        ],
        "_meta": {"offer_id": offer_id, "picture_url": picture_url, "source": "results"},
    }


def _db_row_to_result_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["text_detection"] = json.loads(d.get("text_json") or "{}")
    raw_attrs = json.loads(d.get("attributes_json") or "{}")
    if raw_attrs and not any(k in raw_attrs for k in ("clothing", "other")):
        raw_attrs = {"clothing": raw_attrs}
    d["direction_attributes"] = raw_attrs
    return d


def build_train_results_jsonl(
    db_path: str | Path,
    output_jsonl: str | Path,
    *,
    include_images: bool = True,
    queued_offer_ids: list[str] | None = None,
    skip_offer_ids: set[str] | None = None,
    auto_min_confidence: int | None = None,
    auto_max_examples: int = 400,
) -> dict:
    """
    Примеры из SQLite results: явная очередь offer_id + опционально авто-выбор по avg_confidence.
    skip_offer_ids: не брать (например уже есть в corrections — их даёт отдельный train из правок).
    """
    db_path = Path(db_path)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        return {"error": f"Results DB not found: {db_path}", "valid_examples": 0}
    skip = set(skip_offer_ids or set())
    queued = [str(x).strip() for x in (queued_offer_ids or []) if str(x).strip()]
    examples: list[dict] = []
    seen: set[str] = set()

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        for oid in queued:
            if oid in skip or oid in seen:
                continue
            row = con.execute("SELECT * FROM results WHERE offer_id = ?", (oid,)).fetchone()
            if not row:
                continue
            r = _db_row_to_result_dict(row)
            if (r.get("error") or "").strip():
                continue
            ex = result_dict_to_sharegpt(r, include_images)
            if ex:
                examples.append(ex)
                seen.add(oid)

        if auto_min_confidence is not None:
            q = (
                "SELECT * FROM results WHERE avg_confidence >= ? AND (error IS NULL OR error = '') "
                "ORDER BY avg_confidence DESC"
            )
            rows = con.execute(q, (int(auto_min_confidence),)).fetchall()
            auto_added = 0
            cap = auto_max_examples if auto_max_examples and auto_max_examples > 0 else 10**9
            for row in rows:
                if auto_added >= cap:
                    break
                r = _db_row_to_result_dict(row)
                oid = str(r.get("offer_id") or "").strip()
                if not oid or oid in skip or oid in seen:
                    continue
                if (r.get("error") or "").strip():
                    continue
                ex = result_dict_to_sharegpt(r, include_images)
                if ex:
                    examples.append(ex)
                    seen.add(oid)
                    auto_added += 1
    finally:
        con.close()

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return {"valid_examples": len(examples), "output_jsonl": str(output_jsonl), "error": None}


def build_dataset(
    corrections_path: str | Path,
    output_dir: str | Path,
    include_images: bool = False,
    min_corrections: int = 1,
) -> dict:
    """
    Build JSONL dataset from corrections file.

    Returns summary: {total_corrections, valid_examples, output_jsonl, output_json}
    """
    corrections_path = Path(corrections_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corrections_path.exists():
        return {"error": f"Corrections file not found: {corrections_path}", "valid_examples": 0}

    with open(corrections_path, encoding="utf-8") as f:
        corrections: list[dict] = json.load(f)

    if len(corrections) < min_corrections:
        return {
            "error": f"Need at least {min_corrections} corrections, got {len(corrections)}",
            "valid_examples": 0,
        }

    examples = []
    for c in corrections:
        ex = _correction_to_sharegpt(c, include_images)
        if ex:
            examples.append(ex)

    output_jsonl = output_dir / "train.jsonl"
    output_json = output_dir / "train_export.json"

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    return {
        "total_corrections": len(corrections),
        "valid_examples": len(examples),
        "output_jsonl": str(output_jsonl),
        "output_json": str(output_json),
        "error": None,
    }


def _external_row_to_sharegpt(row: dict, images_dir: Path | None, include_image: bool) -> dict | None:
    """Convert one external JSONL row to ShareGPT format. row: image_path or image_url, attributes (dict EN)."""
    image_path = row.get("image_path") or row.get("image_url", "")
    attrs = row.get("attributes") or {}
    if not attrs:
        return None
    answer = {k: {"value": v, "confidence": 95} for k, v in attrs.items()}
    user_content = []
    if include_image and image_path:
        path = Path(image_path)
        if not path.is_absolute() and images_dir:
            path = images_dir / path
        if path.exists():
            try:
                b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                user_content.append({"type": "image", "image": b64})
            except Exception:
                pass
    user_content.append({
        "type": "text",
        "text": "Describe clothing attributes from the image. Reply with JSON only (keys: attribute names, values: {\"value\": \"...\", \"confidence\": 0-100}).",
    })
    return {
        "conversations": [
            {"role": "system", "value": SYSTEM_PROMPT_EN},
            {"role": "user", "value": user_content},
            {"role": "assistant", "value": json.dumps(answer, ensure_ascii=False)},
        ],
        "_meta": {"image_path": image_path},
    }


def build_from_external(
    jsonl_path: str | Path,
    output_dir: str | Path,
    include_images: bool = True,
    max_examples: int = 0,
    images_dir: str | Path | None = None,
    progress_callback: Optional[Callable[..., None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    skip_first_n: int = 0,
) -> dict:
    """
    Build ShareGPT JSONL from external JSONL.
    skip_first_n: skip this many lines at start of source (to continue after stop).
    When skip_first_n > 0, existing train_external.jsonl is loaded and new examples appended.
    """
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not jsonl_path.exists():
        return {"error": f"File not found: {jsonl_path}", "valid_examples": 0}

    out_jsonl = output_dir / "train_external.jsonl"
    existing_examples: list[dict] = []
    if skip_first_n > 0 and out_jsonl.exists():
        with open(out_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    img_dir = Path(images_dir) if images_dir else jsonl_path.parent
    examples: list[dict] = []
    stopped = False
    skipped = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if stop_check and stop_check():
                stopped = True
                break
            line = line.strip()
            if not line:
                continue
            if skipped < skip_first_n:
                skipped += 1
                continue
            if max_examples and len(examples) >= max_examples:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ex = _external_row_to_sharegpt(row, img_dir, include_images)
            if ex:
                examples.append(ex)
            if progress_callback and len(examples) % 100 == 0 and len(examples) > 0:
                progress_callback(len(existing_examples) + len(examples))
    if progress_callback:
        progress_callback(len(existing_examples) + len(examples))

    all_examples = existing_examples + examples
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return {
        "valid_examples": len(all_examples),
        "output_jsonl": str(out_jsonl),
        "error": None,
        "stopped": stopped,
    }


def dedupe_sharegpt_examples(examples: list[dict]) -> list[dict]:
    """
    Один пример на нормализованный URL картинки (первый в порядке следования выигрывает —
    обычно сначала идут правки, затем внешний JSONL, затем авто из БД).
    Без URL — дедуп по offer_id в _meta; иначе строка остаётся.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for ex in examples:
        meta = ex.get("_meta") if isinstance(ex.get("_meta"), dict) else {}
        url = normalize_picture_url(meta.get("picture_url") or "")
        if url:
            if url in seen:
                continue
            seen.add(url)
        else:
            oid = str(meta.get("offer_id") or "").strip()
            if oid:
                k = f"oid:{oid}"
                if k in seen:
                    continue
                seen.add(k)
        out.append(ex)
    return out


def dedupe_train_jsonl_file(path: str | Path) -> dict:
    """Перезаписать JSONL, убрав дубликаты по URL/offer_id. Возвращает {kept, dropped, error}."""
    path = Path(path)
    if not path.exists():
        return {"kept": 0, "dropped": 0, "error": "missing"}
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except OSError as e:
        return {"kept": 0, "dropped": 0, "error": str(e)}
    n0 = len(rows)
    rows = dedupe_sharegpt_examples(rows)
    with open(path, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return {"kept": len(rows), "dropped": n0 - len(rows), "error": None}


def export_low_confidence_review(
    db_path: str | Path,
    out_path: str | Path,
    *,
    max_confidence: int = 78,
    limit: int = 250,
    skip_offer_ids: set[str] | None = None,
) -> dict:
    """Список офферов с низкой уверенностью для последующей разметки (без скачивания картинок)."""
    db_path = Path(db_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip = skip_offer_ids or set()
    if not db_path.exists():
        return {"error": "no db", "count": 0}
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT offer_id, name, picture_url, avg_confidence FROM results "
            "WHERE avg_confidence < ? AND (error IS NULL OR error = '') "
            "ORDER BY avg_confidence ASC",
            (int(max_confidence),),
        ).fetchall()
    finally:
        con.close()
    items: list[dict] = []
    for row in rows:
        oid = str(row["offer_id"] or "").strip()
        if not oid or oid in skip:
            continue
        items.append(
            {
                "offer_id": oid,
                "name": row["name"] or "",
                "picture_url": row["picture_url"] or "",
                "avg_confidence": int(row["avg_confidence"] or 0),
            }
        )
        if len(items) >= int(limit):
            break
    payload = {
        "generated_for": "manual_review_or_finetune_queue",
        "max_confidence": int(max_confidence),
        "items": items,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"count": len(items), "path": str(out_path), "error": None}


def build_eval_anchors_jsonl(
    db_path: str | Path,
    corrections: list[dict],
    cache_dir: str | Path,
    out_path: str | Path,
    *,
    max_total: int = 36,
    image_max_size: int = 1024,
    timeout: int = 25,
) -> dict:
    """
    JSONL для блока «Оценка до/после»: только строки с локальным image_path (кэш по URL).
    Половина — из правок (по одному на URL), остальное — случайная выборка уверенных из БД.
    """
    import random

    from attribute_detector import ensure_image_cached

    db_path = Path(db_path)
    cache_dir = Path(cache_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        return {"valid_examples": 0, "error": "no db", "output_jsonl": str(out_path)}

    max_total = max(4, min(int(max_total), 200))
    n_corr_target = max_total // 2

    picked_urls: set[str] = set()
    lines_out: list[dict] = []

    def add_line(offer_id: str, pic_url: str, source: str) -> None:
        if not pic_url.strip():
            return
        key = normalize_picture_url(pic_url)
        if not key or key in picked_urls:
            return
        p = ensure_image_cached(pic_url.strip(), cache_dir, max_size=image_max_size, timeout=timeout)
        if not p or not p.is_file():
            return
        picked_urls.add(key)
        lines_out.append(
            {
                "image_path": str(p.resolve()),
                "offer_id": offer_id,
                "source": source,
                "attributes": {},
            }
        )

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        for c in corrections:
            if len(lines_out) >= n_corr_target:
                break
            oid = str(c.get("offer_id") or "").strip()
            url = (c.get("picture_url") or "").strip()
            if not url and oid:
                row = con.execute("SELECT picture_url FROM results WHERE offer_id=?", (oid,)).fetchone()
                if row and row[0]:
                    url = str(row[0]).strip()
            if oid and url:
                add_line(oid, url, "correction")

        pool = con.execute(
            "SELECT offer_id, picture_url, avg_confidence FROM results "
            "WHERE avg_confidence >= 85 AND (error IS NULL OR error = '') AND picture_url != ''"
        ).fetchall()
    finally:
        con.close()
    random.seed(42)
    pool_list = list(pool)
    random.shuffle(pool_list)
    for row in pool_list:
        if len(lines_out) >= max_total:
            break
        oid = str(row["offer_id"] or "").strip()
        url = (row["picture_url"] or "").strip()
        add_line(oid, url, "auto_high")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in lines_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"valid_examples": len(lines_out), "error": None, "output_jsonl": str(out_path)}


def merge_datasets(
    corrections_jsonl_path: str | Path,
    external_jsonl_path: str | Path | None,
    output_dir: str | Path,
    extra_jsonl_paths: list | None = None,
) -> dict:
    """Склеить JSONL в один train.jsonl: правки, внешний датасет, примеры из results (очередь/авто)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = [Path(corrections_jsonl_path)]
    if external_jsonl_path:
        paths.append(Path(external_jsonl_path))
    for p in extra_jsonl_paths or []:
        if p:
            paths.append(Path(p))
    merged = []
    for p in paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        merged.append(json.loads(line))
    out_path = output_dir / "train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in merged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return {"valid_examples": len(merged), "output_jsonl": str(out_path), "error": None}
