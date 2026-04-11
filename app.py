#!/usr/bin/env python3
"""
Image analyzer — Gradio UI
Tabs: Projects | Feed | Run | Results | Fine-tune | Settings
"""

import copy
import csv
import html
import json
import math
import os
import re
import socket
import subprocess
import sys
import sqlite3
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
import traceback
from collections import defaultdict
from pathlib import Path

import gradio as gr
import requests

import project_manager as pm
import pool_jobs_client
from ollama_pool_ui import format_ollama_pool_status_html
import feed_cache as fc
from picture_dedupe import dedupe_mode_from_config, group_offers_by_picture_dedupe, normalize_picture_url
from attribute_detector import (
    analyze_offer,
    compose_task_prompt_blocks,
    ensure_image_cached,
    inscription_mode_is_same_prompt,
    normalize_ollama_url,
    ollama_list_models,
    ollama_loaded_models,
    ollama_root_health_timeout_s,
    ollama_unload_model,
    resolve_inscription_model,
    warmup_ollama_model,
)

# ── State helpers ─────────────────────────────────────────────────────────────

_current_project: dict = {}
_run_log: list[str] = []
# id обёртки Textbox «Лог обработки» — для CSS и JS (умная прокрутка)
RUN_LOG_TEXTBOX_ELEM_ID = "image-desc-run-log"
# Создание проекта: без явного выбора вертикали кнопка не создаёт проект
CREATE_PROJECT_VERTICAL_PLACEHOLDER = "— Выберите вертикаль —"
# Защита от зависания UI при гигантском логе (в памяти хранится полный список строк)
_RUN_LOG_UI_MAX_LINES = 12000

_run_stop_event = threading.Event()
_run_worker_thread: threading.Thread | None = None
_run_start_lock = threading.Lock()

# Состояние дообучения LoRA: обновляется потоком обучения, читается таймером для прогресса в UI
_train_state: dict = {}
_train_stop_event = threading.Event()

# Состояние экспорта в Ollama: поток пишет лог, таймер обновляет UI
_export_state: dict = {}

# Состояние mini-теста пайплайна дообучения: поток пишет лог, таймер обновляет UI
_mini_test_state: dict = {}

# Сравнение vision-моделей Ollama (тестовый прогон)
_model_bench_state: dict = {}


def _proj() -> dict:
    return _current_project


def _run_log_text_for_ui(lines: list[str]) -> str:
    """Текст для поля лога: полный хвост, без «прыгающего» окна в 25 строк; при переборе — обрезка сверху."""
    if not lines:
        return ""
    if len(lines) <= _RUN_LOG_UI_MAX_LINES:
        return "\n".join(lines)
    n = len(lines)
    tail = lines[-_RUN_LOG_UI_MAX_LINES:]
    return f"(… скрыто {n - _RUN_LOG_UI_MAX_LINES} строк с начала; в консоли полный лог …)\n" + "\n".join(tail)


def _run_progress_status_md(processed: int, total: int, errors: int, *, batch_workers: int = 1) -> str:
    s = f"Обработано: **{processed}/{total}** | Ошибок: {errors}"
    if batch_workers > 1:
        s += f" | **×{batch_workers}** фото параллельно"
    return s


def _proj_name() -> str:
    return _current_project.get("name", "")


def start_model_bench(selected_models, n_offers):
    global _model_bench_state
    name = _proj_name()
    if not name:
        return "❌ Выберите проект на вкладке «Проекты»."
    if not selected_models:
        return "❌ Отметьте хотя бы одну модель в списке."
    _model_bench_state = {"thread": None, "log": [], "result_md": None, "error": None, "done": False}

    def worker():
        try:
            from scripts.bench_vision_models import format_markdown_table, run_benchmark

            def cb(s):
                _model_bench_state.setdefault("log", []).append(s)

            r = run_benchmark(
                name,
                list(selected_models),
                max(1, int(n_offers or 10)),
                None,
                progress_cb=cb,
            )
            _model_bench_state["result_md"] = format_markdown_table(r)
            _model_bench_state["raw"] = r
            _model_bench_state["done"] = True
        except Exception as e:
            _model_bench_state["error"] = str(e)
            _model_bench_state.setdefault("log", []).append(traceback.format_exc())
        finally:
            _model_bench_state["thread"] = None

    t = threading.Thread(target=worker, daemon=True)
    _model_bench_state["thread"] = t
    t.start()
    return "⏳ Запущено. Прогон может занять **много минут** (модели × карточки). Статус обновляется ниже…"


def poll_model_bench():
    global _model_bench_state
    t = _model_bench_state.get("thread")
    logs = _model_bench_state.get("log", [])
    log_tail = "\n".join(logs[-45:])
    if t and t.is_alive():
        return f"⏳ **Идёт сравнение моделей**\n\n```\n{log_tail}\n```"
    err = _model_bench_state.get("error")
    if err:
        return f"❌ **Ошибка:** {err}\n\n```\n{log_tail}\n```"
    if _model_bench_state.get("result_md"):
        return f"✅ **Готово**\n\n{_model_bench_state['result_md']}\n\n---\nЛог:\n```\n{log_tail}\n```"
    if _model_bench_state.get("done") is False and not logs:
        return gr.update()
    return gr.update()


def _results_db() -> Path | None:
    n = _proj_name()
    return pm.results_db_path(n) if n else None


def _cache_db() -> Path | None:
    n = _proj_name()
    return pm.cache_db_path(n) if n else None


def _target_attrs_from_lastp(lastp: dict) -> list[str]:
    raw = lastp.get("task_target_attributes")
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = (lastp.get("task_target_attribute") or "").strip()
    return [x.strip() for x in s.split("\n") if x.strip()]


def _apply_lastp_targets_to_config(cfg: dict, lastp: dict) -> None:
    attrs = _target_attrs_from_lastp(lastp)
    cfg["task_target_attributes"] = attrs
    single = (lastp.get("task_target_attribute") or "").strip()
    cfg["task_target_attribute"] = single if single else "\n".join(attrs)


def _config_for_result_card_badges() -> dict:
    """Проект + последнее задание с «Запуск» — порядок бейджей: целевые атрибуты задания первыми."""
    cfg = dict(_proj() or {})
    try:
        _apply_lastp_targets_to_config(cfg, pm.load_run_prompt_last())
    except Exception:
        pass
    return cfg


def _pack_run_target_attrs(*xs: str) -> list[str]:
    raw = [x.strip() for x in xs if (x or "").strip()]
    try:
        from attribute_detector import canonicalize_target_attribute_lines

        return canonicalize_target_attribute_lines(raw)
    except Exception:
        return raw


def _coerce_run_limit(limit) -> int:
    """Поле «Макс. офферов»: 0 = без лимита. Gradio может отдать float/None."""
    if limit is None:
        return 0
    try:
        return max(0, int(float(limit)))
    except (TypeError, ValueError):
        return 0


def _format_categories_for_log(cats: list[str] | None) -> str:
    """Кратко для лога: последний сегмент пути вместо полной иерархии маркетплейса."""
    if not cats:
        return "весь фид (не выбрано)"
    short: list[str] = []
    for c in cats:
        s = (c or "").strip()
        if not s:
            continue
        short.append(s.rsplit(" / ", 1)[-1].strip() if " / " in s else s)
    return ", ".join(short) if short else "весь фид (не выбрано)"


def _run_prompt_summary_for_log(cfg: dict) -> list[str]:
    """Строки лога: задание с «Запуск» и ключи JSON — тот же расчёт, что в analyze_offer."""
    lines: list[str] = []
    ti = (cfg.get("task_instruction") or "").strip()
    if ti:
        t = ti.replace("\n", " ").strip()
        lines.append(f"Задание: {t[:220]}{'…' if len(t) > 220 else ''}")
    else:
        lines.append("Задание: _(пусто — вкладка «Запуск»)_")
    try:
        from attribute_detector import prepare_visual_analysis_plan, resolve_attributes_for_prompt

        plan = prepare_visual_analysis_plan(cfg)
        tlist = plan["task_target_list"]
        if tlist:
            lines.append(f"Атрибуты (после автодобавления ключа): {tlist!r}")
        vertical = plan["vertical"]
        use_project_task = bool(ti and vertical != "Одежда")
        fp = plan["full_prompt_text"] if plan["use_full_prompt"] else ""
        dir_ids = [d.get("id", "?") for d in plan["directions"]]
        if dir_ids:
            lines.append(f"Направления в прогоне: {', '.join(dir_ids)}")
        all_keys: list[str] = []
        for d in plan["directions"]:
            attrs = d.get("attributes") or []
            custom = (d.get("custom_prompt") or "").strip() or (ti if use_project_task else "")
            ra = resolve_attributes_for_prompt(attrs, tlist, custom, fp)
            for a in ra:
                k = a.get("key")
                if k:
                    all_keys.append(k)
        if all_keys:
            lines.append(f"Ключи JSON (как в analyze_offer): **{sorted(set(all_keys))}**")
        elif ti or tlist or fp:
            lines.append(
                "⚠ Ключи JSON не определены — уточните атрибут (например «цвет металла») или `(metal_color)`."
            )
    except Exception:
        pass
    if cfg.get("use_full_prompt_edit") and (cfg.get("full_prompt_text") or "").strip():
        lines.append("Режим: полный промпт вручную (`full_prompt_text`).")
    return lines


def _fingerprint_json_equal(a: dict, b: dict) -> bool:
    return json.dumps(a or {}, sort_keys=True) == json.dumps(b or {}, sort_keys=True)


def _resolve_offers_for_run(
    project_name: str,
    db,
    cats: list[str] | None,
    limit_n: int,
    all_feed: bool,
    force_reprocess: bool,
    rdb: Path,
) -> tuple[list[dict], int, bool, str | None, dict]:
    """
    Если есть pending_run.json с тем же fingerprint — продолжить ту же выборку.
    Иначе — обычный подбор (и сброс устаревшего pending).
    Возвращает также fingerprint батча (для pending_run.json).
    """
    pc = pm.load_project(project_name) or {}
    fp = pm.run_batch_fingerprint(
        project_name,
        cats,
        limit_n,
        all_feed,
        force_reprocess,
        dedupe_mode_from_config(pc),
        str(pc.get("inscription_mode") or ""),
    )
    pending = pm.load_pending_run(project_name)
    pfp = pending.get("fingerprint") if pending else None
    if pending and pfp and _fingerprint_json_equal(fp, pfp):
        ids = pending.get("offer_ids") or []
        if ids:
            offers: list[dict] = []
            missing = 0
            for oid in ids:
                o = fc.get_offer_by_id(db, str(oid))
                if o:
                    offers.append(o)
                else:
                    missing += 1
            if offers:
                extra = f" (в кэше не найдено offer_id: {missing})" if missing else ""
                return (
                    offers,
                    0,
                    True,
                    f"▶ Продолжение прерванного прогона: в очереди **{len(offers)}** офферов{extra}.",
                    fp,
                )
            extra = f" (не найдено в кэше: {missing})" if missing else ""
            return (
                [],
                0,
                True,
                f"⚠ В **pending_run.json** — **{len(ids)}** offer_id, в кэше фида не найден ни один оффер{extra}. "
                "Загрузите фид в этот проект или проверьте кэш; очередь **не сброшена**.",
                fp,
            )
    if pending:
        pm.clear_pending_run(project_name)
    existing_ids: set[str] = set() if force_reprocess else _get_processed_offer_ids(rdb)
    paf = [a for a in (pc.get("picture_attr_filter") or []) if a.strip()] or None
    pif = [int(x) for x in (pc.get("picture_index_filter") or []) if str(x).isdigit()] or None
    offers, skipped = _collect_offers_to_process(db, cats, limit_n, existing_ids, force_reprocess, picture_attr_filter=paf, picture_index_filter=pif)
    return offers, skipped, False, None, fp


def _collect_offers_to_process(
    db,
    cats: list[str] | None,
    limit_n: int,
    existing_ids: set,
    force_reprocess: bool,
    picture_attr_filter: list[str] | None = None,
    picture_index_filter: list[int] | None = None,
) -> tuple[list[dict], int]:
    """
    Подбор офферов к обработке. Если limit_n > 0 и часть первых N уже в результатах —
    листаем кэш дальше (offset), пока не наберётся нужное число необработанных.
    Возвращает (список офферов, число пропущенных как «уже в результатах» при обходе).
    picture_attr_filter: если задан — picture_urls будет содержать только URL из этих тегов.
    picture_index_filter: если задан — берём только картинки с указанными 1-based индексами.
    """
    paf = picture_attr_filter or None
    pif = picture_index_filter or None
    if force_reprocess:
        if limit_n > 0:
            return fc.get_offers(db, cats, limit=limit_n, picture_attr_filter=paf, picture_index_filter=pif), 0
        return fc.get_offers(db, cats, limit=0, picture_attr_filter=paf, picture_index_filter=pif), 0

    skipped = 0
    if limit_n <= 0:
        all_o = fc.get_offers(db, cats, limit=0, picture_attr_filter=paf, picture_index_filter=pif)
        out = [o for o in all_o if o.get("offer_id") not in existing_ids]
        skipped = len(all_o) - len(out)
        return out, skipped

    chunk = max(limit_n * 5, 200)
    offset = 0
    collected: list[dict] = []
    while len(collected) < limit_n:
        batch = fc.get_offers(db, cats, limit=chunk, offset=offset, picture_attr_filter=paf, picture_index_filter=pif)
        if not batch:
            break
        for o in batch:
            oid = o.get("offer_id")
            if oid in existing_ids:
                skipped += 1
                continue
            collected.append(o)
            if len(collected) >= limit_n:
                break
        offset += len(batch)
        if len(batch) < chunk:
            break
    return collected[:limit_n], skipped


def _clone_analyze_result_for_offer(base: dict, offer: dict) -> dict:
    out = json.loads(json.dumps(base, ensure_ascii=False))
    pics = offer.get("picture_urls") or []
    out["offer_id"] = offer.get("offer_id", "")
    out["name"] = offer.get("name", "")
    out["category"] = offer.get("category", "")
    out["picture_url"] = (pics[0] or "") if pics else ""
    return out


def _warmup_run_models(config_with_cache: dict) -> None:
    """Прогрев основной модели и при отдельном вызове надписей — второй модели, если отличается."""
    warmup_ollama_model(config_with_cache, timeout=90)
    if not bool(config_with_cache.get("extract_inscriptions", True)):
        return
    if inscription_mode_is_same_prompt(config_with_cache):
        return
    main_m = (config_with_cache.get("model") or "").strip()
    tm = resolve_inscription_model(config_with_cache, main_m)
    if tm and tm != main_m:
        warmup_ollama_model({**config_with_cache, "model": tm}, timeout=90)


def get_finetune_dashboard_markdown() -> str:
    """Расширенная сводка для вкладки «Дообучение»: цифры, файлы, пошаговый сценарий."""
    name = _proj_name()
    if not name:
        return "### Дообучение\n\n_Проект не выбран. Откройте вкладку **«Проекты»** и выберите или создайте проект._"
    corrs = pm.load_corrections(name)
    n_corr = len(corrs)
    n_queue = len(pm.load_finetune_queue_offer_ids(name))
    ds_dir = pm.project_dir(name) / "fine_tune_dataset"
    train_path = ds_dir / "train.jsonl"
    ev_path = ds_dir / "eval_anchors.jsonl"
    low_path = ds_dir / "review_low_confidence.json"
    n_train = n_ev = 0
    if train_path.exists():
        try:
            with open(train_path, encoding="utf-8") as f:
                n_train = sum(1 for _ in f if _.strip())
        except OSError:
            pass
    if ev_path.exists():
        try:
            with open(ev_path, encoding="utf-8") as fe:
                n_ev = sum(1 for _ in fe if _.strip())
        except OSError:
            pass
    n_results = 0
    rdb = pm.results_db_path(name)
    if rdb.exists():
        try:
            con = sqlite3.connect(str(rdb))
            try:
                row = con.execute("SELECT COUNT(*) FROM results").fetchone()
                n_results = int(row[0]) if row else 0
            finally:
                con.close()
        except Exception:
            pass
    bullets = [
        f"- **Правок вручную** (`projects/{name}/corrections.json`): **{n_corr}** — правки с вкладки **«Результаты»** (в т.ч. снимок «до правки» для пар «было→стало»).",
        f"- **Ручная очередь дообучения** (кнопка на «Результаты»): **{n_queue}** offer_id — подмешиваются при сборке, если включена соответствующая галка.",
        f"- **Строк в `fine_tune_dataset/train.jsonl`:** **{n_train}** — после кнопки **«Собрать датасет»**.",
        f"- **Якорный eval** (`eval_anchors.jsonl`): **{n_ev}** — для блока **«Оценка до/после»**.",
        f"- **Карточек в БД результатов** (`results.db`): **{n_results}** — источник **авто-примеров** (высокая уверенность) и **пула низкой уверенности**.",
    ]
    if low_path.exists():
        bullets.append(
            f"- **Пул низкой уверенности** (`review_low_confidence.json`) — список офферов для ручной проверки; после разметки снова **«Собрать датасет»**."
        )
    steps = (
        "\n#### Как дообучить модель на своих данных\n"
        "1. **«Результаты»** → карточка товара → **Правка атрибутов** → сохранить → запись в `corrections.json`.\n"
        "2. Ниже **«Собрать датасет»** — формируется `train.jsonl` (правки + опционально уверенные строки из БД + очередь).\n"
        "3. **«Запустить LoRA обучение»** — папка адаптера в проекте.\n"
        "4. **«Экспорт в Ollama»** — имя модели (например `clothes-detector-v2`).\n"
        "5. **Глобальные настройки** → выбрать эту модель для распознавания.\n"
    )
    return "### Сводка по дообучению\n" + "\n".join(bullets) + steps


def get_finetune_step1_info() -> str:
    """Текст верхнего блока вкладки «Дообучение» (расширенная сводка)."""
    return get_finetune_dashboard_markdown()


def _run_state_path() -> Path | None:
    n = _proj_name()
    return pm.run_state_path(n) if n else None


def _load_run_state() -> dict:
    path = _run_state_path()
    if not path or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_run_state(state: dict):
    path = _run_state_path()
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=0)
    except Exception:
        pass


# ── Results DB helpers ────────────────────────────────────────────────────────

RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    offer_id        TEXT PRIMARY KEY,
    name            TEXT,
    category        TEXT,
    picture_url     TEXT,
    avg_confidence  INTEGER,
    text_json       TEXT,
    attributes_json TEXT,
    error           TEXT,
    run_ts          REAL,
    model_name      TEXT DEFAULT ''
);
"""


def _migrate_results_db(con: sqlite3.Connection) -> None:
    """Добавляет недостающие колонки в старые БД результатов."""
    rows = con.execute("PRAGMA table_info(results)").fetchall()
    cols = {r[1] for r in rows}
    if "model_name" not in cols:
        con.execute("ALTER TABLE results ADD COLUMN model_name TEXT DEFAULT ''")


def _init_results_db(db_path: Path):
    con = fc.sqlite_connect(db_path)
    con.executescript(RESULTS_SCHEMA)
    _migrate_results_db(con)
    con.commit()
    con.close()


def _save_result(db_path: Path, result: dict):
    _init_results_db(db_path)
    con = fc.sqlite_connect(db_path)
    _migrate_results_db(con)
    model_saved = (result.get("model") or result.get("model_name") or "").strip()
    con.execute(
        "INSERT OR REPLACE INTO results "
        "(offer_id, name, category, picture_url, avg_confidence, text_json, attributes_json, error, run_ts, model_name) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            result.get("offer_id", ""),
            result.get("name", ""),
            result.get("category", ""),
            result.get("picture_url", ""),
            result.get("avg_confidence", 0),
            json.dumps(result.get("text_detection", {}), ensure_ascii=False),
            json.dumps(result.get("direction_attributes", {}), ensure_ascii=False),
            result.get("error", ""),
            time.time(),
            model_saved,
        ),
    )
    con.commit()
    con.close()


def _load_results(db_path: Path, min_conf: int = 0, max_conf: int = 100, category: str = "") -> list[dict]:
    if not db_path or not db_path.exists():
        return []
    con = fc.sqlite_connect(db_path)
    _migrate_results_db(con)
    con.commit()
    con.row_factory = sqlite3.Row
    q = "SELECT * FROM results WHERE avg_confidence >= ? AND avg_confidence <= ?"
    params: list = [min_conf, max_conf]
    if category and category != "Все":
        q += " AND category = ?"
        params.append(category)
    q += " ORDER BY run_ts DESC"
    rows = con.execute(q, params).fetchall()
    con.close()
    out = []
    for row in rows:
        d = dict(row)
        d["text_detection"] = json.loads(d.get("text_json") or "{}")
        raw_attrs = json.loads(d.get("attributes_json") or "{}")
        # Backward compat: flat attributes -> one direction "clothing"
        if raw_attrs and not ("clothing" in raw_attrs or "other" in raw_attrs):
            raw_attrs = {"clothing": raw_attrs}
        d["direction_attributes"] = raw_attrs
        d["model"] = (d.get("model_name") or "").strip()
        out.append(d)
    return out


def _load_result_by_offer_id(db_path: Path, offer_id: str) -> dict | None:
    if not db_path or not db_path.exists() or not offer_id:
        return None
    con = fc.sqlite_connect(db_path)
    _migrate_results_db(con)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM results WHERE offer_id = ?", (str(offer_id),)).fetchone()
    con.close()
    if not row:
        return None
    d = dict(row)
    d["text_detection"] = json.loads(d.get("text_json") or "{}")
    raw_attrs = json.loads(d.get("attributes_json") or "{}")
    if raw_attrs and not ("clothing" in raw_attrs or "other" in raw_attrs):
        raw_attrs = {"clothing": raw_attrs}
    d["direction_attributes"] = raw_attrs
    d["model"] = (d.get("model_name") or "").strip()
    return d


def _recompute_avg_confidence(merged: dict) -> int:
    scores: list[int] = []
    td = merged.get("text_detection") or {}
    if isinstance(td, dict) and not td.get("error") and td.get("confidence") is not None:
        scores.append(int(td["confidence"]))
    for dr in (merged.get("direction_attributes") or {}).values():
        if not isinstance(dr, dict) or dr.get("error"):
            continue
        for k, v in dr.items():
            if k == "error" or not isinstance(v, dict):
                continue
            if "confidence" in v:
                scores.append(int(v["confidence"]))
    return int(sum(scores) / len(scores)) if scores else 0


def _direction_id_for_attr_key(config: dict, attr_key: str) -> str:
    """Направление (id), в котором объявлен атрибут с данным key; иначе clothing."""
    ak = (attr_key or "").strip()
    if not ak:
        return "clothing"
    for d in config.get("directions") or []:
        did = (d.get("id") or "").strip() or "clothing"
        for a in d.get("attributes") or []:
            if (a.get("key") or "").strip() == ak:
                return did
    return "clothing"


def _offer_ids_sharing_normalized_picture(db_path: Path, picture_url: str) -> list[str]:
    """
    Все offer_id в results с тем же нормализованным URL первой картинки, что и у дедупа по URL в прогоне.
    Разный query (?w=) у одного файла — считается одной картинкой.
    Если URL пустой или не нормализуется — сравнение по точной строке picture_url.
    """
    _init_results_db(db_path)
    con = fc.sqlite_connect(db_path)
    rows = con.execute("SELECT offer_id, picture_url FROM results").fetchall()
    con.close()
    key = normalize_picture_url(picture_url or "")
    out: list[str] = []
    if key:
        for oid, pu in rows:
            if normalize_picture_url(pu or "") == key:
                out.append(str(oid))
        return out
    raw = (picture_url or "").strip()
    if not raw:
        return []
    for oid, pu in rows:
        if (pu or "").strip() == raw:
            out.append(str(oid))
    return out


def _apply_correction_to_stored_result(
    base: dict,
    corrected_attrs: dict[str, str],
    text_val: str | None,
    config: dict,
    glossary: dict,
) -> dict:
    """
    Сливает правку (ключи/значения EN как после normalize_correction_attrs) в сохранённую строку результатов.
    text_val — как в форме (строка с запятыми); None = не менять text_detection.
    """
    out = copy.deepcopy(base)
    da = copy.deepcopy(out.get("direction_attributes") or {})
    if not isinstance(da, dict):
        da = {}
    for attr_key, en_val in (corrected_attrs or {}).items():
        if not isinstance(en_val, str) or not str(en_val).strip():
            continue
        did = _direction_id_for_attr_key(config, str(attr_key))
        if did not in da or not isinstance(da.get(did), dict):
            da[did] = {}
        dest = da[did]
        dest.pop("error", None)
        dest[str(attr_key)] = {"value": str(en_val).strip(), "confidence": 95}
    if corrected_attrs:
        pm.translate_direction_attribute_values_inplace(da, glossary)
        pm.strip_placeholder_attribute_values_inplace(da)
    out["direction_attributes"] = da
    if text_val is not None:
        ts = (text_val or "").strip()
        if ts:
            parts = [s.strip() for s in ts.split(",") if s.strip()]
            out["text_detection"] = {
                "texts": parts if len(parts) > 1 else ([ts] if ts else []),
                "text_found": True,
                "confidence": 95,
            }
        else:
            out["text_detection"] = {"texts": [], "text_found": False, "confidence": 0}
    out["avg_confidence"] = _recompute_avg_confidence(out)
    out["error"] = ""
    return out


def _merge_partial_reanalyze(existing: dict, partial: dict, keys: list[str]) -> dict:
    """Сливает частичный ответ analyze_offer в сохранённую строку результатов."""
    out = copy.deepcopy(existing)
    ks = frozenset(str(x).strip() for x in keys if x and str(x).strip())
    if "__text__" in ks:
        out["text_detection"] = copy.deepcopy(partial.get("text_detection") or {})
    attr_only = {x for x in ks if x != "__text__"}
    for did, attrs in (partial.get("direction_attributes") or {}).items():
        if not isinstance(attrs, dict):
            continue
        if did not in out.get("direction_attributes") or not isinstance(out["direction_attributes"].get(did), dict):
            out.setdefault("direction_attributes", {})[did] = {}
        dest = out["direction_attributes"][did]
        for ak, val in attrs.items():
            if ak == "error":
                continue
            if ak in attr_only and isinstance(val, dict):
                dest[ak] = val
    out["avg_confidence"] = _recompute_avg_confidence(out)
    m = (partial.get("model") or partial.get("model_name") or "").strip()
    if m:
        out["model"] = m
    pe = partial.get("error")
    if pe:
        out["error"] = str(pe)[:500]
    return out


def _text_detection_has_inscription(td: dict) -> bool:
    """True только если в результате реально есть надпись (не только число confidence от модели)."""
    if not isinstance(td, dict) or td.get("error"):
        return False
    txs = td.get("texts")
    if isinstance(txs, list) and any(str(t).strip() for t in txs):
        return True
    if td.get("text_found") is True:
        return True
    if str(td.get("text") or "").strip():
        return True
    return False


def _text_detection_joined_for_export(td: dict | None) -> str | None:
    """
    Текст надписей для CSV — как на карточке результата (без перевода через глоссарий).
    """
    if not isinstance(td, dict) or td.get("error"):
        return None
    parts: list[str] = []
    txs = td.get("texts")
    if isinstance(txs, list):
        for t in txs:
            s = str(t).strip() if t is not None else ""
            if s:
                parts.append(s)
    if not parts:
        s = str(td.get("text") or "").strip()
        if s:
            parts.append(s)
    if not parts:
        return None
    return ", ".join(parts)


def _confidence_for_attribute(result: dict, attr_key: str) -> int | None:
    if not attr_key or attr_key == "__avg__":
        return None
    if attr_key == "__text__":
        td = result.get("text_detection") or {}
        if not isinstance(td, dict) or not _text_detection_has_inscription(td):
            return None
        if td.get("confidence") is not None:
            return int(td["confidence"])
        return 0
    for _d, attrs in (result.get("direction_attributes") or {}).items():
        if not isinstance(attrs, dict):
            continue
        ent = attrs.get(attr_key)
        if not isinstance(ent, dict):
            continue
        if ent.get("confidence") is not None:
            return int(ent["confidence"])
        if (ent.get("value") or "").strip():
            return 0
    return None


def _filter_results_list(
    results: list[dict],
    attr_key: str | None,
    min_c: int,
    max_c: int,
) -> list[dict]:
    """Фильтр по средней уверенности карточки или по уверенности одного атрибута / надписей."""
    if not attr_key or attr_key == "__avg__":
        return [r for r in results if min_c <= int(r.get("avg_confidence") or 0) <= max_c]
    out: list[dict] = []
    for r in results:
        c = _confidence_for_attribute(r, attr_key)
        if c is None:
            continue
        if min_c <= c <= max_c:
            out.append(r)
    return out


# Плейсхолдер в Dropdown: не сужать по значению атрибута
_RESULT_ATTR_VALUE_ANY_LABEL = "— любое значение —"


def _get_attr_raw_value(result: dict, attr_key: str) -> str:
    """Первое непустое value по ключу среди направлений."""
    if not attr_key or attr_key in ("__avg__", "__text__"):
        return ""
    for _d, attrs in (result.get("direction_attributes") or {}).items():
        if not isinstance(attrs, dict):
            continue
        ent = attrs.get(attr_key)
        if not isinstance(ent, dict):
            continue
        v = ent.get("value")
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _normalize_attr_value_filter_pick(pick: str | None) -> str:
    s = (pick or "").strip()
    if not s or s == _RESULT_ATTR_VALUE_ANY_LABEL:
        return ""
    if s.startswith("—") and "любое" in s.lower():
        return ""
    return s


def _distinct_values_for_attr_in_results(
    results: list[dict],
    attr_key: str,
    glossary: dict,
    max_opts: int = 500,
) -> list[str]:
    """Уникальные отображаемые значения (RU через глоссарий) для выпадающего списка."""
    opts = [_RESULT_ATTR_VALUE_ANY_LABEL]
    if attr_key in ("__avg__", "", None):
        return opts
    if attr_key == "__text__":
        return opts
    seen: dict[str, str] = {}
    for r in results:
        raw_v = _get_attr_raw_value(r, attr_key)
        if not raw_v:
            continue
        disp = pm.translate_attribute_value(raw_v, glossary).strip()
        if disp:
            seen.setdefault(disp.casefold(), disp)
        if len(seen) >= max_opts:
            break
    rest = sorted(seen.values(), key=lambda x: x.casefold())
    return opts + rest


def _result_matches_attr_value_needle(result: dict, attr_key: str, needle: str, glossary: dict) -> bool:
    """Совпадение по подстроке без учёта регистра: сырое EN и переведённое RU значение."""
    n = (needle or "").strip().casefold()
    if not n:
        return True
    if attr_key == "__text__":
        blob = (_text_detection_joined_for_export(result.get("text_detection")) or "").casefold()
        return n in blob
    raw_v = (_get_attr_raw_value(result, attr_key) or "").strip()
    if not raw_v:
        return False
    ru = pm.translate_attribute_value(raw_v, glossary).strip().casefold()
    raw_cf = raw_v.casefold()
    return n in raw_cf or n in ru or raw_cf == n or ru == n


def _results_tab_filtered_list(
    rdb: Path,
    proj: dict,
    min_c: int,
    max_c: int,
    category: str,
    hide_dup: bool,
    filter_attr_val: str | None,
    attr_value_pick: str | None = None,
    model_pick: str | None = None,
) -> tuple[list[dict], str, int, list[str], int, list[str], list[str]]:
    """
    Тот же список записей, что показывает вкладка «Результаты» после «Обновить».
    Возвращает (results, fkey, loaded_hint, value_dropdown_choices, n_before_value_filter,
    filter_attr_values, filter_attr_labels).
    """
    glossary = pm.load_attribute_glossary()
    cats = _results_categories(rdb)
    cat = (category or "Все").strip()
    if cat not in cats:
        cat = "Все"
    all_rows = _load_results(rdb, 0, 100, cat)
    mp = (model_pick or "").strip()
    if mp and mp != "Все":
        all_rows = [r for r in all_rows if _result_matches_model_filter(r, mp)]
    db_keys = _collect_attr_keys_from_results(all_rows)
    fval_list, flab_list = _merge_result_filter_attr_choices(proj, db_keys)
    fkey = _resolve_result_filter_attr_key(filter_attr_val, fval_list, flab_list)
    if fkey not in fval_list and fval_list:
        fkey = fval_list[0]

    if fkey == "__avg__":
        results_pre = [
            r for r in all_rows if min_c <= int(r.get("avg_confidence") or 0) <= max_c
        ]
        loaded_hint = len(results_pre)
        if hide_dup:
            results_pre = _dedupe_results_by_image_url(results_pre)
    else:
        loaded_hint = len(all_rows)
        results_pre = _filter_results_list(all_rows, fkey, min_c, max_c)
        if hide_dup:
            results_pre = _dedupe_results_by_image_url(results_pre)

    value_choices = _distinct_values_for_attr_in_results(results_pre, fkey, glossary)
    needle = _normalize_attr_value_filter_pick(attr_value_pick)

    n_pre = len(results_pre)
    if not needle or fkey == "__avg__":
        results = results_pre
    else:
        results = [
            r
            for r in results_pre
            if _result_matches_attr_value_needle(r, fkey, needle, glossary)
        ]

    return results, fkey, loaded_hint, value_choices, n_pre, fval_list, flab_list


def _results_attr_value_dropdown_update(
    attr_value_pick: str | None,
    value_choices: list[str],
    fkey: str,
):
    """Обновление выпадающего списка значений; при «средняя уверенность» поле отключено."""
    vc = list(value_choices) if value_choices else [_RESULT_ATTR_VALUE_ANY_LABEL]
    if fkey == "__avg__":
        return gr.update(
            choices=vc,
            value=_RESULT_ATTR_VALUE_ANY_LABEL,
            interactive=False,
        )
    av = (attr_value_pick or "").strip()
    if not av or av == _RESULT_ATTR_VALUE_ANY_LABEL:
        return gr.update(choices=vc, value=_RESULT_ATTR_VALUE_ANY_LABEL, interactive=True)
    if av in vc:
        return gr.update(choices=vc, value=av, interactive=True)
    return gr.update(choices=vc, value=av, interactive=True)


def _safe_results_page_index(page_val) -> int:
    try:
        return max(1, int(float(page_val)))
    except (TypeError, ValueError):
        return 1


def _selection_covers_all_on_page(sel: list, page_slice: list[dict]) -> bool:
    """True, если все offer_id на текущей странице есть в sel (для общей галочки)."""
    oids = [str(r.get("offer_id", "")).strip() for r in page_slice if r.get("offer_id")]
    if not oids:
        return False
    sset = {str(x).strip() for x in (sel or []) if x}
    return all(o in sset for o in oids)


def _canonical_inscription_mode_stored(raw) -> str:
    """Нормализация после старого бага: в config могла попасть русская подпись Radio вместо ключа."""
    s = str(raw or "separate_call").strip().lower()
    if s in ("separate_call", "same_prompt"):
        return s
    if "тот же" in s or "один запрос" in s:
        return "same_prompt"
    return "separate_call"


def _result_filter_attr_choices(config: dict) -> tuple[list[str], list[str]]:
    """(values, labels): values — ключи (__avg__, __text__, key атрибута); labels — подписи в UI."""
    values: list[str] = ["__avg__"]
    labels: list[str] = ["— По средней уверенности карточки —"]
    seen: set[str] = set()
    for d in config.get("directions", []):
        for a in d.get("attributes", []):
            key = (a.get("key") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            lab = (a.get("label") or key).strip()
            values.append(key)
            labels.append(f"{lab} ({key})")
    values.append("__text__")
    labels.append("Надписи на фото (text_detection)")
    return values, labels


def _collect_attr_keys_from_results(rows: list[dict]) -> list[str]:
    """Ключи атрибутов, реально присутствующие в сохранённых direction_attributes (по всем направлениям)."""
    seen: set[str] = set()
    for r in rows:
        for _did, attrs in (r.get("direction_attributes") or {}).items():
            if not isinstance(attrs, dict) or attrs.get("error"):
                continue
            for k, ent in attrs.items():
                if k == "error" or not isinstance(ent, dict):
                    continue
                seen.add(k)
    return sorted(seen, key=lambda x: x.casefold())


def _merge_result_filter_attr_choices(proj: dict, db_keys: list[str]) -> tuple[list[str], list[str]]:
    """
    Базовый список из конфига проекта + атрибуты, которые есть в результатах, но не прописаны в directions
    (например metal_color у ювелирки при шаблоне одежды).
    """
    base_v, base_l = _result_filter_attr_choices(proj)
    if len(base_v) < 2 or base_v[-1] != "__text__":
        return base_v, base_l
    text_v = base_v.pop()
    text_l = base_l.pop()
    in_list = set(base_v)
    key_to_label: dict[str, str] = {}
    for a in pm.get_all_attribute_definitions(proj):
        k = (a.get("key") or "").strip()
        if k:
            key_to_label[k] = (a.get("label") or k).strip()
    for k in db_keys:
        if not k or k in in_list:
            continue
        in_list.add(k)
        lab = key_to_label.get(k) or k.replace("_", " ")
        base_v.append(k)
        base_l.append(f"{lab} ({k})")
    base_v.append(text_v)
    base_l.append(text_l)
    return base_v, base_l


def _resolve_result_filter_attr_key(
    selected: str | None,
    values: list[str],
    labels: list[str],
) -> str:
    """Из значения Dropdown (подпись или ключ) получить канонический ключ для фильтра."""
    s = (selected or "").strip()
    if not s:
        return "__avg__" if "__avg__" in values else (values[0] if values else "__avg__")
    if s in values:
        return s
    for i, lab in enumerate(labels):
        if lab == s and i < len(values):
            return values[i]
    if "(" in s and s.rstrip().endswith(")"):
        inner = s.rsplit("(", 1)[-1][:-1].strip()
        if inner in values:
            return inner
    return "__avg__" if "__avg__" in values else (values[0] if values else "__avg__")


def _result_filter_label_for_key(key: str, values: list[str], labels: list[str]) -> str:
    try:
        i = values.index(key)
        if i < len(labels):
            return labels[i]
    except ValueError:
        pass
    return labels[0] if labels else key


def _dedupe_results_by_image_url(results: list[dict]) -> list[dict]:
    """
    Один ряд на уникальный URL картинки (разные размеры SKU с одним фото).
    Оставляем запись с большей уверенностью, при равенстве — более свежий run_ts.
    Порядок — как первое появление группы в исходном списке.
    """
    if not results:
        return results
    groups: dict[str, list[dict]] = defaultdict(list)
    first_pos: dict[str, int] = {}
    for i, r in enumerate(results):
        url = (r.get("picture_url") or "").strip()
        key = normalize_picture_url(url) if url else f"__nopic__:{r.get('offer_id', '')}"
        groups[key].append(r)
        if key not in first_pos:
            first_pos[key] = i

    def row_score(r: dict) -> tuple:
        return (int(r.get("avg_confidence") or 0), float(r.get("run_ts") or 0))

    picks: list[tuple[int, dict]] = []
    for key, grp in groups.items():
        best = max(grp, key=row_score)
        picks.append((first_pos[key], best))
    picks.sort(key=lambda x: x[0])
    return [p[1] for p in picks]


def _results_categories(db_path: Path) -> list[str]:
    if not db_path or not db_path.exists():
        return []
    con = fc.sqlite_connect(db_path)
    rows = con.execute("SELECT DISTINCT category FROM results ORDER BY category").fetchall()
    con.close()
    return ["Все"] + [r[0] for r in rows if r[0]]


# Подпись в выпадающем списке «Результаты»: строки без model_name в БД
_RESULT_MODEL_UNKNOWN_LABEL = "(не указано)"


def _results_model_choices(db_path: Path, category: str) -> list[str]:
    """Список значений для фильтра по полю model_name (первый пункт — «Все»)."""
    if not db_path or not db_path.exists():
        return ["Все"]
    con = fc.sqlite_connect(db_path)
    _migrate_results_db(con)
    cat = (category or "Все").strip()
    if cat and cat != "Все":
        rows = con.execute(
            "SELECT DISTINCT TRIM(COALESCE(model_name, '')) AS m FROM results WHERE category = ?",
            (cat,),
        ).fetchall()
    else:
        rows = con.execute("SELECT DISTINCT TRIM(COALESCE(model_name, '')) AS m FROM results").fetchall()
    con.close()
    values = {(r[0] or "").strip() for r in rows}
    has_empty = "" in values
    non_empty = sorted([v for v in values if v], key=lambda x: x.casefold())
    out = ["Все"]
    if has_empty:
        out.append(_RESULT_MODEL_UNKNOWN_LABEL)
    out.extend(non_empty)
    return out


def _result_matches_model_filter(result: dict, model_pick: str | None) -> bool:
    pick = (model_pick or "").strip()
    if not pick or pick == "Все":
        return True
    m = (result.get("model") or result.get("model_name") or "").strip()
    if pick == _RESULT_MODEL_UNKNOWN_LABEL:
        return not m
    return m == pick


def _delete_result_by_offer_id(db_path: Path, offer_id: str) -> bool:
    """Удаляет одну запись из results по offer_id. Возвращает True если удалено."""
    if not db_path or not db_path.exists() or not offer_id:
        return False
    try:
        con = fc.sqlite_connect(db_path)
        cur = con.execute("DELETE FROM results WHERE offer_id = ?", (offer_id,))
        deleted = cur.rowcount > 0
        con.commit()
        con.close()
        return deleted
    except Exception:
        return False


def _normalize_offer_id_list(raw) -> list[str]:
    """Стабильный список offer_id для State и сравнения с БД."""
    out: list[str] = []
    seen: set[str] = set()
    for x in raw or []:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _prune_selection_to_visible_results(results: list[dict], sel: list[str]) -> list[str]:
    """Оставить в выборе только id, которые есть в текущем отфильтрованном списке."""
    allowed: set[str] = set()
    for r in results:
        oid = r.get("offer_id")
        if oid is None:
            continue
        s = str(oid).strip()
        if s:
            allowed.add(s)
    return [x for x in sel if x in allowed]


def _delete_results_batch(db_path: Path, offer_ids: list[str]) -> int:
    """Удаляет много строк за несколько запросов (лимит переменных SQLite)."""
    if not db_path or not db_path.exists():
        return 0
    cleaned = _normalize_offer_id_list(offer_ids)
    if not cleaned:
        return 0
    con = fc.sqlite_connect(db_path)
    deleted = 0
    chunk = 400
    try:
        for i in range(0, len(cleaned), chunk):
            part = cleaned[i : i + chunk]
            ph = ",".join("?" * len(part))
            cur = con.execute(f"DELETE FROM results WHERE offer_id IN ({ph})", part)
            deleted += cur.rowcount or 0
        con.commit()
    finally:
        con.close()
    return deleted


def _merge_offer_ids_into_queue(queue: list, new_ids: list[str]) -> list[str]:
    q = _normalize_offer_id_list(queue)
    merged = list(q)
    have = set(merged)
    for nid in new_ids:
        n = str(nid).strip()
        if n and n not in have:
            merged.append(n)
            have.add(n)
    return merged


def _rr_queue_summary_markdown(ids: list[str]) -> str:
    """Краткая сводка очереди повторного прогона без тысяч пунктов в multiselect."""
    n = len(ids)
    if n == 0:
        return (
            "**Очередь повторной обработки:** пусто.\n\n"
            "Добавляйте офферы кнопками **«В список: …»** в блоке выше или "
            "**«В очередь: отмеченные галочками»** под карточками. "
            "Список хранится в памяти сессии, без прокрутки тысяч строк."
        )
    preview = ", ".join(ids[:30])
    more = f"\n\n… **ещё {n - min(30, n)}**." if n > 30 else ""
    return f"**В очереди повторной обработки: {n}** offer_id.{more}\n\n`{preview}`"


def _get_processed_offer_ids(db_path: Path) -> set[str]:
    """Возвращает множество offer_id, уже записанных в results (чтобы не обрабатывать повторно)."""
    if not db_path or not db_path.exists():
        return set()
    con = fc.sqlite_connect(db_path)
    rows = con.execute("SELECT offer_id FROM results").fetchall()
    con.close()
    return {r[0] for r in rows if r[0]}


def _key_to_label_map(config: dict) -> dict[str, str]:
    """Из конфига направлений: key -> русский label для экспорта."""
    out = {}
    for d in config.get("directions", []):
        for a in d.get("attributes", []):
            key = a.get("key", "")
            if key:
                out[key] = (a.get("label") or key).strip()
    return out


def _console_result_line(result: dict, config: dict) -> str:
    """Краткая строка атрибутов для вывода в терминал (без надписей)."""
    parts: list[str] = []
    glossary = pm.load_attribute_glossary()
    k2l = _key_to_label_map(config)
    for _dir_id, attrs in (result.get("direction_attributes") or {}).items():
        if not attrs or not isinstance(attrs, dict):
            continue
        for k, v in attrs.items():
            if k == "error" or not isinstance(v, dict):
                continue
            parts.append(
                f'{k2l.get(k, k)}={pm.translate_attribute_value(v.get("value", "—"), glossary)}'
            )
    s = " | ".join(parts)
    if len(s) > 220:
        return s[:217] + "…"
    return s or "—"


def _profile_run_log_lines(result: dict) -> list[str]:
    """Строки для лога UI при IMAGE_DESC_PROFILE=1 (поле _profile в результате analyze_offer)."""
    prof = result.get("_profile")
    if not prof:
        return []
    lines = [
        "      Профиль (IMAGE_DESC_PROFILE): "
        f"total_wall_ms={prof.get('total_wall_ms')} "
        f"image_prep_ms={prof.get('image_prep_ms')} "
        f"sum_vision_ms={prof.get('vision_calls_sum_ms')} "
        f"workers={prof.get('max_parallel_vision')}",
    ]
    for c in prof.get("vision_calls") or []:
        if not isinstance(c, dict):
            continue
        t = c.get("task", "?")
        b = c.get("backend", "?")
        ms = c.get("ms", "?")
        suf = ""
        if c.get("failed"):
            suf = " failed"
        elif c.get("note"):
            suf = f" ({c.get('note')})"
        lines.append(f"         {t} [{b}]: {ms} ms{suf}")
    return lines


def _run_single_group_batch(
    group: list,
    *,
    total: int,
    rdb: Path,
    cache_dir: str,
    max_size: int,
    dedupe_infer: bool,
    config_with_cache: dict,
    stop_event: threading.Event,
    next_offer_display_num,
) -> tuple[list[str], int, int, set[str], str | None]:
    """
    Одна группа дедупа: vision по представителю, сохранение по всем SKU.
    Используется при batch_offer_workers > 1; вызывается из пула потоков.
    """
    log_lines: list[str] = []
    processed = 0
    errors = 0
    saved_ids: set[str] = set()
    last_image: str | None = None

    def _emit(offer: dict, n_disp: int, same_pic_copy: bool) -> None:
        oid = offer.get("offer_id", "")
        oname = (offer.get("name") or "")[:50]
        pic_url = (offer.get("picture_urls") or [None])[0]
        log_lines.append(f"[{n_disp}/{total}] Оффер {oid} — {oname}")
        log_lines.append(
            f"      offer_id={oid}  picture_url={ (pic_url or '')[:100] }{'...' if (pic_url or '') and len(pic_url or '') > 100 else ''}"
        )
        if same_pic_copy:
            log_lines.append("      → тот же URL фото — **vision не вызываем**")

    if stop_event.is_set():
        return log_lines, processed, errors, saved_ids, last_image

    rep = group[0]
    rep_id = str(rep.get("offer_id", ""))
    pic_url = (rep.get("picture_urls") or [None])[0]
    if os.environ.get("IMAGE_DESC_POOL_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on"):
        log_lines.append(
            f"      [пул] поток **{threading.current_thread().name}** → vision, представитель группы **{rep_id}**"
        )
    img_path = ensure_image_cached(pic_url, cache_dir, max_size) if pic_url else None
    if img_path:
        log_lines.append(f"      Картинка (представитель группы): путь: {Path(img_path).resolve()}")
        last_image = str(Path(img_path).resolve())
    else:
        log_lines.append("      Картинка: не загружена (представитель группы)")

    try:
        result = analyze_offer(rep, config_with_cache)
        result["model"] = (config_with_cache.get("model") or "").strip()
        for o in group:
            if stop_event.is_set():
                break
            oid = str(o.get("offer_id", ""))
            n_disp = next_offer_display_num()
            same_copy = oid != rep_id and dedupe_infer
            _emit(o, n_disp, same_copy)
            pic_o = (o.get("picture_urls") or [None])[0]
            img_o = ensure_image_cached(pic_o, cache_dir, max_size) if pic_o else None
            if img_o:
                last_image = str(Path(img_o).resolve())
            to_save = _clone_analyze_result_for_offer(result, o) if oid != rep_id else result
            to_save["model"] = result.get("model", "")
            _save_result(rdb, to_save)
            conf = to_save.get("avg_confidence", 0)
            err = to_save.get("error")
            if err:
                errors += 1
                log_lines.append(f"      Анализ: ошибка — {err[:100]}")
                for _pl in _profile_run_log_lines(to_save):
                    log_lines.append(_pl)
                print(f"  [{n_disp}/{total}] {oid}  Ошибка: {err[:60]}")
            else:
                log_lines.append(f"      Анализ: готово. Уверенность: {conf}%")
                for _pl in _profile_run_log_lines(to_save):
                    log_lines.append(_pl)
                _glossary = pm.load_attribute_glossary()
                _key_to_label = _key_to_label_map(config_with_cache)
                parts = []
                for _dir_id, attrs in (to_save.get("direction_attributes") or {}).items():
                    if attrs and isinstance(attrs, dict):
                        short = ", ".join(
                            f"{_key_to_label.get(k, k)}={pm.translate_attribute_value(v.get('value', ''), _glossary)}"
                            for k, v in attrs.items() if k != "error" and isinstance(v, dict)
                        )
                        if short:
                            parts.append(short)
                if parts:
                    log_lines.append("      Результат: " + " | ".join(parts)[:200])
                print(
                    f"  [{n_disp}/{total}] {oid}  OK, уверенность {conf}%  "
                    f"{_console_result_line(to_save, config_with_cache)}"
                )
            saved_ids.add(oid)
            processed += 1
            log_lines.append("")
    except Exception as e:
        for o in group:
            if stop_event.is_set():
                break
            oid = str(o.get("offer_id", ""))
            n_disp = next_offer_display_num()
            _emit(o, n_disp, False)
            err_msg = str(e).strip()
            if len(err_msg) > 100:
                err_msg = err_msg[:97] + "..."
            errors += 1
            log_lines.append(f"      Анализ: ошибка — {err_msg}")
            print(f"  [{n_disp}/{total}] {oid}  Ошибка: {err_msg[:60]}")
            processed += 1
            log_lines.append("")
            pic_o = (o.get("picture_urls") or [None])[0]
            img_o = ensure_image_cached(pic_o, cache_dir, max_size) if pic_o else None
            if img_o:
                last_image = str(Path(img_o).resolve())

    return log_lines, processed, errors, saved_ids, last_image


def _result_choice_label(r: dict) -> str:
    """Строка для multiselect на вкладке «Результаты»: offer_id | conf% | название."""
    oid = str(r.get("offer_id", ""))
    conf = r.get("avg_confidence", 0)
    name = (r.get("name") or "").replace("|", "·").strip()[:48]
    return f"{oid} | {conf}% | {name}"


def _parse_offer_id_from_choice(s: str) -> str:
    return (s or "").split("|", 1)[0].strip()


def _global_model_choices(ollama_url: str | None = None) -> list[str]:
    """Стандартные имена + `ollama list` для выпадающего списка моделей."""
    predefined = [
        "qwen3.5:4b",
        "qwen3.5:9b",
        "qwen3.5:35b",
        "qwen2.5-vl:3b",
        "qwen2.5-vl:7b",
        "qwen2.5-vl:72b",
    ]
    g = pm.get_global_settings()
    url = (ollama_url or g.get("ollama_url") or "http://127.0.0.1:11435").strip()
    try:
        from_ollama = ollama_list_models(url, timeout=3)
    except Exception:
        from_ollama = []
    merged = sorted(set(predefined) | set(from_ollama))
    return merged if merged else predefined


def _reprocess_results_worker(
    offer_ids: list[str],
    model_name: str,
    reanalyze_keys: list[str] | None = None,
) -> None:
    """Повторный прогон выбранных offer_id с другой моделью (фоновый поток). reanalyze_keys — частичный прогон."""
    name = _proj_name()
    if not name:
        print("[Reprocess] Проект не выбран")
        return
    cache_db = _cache_db()
    rdb = _results_db()
    if not cache_db or not cache_db.exists():
        print("[Reprocess] Кэш фида не найден — загрузите фид на вкладке «Фид»")
        return
    if not rdb or not rdb.exists():
        print("[Reprocess] База результатов недоступна")
        return
    global_settings = pm.get_global_settings()
    lastp = pm.load_run_prompt_last()
    cache_dir = pm.image_cache_dir(name)
    cfg_base = pm.strip_legacy_prompt_from_config({**global_settings, **_proj(), "image_cache_dir": str(cache_dir)})
    cfg_base["task_instruction"] = (lastp.get("task_instruction") or "").strip()
    cfg_base["task_constraints"] = (lastp.get("task_constraints") or "").strip()
    cfg_base["task_examples"] = (lastp.get("task_examples") or "").strip()
    _apply_lastp_targets_to_config(cfg_base, lastp)
    cfg_base["dynamic_clothing_attributes"] = bool(cfg_base.get("dynamic_clothing_attributes", True))
    cfg_base["use_full_prompt_edit"] = bool(lastp.get("use_full_prompt_edit"))
    cfg_base["full_prompt_text"] = (lastp.get("full_prompt_text") or "").strip()
    cfg_base["model"] = (model_name or "").strip() or global_settings.get("model", "qwen3.5:35b")
    keys = [str(x).strip() for x in (reanalyze_keys or []) if x and str(x).strip()]
    mode = f"ключи={keys!r}" if keys else "полный прогон"
    print(f"[Reprocess] старт: офферов={len(offer_ids)}, модель={cfg_base['model']}, проект={name}, {mode}")
    ok = 0
    for oid in offer_ids:
        oid = str(oid).strip()
        if not oid:
            continue
        offer = fc.get_offer_by_id(cache_db, oid)
        if not offer:
            print(f"[Reprocess] offer_id={oid} — нет в кэше фида (нужен тот же фид, что при обработке)")
            continue
        try:
            if keys:
                existing = _load_result_by_offer_id(rdb, oid)
                if not existing:
                    result = analyze_offer(offer, cfg_base)
                else:
                    partial = analyze_offer(offer, cfg_base, reanalyze_keys=keys)
                    err_p = partial.get("error") or ""
                    if err_p and "ни один ключ не совпал" in str(err_p):
                        print(f"[Reprocess] {oid} пропуск частичного: {err_p}")
                        continue
                    result = _merge_partial_reanalyze(existing, partial, keys)
                    result.setdefault("offer_id", oid)
                    result.setdefault("name", offer.get("name", ""))
                    result.setdefault("category", offer.get("category", ""))
                    pic0 = (offer.get("picture_urls") or [""])[0]
                    if pic0:
                        result["picture_url"] = pic0
            else:
                result = analyze_offer(offer, cfg_base)
            result["model"] = (cfg_base.get("model") or "").strip()
            _save_result(rdb, result)
            conf = result.get("avg_confidence", 0)
            line = _console_result_line(result, cfg_base)
            print(f"[Reprocess] {oid} OK  уверенность={conf}%  {line}")
            for _pl in _profile_run_log_lines(result):
                print(f"[Reprocess] {_pl.strip()}")
            ok += 1
        except Exception as e:
            print(f"[Reprocess] {oid} ошибка: {e}")
    print(f"[Reprocess] готово: обновлено {ok} из {len(offer_ids)}")


def _export_results_to_csv(
    results: list[dict],
    config: dict,
    glossary: dict,
    out_dir: Path | None = None,
) -> Path | None:
    """
    Выгружает переданный список результатов (уже отфильтрованный как на вкладке «Результаты»).
    Колонки: offer_id, название, ссылка_на_картинку, атрибут, значение.
    Плюс строки **Надпись** из text_detection (как на карточке), без прогона через глоссарий.
    """
    if not results:
        return None
    key_to_label = _key_to_label_map(config)
    dir_path = out_dir or Path(".")
    dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = dir_path / f"results_export_{timestamp}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["offer_id", "название", "ссылка_на_картинку", "атрибут", "значение"])
        for r in results:
            oid = r.get("offer_id", "")
            name = (r.get("name") or "").strip()
            pic_url = (r.get("picture_url") or "").strip()
            for _dir_id, attrs in (r.get("direction_attributes") or {}).items():
                if not isinstance(attrs, dict):
                    continue
                for key, entry in attrs.items():
                    if key == "error" or not isinstance(entry, dict):
                        continue
                    label_ru = key_to_label.get(key, key)
                    raw_val = entry.get("value", "")
                    value_ru = pm.translate_attribute_value(raw_val, glossary) if raw_val else ""
                    writer.writerow([oid, name, pic_url, label_ru, value_ru])
            ins = _text_detection_joined_for_export(r.get("text_detection"))
            if ins:
                writer.writerow([oid, name, pic_url, "Надпись", ins])

    return csv_path


def _export_results_to_csv_light(
    results: list[dict],
    config: dict,
    glossary: dict,
    out_dir: Path | None = None,
) -> Path | None:
    """
    Узкая таблица для выгрузки: external_id (= offer_id из фида), attribute_name, attribute_value.
    Без названия оффера и URL картинки.
    """
    if not results:
        return None
    key_to_label = _key_to_label_map(config)
    dir_path = out_dir or Path(".")
    dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = dir_path / f"results_export_light_{timestamp}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["external_id", "attribute_name", "attribute_value"])
        for r in results:
            oid = r.get("offer_id", "")
            for _dir_id, attrs in (r.get("direction_attributes") or {}).items():
                if not isinstance(attrs, dict):
                    continue
                for key, entry in attrs.items():
                    if key == "error" or not isinstance(entry, dict):
                        continue
                    label_ru = key_to_label.get(key, key)
                    raw_val = entry.get("value", "")
                    value_ru = pm.translate_attribute_value(raw_val, glossary) if raw_val else ""
                    writer.writerow([oid, label_ru, value_ru])
            ins = _text_detection_joined_for_export(r.get("text_detection"))
            if ins:
                writer.writerow([oid, "Надпись", ins])

    return csv_path


# ── Confidence badge HTML ──────────────────────────────────────────────────────

def _badge(label: str, value: str, confidence: int) -> str:
    if confidence >= 80:
        color = "#22c55e"
    elif confidence >= 50:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    return (
        f'<span style="display:inline-block;margin:2px 3px;padding:2px 7px;'
        f'border-radius:12px;background:{color}22;border:1px solid {color};'
        f'font-size:12px;color:#111;">'
        f"<b>{label}:</b> {value} <span style='opacity:.6'>({confidence}%)</span></span>"
    )


def _badge_inscription(label: str, value: str, confidence: int) -> str:
    """Оранжевый бейдж для надписей на фото (отдельно от шкалы уверенности атрибутов)."""
    color = "#ea580c"
    return (
        f'<span style="display:inline-block;margin:2px 3px;padding:2px 7px;'
        f'border-radius:12px;background:{color}22;border:1px solid {color};'
        f'font-size:12px;color:#111;">'
        f"<b>{html.escape(label)}:</b> {html.escape(value)} "
        f"<span style='opacity:.6'>({int(confidence)}%)</span></span>"
    )


def _result_run_meta_html(r: dict) -> str:
    ts = r.get("run_ts")
    model = (r.get("model") or "").strip()
    parts: list[str] = []
    if ts is not None:
        try:
            parts.append(time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts))))
        except (TypeError, ValueError, OSError):
            pass
    if model:
        parts.append(f"модель: {model}")
    if not parts:
        return ""
    esc = " · ".join(html.escape(p) for p in parts)
    return f'<div style="font-size:10px;color:#666;margin-bottom:6px;">{esc}</div>'


def _result_card_html(r: dict, badge_config: dict | None = None) -> str:
    from attribute_detector import parse_target_attribute_lines_to_keys, parse_task_target_list_from_config

    direction_attrs = r.get("direction_attributes", {})
    text_det = r.get("text_detection", {})
    pic = r.get("picture_url", "")
    conf = r.get("avg_confidence", 0)
    glossary = pm.load_attribute_glossary()

    if conf >= 80:
        border = "#22c55e"
    elif conf >= 50:
        border = "#f59e0b"
    else:
        border = "#ef4444"

    std_keys = frozenset(pm.default_clothing_standard_keys())
    prompt_keys: frozenset[str] = frozenset()
    if badge_config:
        tlist, _ = parse_task_target_list_from_config(badge_config)
        prompt_keys = frozenset(parse_target_attribute_lines_to_keys(tlist))

    badge_rows: list[tuple[int, str, str, str, int]] = []
    for _dir_name, attrs in direction_attrs.items():
        if isinstance(attrs, dict) and attrs.get("error"):
            continue
        for key, entry in (attrs or {}).items():
            if key == "error" or not isinstance(entry, dict):
                continue
            label = entry.get("label", key)
            value = pm.translate_attribute_value(entry.get("value", "—"), glossary)
            c = int(entry.get("confidence", 0))
            if key in prompt_keys:
                prio = 0
            elif key not in std_keys:
                prio = 1
            else:
                prio = 2
            badge_rows.append((prio, str(key), label, value, c))
    badge_rows.sort(key=lambda t: (t[0], t[1].casefold()))
    badges = "".join(_badge(row[2], row[3], row[4]) for row in badge_rows)

    # Надписи из БД показываем всегда, если они есть — не зависит от текущих галок «Искать надписи».
    texts = text_det.get("texts") or []
    if texts:
        text_joined = ", ".join(str(t).strip() for t in texts if str(t).strip())
        if text_joined:
            text_conf = int(text_det.get("confidence") or 0)
            badges += _badge_inscription("Надпись", text_joined, text_conf)

    pic_esc = html.escape(pic, quote=True)
    if pic:
        thumb = (
            f'<a href="{pic_esc}" target="_blank" rel="noopener noreferrer" title="Открыть оригинал">'
            f'<img src="{pic_esc}" alt="" style="width:120px;height:120px;object-fit:cover;'
            f'border-radius:6px;flex-shrink:0;display:block;" onerror="this.style.display=\'none\'"></a>'
            f'<div style="margin-top:4px;font-size:11px;">'
            f'<a href="{pic_esc}" target="_blank" rel="noopener noreferrer">Ссылка на фото</a>'
            f' · <details style="display:inline;vertical-align:top;">'
            f'<summary style="cursor:pointer;display:inline;user-select:none;">Крупнее</summary>'
            f'<div style="margin-top:6px;"><img src="{pic_esc}" alt="" '
            f'style="max-width:min(92vw,560px);max-height:70vh;width:auto;height:auto;object-fit:contain;'
            f'border-radius:8px;border:1px solid #ddd;"></div></details></div>'
        )
        img_html = f'<div style="flex-shrink:0;">{thumb}</div>'
    else:
        img_html = ""

    meta = _result_run_meta_html(r)

    return (
        f'<div style="display:flex;gap:12px;align-items:flex-start;padding:10px;'
        f'border:1px solid {border};border-radius:10px;margin-bottom:8px;background:#fafafa;">'
        f"{img_html}"
        f'<div style="flex:1;min-width:0;">'
        f'{meta}'
        f'<div style="font-weight:600;font-size:13px;margin-bottom:4px;">'
        f'{html.escape((r.get("name") or "")[:60])} '
        f'<span style="color:#888;font-size:11px">#{html.escape(str(r.get("offer_id","")))}</span></div>'
        f'<div style="font-size:11px;color:#666;margin-bottom:4px;">{html.escape(r.get("category") or "")}</div>'
        f"<div>{badges}</div>"
        f'<div style="font-size:11px;color:#888;margin-top:4px;">Ср. уверенность: <b>{conf}%</b></div>'
        f"</div></div>"
    )


def _result_to_correction_form(r: dict, config: dict | None = None, glossary: dict | None = None) -> tuple[str, str, str]:
    """Из результата оффера собирает (offer_id, JSON атрибутов, текст на одежде) для формы правки.
    Если передан config и glossary — JSON с ключами и значениями по-русски."""
    attrs = {}
    for _dir, adict in (r.get("direction_attributes") or {}).items():
        if not isinstance(adict, dict):
            continue
        for k, v in (adict or {}).items():
            if isinstance(v, dict) and "value" in v:
                attrs[k] = v["value"]
    text = ", ".join((r.get("text_detection") or {}).get("texts") or [])
    if config and glossary is not None:
        attrs_json = pm.attrs_to_russian_json(attrs, config, glossary)
    else:
        attrs_json = json.dumps(attrs, ensure_ascii=False, indent=2)
    return r.get("offer_id", ""), attrs_json, text


# Максимум карточек с кнопкой «править» на вкладке Результаты (остальное — одним блоком HTML)
RESULTS_PAGE_SIZE = 25
MAX_RESULT_CARDS = RESULTS_PAGE_SIZE


def _correction_hint_md() -> str:
    """Текст подсказки для формы правки: ключи и значения по-русски."""
    glossary = pm.load_attribute_glossary()
    lines = ["Пишите **ключи** и **значения** по-русски — при сохранении подставятся английские эквиваленты.", ""]
    for d in pm.DEFAULT_DIRECTIONS:
        for a in d.get("attributes", []):
            label = a.get("label") or a.get("key", "")
            opts = a.get("options", [])
            ru_opts = [pm.translate_attribute_value(o, glossary) for o in opts]
            lines.append(f"- **{label}**: " + ", ".join(ru_opts))
    lines.append("")
    lines.append("Пример: `{\"Длина рукава\": \"длинный\", \"Воротник\": \"поло\"}`")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Projects
# ═══════════════════════════════════════════════════════════════════════════════

def tab_projects(app: gr.Blocks | None = None, header_project: gr.Markdown | None = None):
    with gr.Tab("Проекты") as projects_tab:
        gr.Markdown("## Управление проектами")
        gr.Markdown(
            "Проект — это **имя** (не путь). Создаётся папка `projects/<имя>/` с конфигом, кэшем и результатами. "
            "**Путь к YML/XML-фиду с картинками** указывается на вкладке **Фид** (например `C:\\Users\\...\\yml-feed.1967.global.xml`)."
        )
        with gr.Column():
            project_list = gr.Dropdown(
                label="Существующий проект (из папки projects/)",
                choices=pm.list_projects(),
                value=None,
            )
            btn_load = gr.Button("Загрузить проект", variant="primary")
            gr.Markdown("---")
            new_name = gr.Textbox(
                label="Имя нового проекта",
                placeholder="например: my_project или одежда_2025",
                info="Латиница, цифры, подчёркивание. Будет создана папка projects/<имя>/",
            )
            _vert_create_choices = [CREATE_PROJECT_VERTICAL_PLACEHOLDER] + pm.get_vertical_choices()
            new_vertical = gr.Dropdown(
                label="Вертикаль (сфера)",
                choices=_vert_create_choices,
                value=CREATE_PROJECT_VERTICAL_PLACEHOLDER,
                allow_custom_value=True,
                info="Выберите из списка, **или «Другое»** и введите название ниже, **или введите своё название прямо в поле** (сохранится в общий список). Список обновляется после «Сохранить всё» в Настройках.",
            )
            new_vertical_other = gr.Textbox(
                label="Название вертикали (при выборе «Другое»)",
                placeholder="Введите название — оно сохранится и будет предлагаться в списке",
                visible=False,
            )
            gr.Markdown("_Промпт и атрибуты задаются на вкладке **«Запуск»** (пресеты), не в проекте._")
            btn_create = gr.Button("Создать новый проект")
            project_status = gr.Markdown("_Проект не выбран_", elem_classes=["project-status"])

        def load_project(name):
            global _current_project
            if not name:
                return "_Выберите проект из списка_", "_Не выбран_"
            try:
                _current_project = pm.load_project(name)
                pm.set_last_project(name)
                g = pm.get_global_settings()
                status = f"✅ Загружен проект **{name}**\n\nМодель (общая для всех): `{g.get('model','')}`\nOllama: `{g.get('ollama_url','')}`"
                return status, f"Проект: **{name}**"
            except Exception as e:
                return f"❌ {e}", "_Не выбран_"

        def create_project(name, vertical, custom_vertical_name):
            global _current_project
            name = (name or "").strip()
            if not name:
                return "_Введите название_", gr.update(), gr.update(), "_Не выбран_"
            v = (vertical or "").strip()
            if not v or v == CREATE_PROJECT_VERTICAL_PLACEHOLDER:
                return (
                    "❌ **Выберите вертикаль** в выпадающем списке — без этого проект не создаётся.",
                    gr.update(),
                    gr.update(),
                    "_Не выбран_",
                )
            if v == pm.VERTICAL_OTHER:
                effective = (custom_vertical_name or "").strip()
                if not effective:
                    return (
                        "❌ Выбрано «Другое» — **введите название вертикали** в поле ниже.",
                        gr.update(),
                        gr.update(),
                        "_Не выбран_",
                    )
            else:
                effective = v
            if effective not in pm.get_vertical_choices():
                pm.add_custom_vertical(effective)
            try:
                _current_project = pm.create_project(
                    name,
                    vertical=effective,
                )
                pm.set_last_project(name)
                proj_choices = pm.list_projects()
                vert_choices = pm.get_vertical_choices()
                create_dd_choices = [CREATE_PROJECT_VERTICAL_PLACEHOLDER] + vert_choices
                return (
                    f"✅ Создан проект **{name}** (вертикаль: **{effective}**)",
                    gr.update(choices=proj_choices, value=name),
                    gr.update(choices=create_dd_choices, value=CREATE_PROJECT_VERTICAL_PLACEHOLDER),
                    f"Проект: **{name}**",
                )
            except Exception as e:
                return f"❌ {e}", gr.update(), gr.update(), "_Не выбран_"

        def restore_project():
            global _current_project
            last = pm.get_last_project()
            choices = pm.list_projects()
            if not last or last not in choices:
                return gr.update(choices=choices), "_Проект не выбран_", "_Не выбран_"
            try:
                _current_project = pm.load_project(last)
                g = pm.get_global_settings()
                status = f"✅ Загружен проект **{last}**\n\nМодель (общая для всех): `{g.get('model','')}`\nOllama: `{g.get('ollama_url','')}`"
                return gr.update(choices=choices, value=last), status, f"Проект: **{last}**"
            except Exception:
                return gr.update(choices=choices), "_Проект не выбран_", "_Не выбран_"

        def toggle_vertical_other(vertical):
            return gr.update(visible=(vertical == pm.VERTICAL_OTHER))

        new_vertical.change(
            toggle_vertical_other,
            inputs=[new_vertical],
            outputs=[new_vertical_other],
        )
        if header_project is not None:
            btn_load.click(load_project, inputs=project_list, outputs=[project_status, header_project])
            btn_create.click(
                create_project,
                inputs=[new_name, new_vertical, new_vertical_other],
                outputs=[project_status, project_list, new_vertical, header_project],
            )
            if app is not None:
                app.load(restore_project, outputs=[project_list, project_status, header_project])
        else:
            def load_project_only(name):
                return load_project(name)[0]
            def create_project_only(name, vertical, custom_vertical_name):
                r = create_project(name, vertical, custom_vertical_name)
                return r[0], r[1]
            def restore_project_only():
                r = restore_project()
                return r[0], r[1]
            btn_load.click(load_project_only, inputs=project_list, outputs=project_status)
            btn_create.click(
                create_project_only,
                inputs=[new_name, new_vertical, new_vertical_other],
                outputs=[project_status, project_list],
            )
            if app is not None:
                app.load(restore_project_only, outputs=[project_list, project_status])

    def _refresh_create_vertical_dropdown():
        vc = pm.get_vertical_choices()
        return gr.update(choices=[CREATE_PROJECT_VERTICAL_PLACEHOLDER] + vc)

    if app is not None:
        projects_tab.select(_refresh_create_vertical_dropdown, outputs=[new_vertical])

    return new_vertical


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Feed
# ═══════════════════════════════════════════════════════════════════════════════

def tab_feed():
    with gr.Tab("Фид"):
        gr.Markdown("## Фид с картинками (YML/XML)")
        gr.Markdown("Укажите **путь к файлу** на диске или **перетащите файл** в область ниже.")
        feed_upload = gr.File(
            label="Перетащите файл сюда (YML / XML)",
            file_types=[".xml", ".yml", ".yaml"],
        )
        with gr.Row():
            feed_path_box = gr.Textbox(
                label="Или путь к файлу на диске",
                placeholder="C:\\Users\\1\\Downloads\\yml-feed.1967.global.xml",
                scale=5,
            )
            btn_parse = gr.Button("Загрузить и кэшировать", variant="primary", scale=1)

        cache_status = gr.Markdown("_Выберите проект и нажмите «Показать статус фида» — или загрузите фид по пути выше._")
        categories_table = gr.Dataframe(
            headers=["Категория", "Офферов"],
            label="Категории в фиде",
            interactive=False,
        )

        # ── Источники картинок и режим мульти-изображений ──────────────────────
        with gr.Accordion("Источник картинок и мульти-изображения", open=True):
            gr.Markdown(
                "После загрузки фида здесь появятся **XML-теги**, в которых были найдены ссылки на картинки. "
                "Выберите нужные — именно из них будут браться URL для обработки. "
                "Пустой выбор = использовать **все** найденные картинки (как раньше)."
            )
            feed_attr_choices = gr.CheckboxGroup(
                label="Теги-источники картинок (из последнего загруженного фида)",
                choices=[],
                value=[],
                info="Например: picture, param:picture, photo. После «Загрузить и кэшировать» список обновится автоматически.",
            )
            feed_picture_indices = gr.CheckboxGroup(
                label="Какие по счёту фото брать (пусто = все)",
                choices=[
                    ("1-е", 1), ("2-е", 2), ("3-е", 3),
                    ("4-е", 4), ("5-е", 5),
                ],
                value=[],
                info="Выберите порядковые номера фото в рамках выбранного тега. Например: «2-е» = второй <picture> у каждого оффера. Пусто = все.",
            )
            feed_multi_image_mode = gr.Radio(
                label="Режим нескольких картинок одного оффера",
                choices=[
                    ("Только первая картинка (быстро, классика)", "first_only"),
                    ("Модель выбирает лучшую картинку (best_select)", "best_select"),
                    ("Все картинки передаются в один запрос (all_images)", "all_images"),
                ],
                value="first_only",
                info=(
                    "**best_select** — небольшой vision-запрос определяет лучшую картинку, затем основной анализ. "
                    "**all_images** — все картинки уходят модели в одном запросе (медленнее, но выше точность при нескольких ракурсах). "
                    "Поддерживается Ollama-моделями с vision (qwen2.5-vl, qwen3.5 и др.)."
                ),
            )
            btn_save_picture_settings = gr.Button("Сохранить настройки источника картинок в проект", variant="primary", size="sm")
            picture_settings_status = gr.Markdown("")

        def _resolve_feed_path(path_typed, uploaded_path):
            if uploaded_path and str(uploaded_path).strip():
                p = Path(str(uploaded_path).strip())
                if p.exists():
                    return str(p.resolve())
            path_str = (path_typed or "").strip().strip('"').strip("'")
            if path_str and Path(path_str).exists():
                return str(Path(path_str).resolve())
            return ""

        def _attr_choices_from_db(db_path) -> list[str]:
            try:
                return fc.get_feed_image_attr_names(db_path)
            except Exception:
                return []

        def _cfg_picture_values():
            """Достаёт текущие настройки картинок из проекта."""
            cfg = _proj()
            filt = cfg.get("picture_attr_filter") or []
            mode = cfg.get("multi_image_mode") or "first_only"
            indices = [int(x) for x in (cfg.get("picture_index_filter") or []) if str(x).isdigit()]
            return filt, mode, indices

        def parse_feed(path_typed, uploaded_path):
            name = _proj_name()
            if not name:
                return "❌ Сначала выберите проект", gr.update(value=[]), None, gr.update(), gr.update(), gr.update()
            feed_path = _resolve_feed_path(path_typed, uploaded_path)
            if not feed_path:
                return "❌ Укажите путь к существующему файлу или перетащите файл в область загрузки", gr.update(value=[]), None, gr.update(), gr.update(), gr.update()
            try:
                db = pm.cache_db_path(name)
                summary = fc.parse_feed_to_cache(feed_path, db)
                pm.save_project({**_proj(), "feed_path": feed_path})
                rows = [[cat, cnt] for cat, cnt in sorted(summary["categories"].items())]
                msg = f"✅ Загружено **{summary['total']}** офферов в {len(summary['categories'])} категориях. Файл: `{feed_path}`"
                if summary.get("total", 0) == 0:
                    msg += (
                        "\n\n⚠ **0 офферов** — внутри `<offer>` не найдено ни одной ссылки на картинку (`https://`, `http://` или `//`). "
                        "Проверьте теги `<picture>` / `<param>`, целостность файла (очень длинный фид в одну строку — нормально)."
                    )
                # Обновляем список тегов-источников
                attr_names = _attr_choices_from_db(db)
                saved_filter, saved_mode, saved_indices = _cfg_picture_values()
                # Оставляем только те из сохранённого фильтра, которые ещё присутствуют в фиде
                active_filter = [a for a in saved_filter if a in attr_names]
                return (
                    msg,
                    rows,
                    gr.update(value=None),
                    gr.update(choices=attr_names, value=active_filter),
                    gr.update(value=saved_mode),
                    gr.update(value=saved_indices),
                )
            except Exception as e:
                return f"❌ {e}", gr.update(value=[]), None, gr.update(), gr.update(), gr.update()

        def refresh_cache():
            name = _proj_name()
            if not name:
                return "Сначала выберите проект на вкладке **«Проекты»**. Затем здесь нажмите «Показать статус фида» или загрузите фид.", gr.update(value=[]), gr.update(), gr.update(), gr.update()
            db = pm.cache_db_path(name)
            if not db.exists():
                return "**Фид не загружен.** Укажите путь к YML/XML выше и нажмите «Загрузить и кэшировать».", gr.update(value=[]), gr.update(), gr.update(), gr.update()
            cats = fc.get_categories(db)
            rows = [[cat, cnt] for cat, cnt in sorted(cats.items())]
            meta = fc.get_cache_meta(db)
            total = meta.get("offer_count", "?")
            attr_names = _attr_choices_from_db(db)
            saved_filter, saved_mode, saved_indices = _cfg_picture_values()
            active_filter = [a for a in saved_filter if a in attr_names]
            return (
                f"**Фид загружен (кэш).** Офферов: **{total}**, категорий: **{len(cats)}**. Ниже таблица.",
                gr.update(value=rows),
                gr.update(choices=attr_names, value=active_filter),
                gr.update(value=saved_mode),
                gr.update(value=saved_indices),
            )

        def save_picture_settings(attr_filter, multi_mode, index_filter):
            name = _proj_name()
            if not name:
                return "❌ Сначала выберите проект"
            filt = [a for a in (attr_filter or []) if a.strip()]
            mode = (multi_mode or "first_only").strip()
            indices = [int(x) for x in (index_filter or []) if str(x).isdigit()]
            updated = {**_proj(), "picture_attr_filter": filt, "multi_image_mode": mode, "picture_index_filter": indices}
            try:
                pm.save_project(updated)
                global _current_project
                _current_project = updated
                filt_txt = ", ".join(f"`{a}`" for a in filt) if filt else "_все теги_"
                mode_labels = {
                    "first_only": "Только первая картинка",
                    "best_select": "Модель выбирает лучшую",
                    "all_images": "Все картинки в один запрос",
                }
                idx_txt = ", ".join(str(i) for i in indices) if indices else "_все_"
                return f"✅ Сохранено. Теги: {filt_txt} · Режим: **{mode_labels.get(mode, mode)}** · Индексы фото: {idx_txt}"
            except Exception as e:
                return f"❌ {e}"

        btn_parse.click(
            parse_feed,
            inputs=[feed_path_box, feed_upload],
            outputs=[cache_status, categories_table, feed_upload, feed_attr_choices, feed_multi_image_mode, feed_picture_indices],
        )

        btn_refresh = gr.Button("Показать статус фида / обновить категории", variant="secondary")
        btn_refresh.click(
            refresh_cache,
            outputs=[cache_status, categories_table, feed_attr_choices, feed_multi_image_mode, feed_picture_indices],
        )

        btn_save_picture_settings.click(
            save_picture_settings,
            inputs=[feed_attr_choices, feed_multi_image_mode, feed_picture_indices],
            outputs=[picture_settings_status],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Run
# ═══════════════════════════════════════════════════════════════════════════════

def _run_model_hint_text() -> str:
    g = pm.get_global_settings()
    m = g.get("model", "?")
    bw = max(1, min(16, int(g.get("batch_offer_workers", 1) or 1)))
    mpv = int(g.get("max_parallel_vision", 0) or 0)
    mpv_txt = "без лимита" if mpv <= 0 else str(mpv)
    _par = (
        f" **Разных фото одновременно:** **{bw}**; **запросов к модели на одно фото:** **{mpv_txt}** — вкладка **«Настройки»**."
    )
    if bw > 1:
        _par += f" Во время прогона в статусе будет **«×{bw} фото параллельно»**."
    try:
        from attribute_detector import _is_adapter_path
        if _is_adapter_path(m):
            return f"**Модель для обработки:** **{m}**. Сменить — вкладка **«Настройки»**.{_par}"
    except Exception:
        pass
    url = g.get("ollama_url", "http://127.0.0.1:11435")
    try:
        loaded = ollama_loaded_models(url, timeout=3)
        in_mem = ", ".join(l.get("name", "?") for l in loaded) if loaded else "ни одной"
    except Exception:
        in_mem = "—"
    if in_mem != "ни одной" and in_mem != "—":
        return (
            f"**Модель для обработки** (из Настроек): **{m}** — при запуске запрос пойдёт именно в неё. "
            f"Сейчас в памяти Ollama: **{in_mem}**. Если нужно обрабатывать другой моделью — смените в **«Настройки»**.{_par}"
        )
    return (
        f"**Модель для обработки** (из Настроек): **{m}**. В памяти Ollama сейчас пусто — при первом запросе подгрузится эта модель. "
        "Изменить — вкладка **«Настройки»** (qwen3.5:4b/9b/35b или своя)."
        + _par
    )


def _ollama_status_html() -> str:
    url = pm.get_global_settings().get("ollama_url", "http://127.0.0.1:11435")
    check = normalize_ollama_url(url).rstrip("/") + "/"
    try:
        r = requests.get(check, timeout=ollama_root_health_timeout_s(url))
        r.raise_for_status()
        models = ollama_loaded_models(url, timeout=3)
        if models:
            parts = [f"<b>{html.escape(m.get('name', '?'))}</b> ({(m.get('size_vram') or 0) / (1024**3):.1f} ГБ)" for m in models]
            in_mem = "В памяти: " + ", ".join(parts) + ". Выгрузка — в Настройки."
        else:
            in_mem = "В памяти пусто (модель подгрузится при первом запросе)."
        return (
            "<div style='padding:8px 14px;border-radius:6px;background:#dcfce7;border:1px solid #86efac;color:#166534;font-weight:500'>"
            "✅ Ollama доступен — "
            + url
            + "<br><small>"
            + in_mem
            + "</small></div>"
        )
    except Exception:
        return (
            "<div style='padding:8px 14px;border-radius:6px;background:#fee2e2;border:1px solid #fca5a5;color:#991b1b;font-weight:500'>"
            "❌ Ollama недоступен (" + url + ")<br>"
            "<small>Установите Ollama: <a href='https://ollama.com/download' target='_blank'>ollama.com/download</a> → запустите установщик → перезапустите run.bat</small></div>"
        )


def tab_run(header_model_badge: gr.HTML | None = None):
    _rp = pm.load_run_prompt_last()

    def _run_preset_choices() -> list[str]:
        n = _proj_name()
        if not n:
            return []
        return [p.get("name", "") for p in pm.load_project_prompt_presets(n) if p.get("name")]

    with gr.Tab("Запуск"):
        _bench_model_choices = [
            "qwen3.5:4b",
            "qwen3.5:9b",
            "qwen3.5:27b",
            "qwen3.5:35b",
            "qwen3.5:35b-a3b",
        ]

        # ── Статус Ollama (компактная строка) ────────────────────────────────
        with gr.Row():
            ollama_status_html = gr.HTML(
                value="<div style='padding:6px 14px;border-radius:6px;background:#f3f4f6;color:#6b7280'>⏳ Проверка Ollama…</div>"
            )
            ollama_pool_status_html = gr.HTML(
                value="<div style='padding:6px 14px;font-size:13px;color:#64748b'>⏳ Пул Ollama…</div>",
                scale=0,
            )
            btn_check_ollama = gr.Button("Проверить Ollama", size="sm", variant="secondary", scale=0)
        run_model_hint = gr.Markdown(value="", visible=False)

        # ── Категории + лимит ────────────────────────────────────────────────
        with gr.Row(elem_classes=["run-tab-filters-row"]):
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Категории")
                cat_search_box = gr.Textbox(
                    label="Поиск категорий",
                    placeholder="Введите часть названия…",
                    scale=1,
                )
                categories_check = gr.CheckboxGroup(
                    label="Отметьте нужные или оставьте пустым (= весь фид)",
                    choices=[],
                    elem_classes=["run-categories-list"],
                )
                btn_load_cats = gr.Button("Обновить список категорий", size="sm", variant="secondary")
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Лимит и режимы")
                limit_box = gr.Number(
                    label="Макс. офферов (0 = без лимита)",
                    value=10,
                    precision=0,
                    minimum=0,
                )
                with gr.Row():
                    btn_limit_10 = gr.Button("10", size="sm", min_width=40)
                    btn_limit_50 = gr.Button("50", size="sm", min_width=40)
                    btn_limit_100 = gr.Button("100", size="sm", min_width=40)
                    btn_limit_500 = gr.Button("500", size="sm", min_width=40)
                    btn_limit_0 = gr.Button("∞", size="sm", min_width=40)
                process_all = gr.Checkbox(
                    label="Обработать весь фид (игнорировать категории)",
                    value=False,
                )
                force_reprocess = gr.Checkbox(
                    label="Перезаписать уже обработанные (запустить модель заново)",
                    value=False,
                )

        # ── Промпт и пресеты ─────────────────────────────────────────────────
        gr.Markdown("### Промпт и пресеты")
        with gr.Row():
            run_prompt_preset = gr.Dropdown(
                label="Пресет проекта",
                choices=_run_preset_choices(),
                value=None,
                allow_custom_value=False,
                scale=3,
            )
            btn_refresh_preset_list = gr.Button("↻", size="sm", variant="secondary", scale=0)
            btn_edit_preset = gr.Button("✏️ Изменить", size="sm", variant="primary", scale=0)
            btn_new_preset = gr.Button("➕ Новый", size="sm", variant="secondary", scale=0)
        with gr.Row():
            btn_reset_clothing_run = gr.Button(
                "🧥 Только одежда: сбросить задание, атрибуты и пресет",
                size="sm",
                variant="secondary",
            )
        run_prefs_saved = gr.Markdown(value="")
        with gr.Row(elem_classes=["run-tab-main-row"]):
            with gr.Column(scale=1, min_width=448):
                prompt_main_group = gr.Group(visible=True)
                with prompt_main_group:
                    gr.Markdown("#### Редактор промпта")
                    with gr.Row():
                        preset_save_name = gr.Textbox(
                            label="Имя пресета",
                            placeholder="например: Ювелирка — цвет металла",
                            scale=4,
                        )
                        btn_save_preset = gr.Button("Сохранить как пресет", size="sm", variant="primary", scale=1)
                        btn_delete_preset = gr.Button("Удалить", size="sm", variant="stop", scale=0)
                    run_task_instruction = gr.Textbox(
                        label="Задание",
                        placeholder="Коротко, своими словами. К промпту автоматически добавятся: фокус на фото, «верь глазам», формат JSON, русский, уверенность.",
                        lines=3,
                        value=_rp.get("task_instruction") or "",
                    )
                    run_task_constraints = gr.Textbox(
                        label="Доп. ограничения (опционально)",
                        placeholder="Например: не путать цвет металла с цветом камня; только видимое на фото.",
                        lines=2,
                        value=_rp.get("task_constraints") or "",
                    )
                    run_task_examples = gr.Textbox(
                        label="Примеры и уточнения (опционально)",
                        placeholder="«золотое кольцо с бриллиантом» — цвет металла с картинки; если на фото серебро, а в названии золото — ответ серебро.",
                        lines=4,
                        value=_rp.get("task_examples") or "",
                    )
                    _ta_init = _target_attrs_from_lastp(_rp)
                    while len(_ta_init) < 5:
                        _ta_init.append("")
                    _ta_init = _ta_init[:5]
                    _ta_n0 = max(1, min(5, len([x for x in _ta_init if x.strip()]) or 1))
                    gr.Markdown(
                        "**Извлекаемые атрибуты** (опционально). Для не-одежды — ключи JSON "
                        "(например `Цвет металла (metal_color)`). Для «Одежды» оставьте пустыми."
                    )
                    run_ta_visible_n = gr.State(_ta_n0)
                    run_task_target_attr_0 = gr.Textbox(
                        label="Атрибут 1",
                        placeholder="Например: Цвет металла (metal_color)",
                        lines=1,
                        value=_ta_init[0],
                    )
                    run_task_target_attr_1 = gr.Textbox(
                        label="Атрибут 2",
                        lines=1,
                        value=_ta_init[1],
                        visible=_ta_n0 >= 2,
                    )
                    run_task_target_attr_2 = gr.Textbox(
                        label="Атрибут 3",
                        lines=1,
                        value=_ta_init[2],
                        visible=_ta_n0 >= 3,
                    )
                    run_task_target_attr_3 = gr.Textbox(
                        label="Атрибут 4",
                        lines=1,
                        value=_ta_init[3],
                        visible=_ta_n0 >= 4,
                    )
                    run_task_target_attr_4 = gr.Textbox(
                        label="Атрибут 5",
                        lines=1,
                        value=_ta_init[4],
                        visible=_ta_n0 >= 5,
                    )
                    with gr.Row():
                        btn_ta_add = gr.Button("+ строка атрибута", size="sm", variant="secondary")
                        btn_ta_remove = gr.Button("− убрать строку", size="sm", variant="secondary")
                        btn_save_last_only = gr.Button(
                            "Сохранить поля",
                            size="sm",
                            variant="secondary",
                        )
                    with gr.Accordion("Свой полный текст для модели (обычно не нужно)", open=False):
                        gr.Markdown(
                            "По умолчанию полный запрос **собирается из полей выше** — это нормальный путь. "
                            "Откройте этот блок только если нужно **вручную** подменить весь текст, отправляемый в Ollama."
                        )
                        run_use_full_prompt = gr.Checkbox(
                            label="Не собирать из полей — слать в модель только текст из окна ниже",
                            value=bool(_rp.get("use_full_prompt_edit")),
                        )
                        btn_compose_prompt = gr.Button("Подставить в окно предпросмотр из полей выше", size="sm", variant="secondary")
                        run_full_prompt_text = gr.Textbox(
                            label="Текст (только если включена галка выше)",
                            placeholder="Пусто = не используется, пока галка выключена.",
                            lines=10,
                            value=_rp.get("full_prompt_text") or "",
                        )

            with gr.Column(scale=1):
                _init_state = _load_run_state()
                _init_status = (_init_state.get("status") or "_Ожидание_") if _init_state else "_Ожидание_"
                _init_log = (
                    _run_log_text_for_ui(list(_init_state.get("log") or [])) if _init_state else ""
                )
                _init_img = _init_state.get("current_image") if _init_state else None
                run_status = gr.Markdown(value=_init_status)
                run_current_image = gr.Image(
                    label="Текущая картинка",
                    elem_id="run-current-image",
                    height=400,
                    show_label=True,
                    value=_init_img,
                    interactive=False,
                    buttons=["download", "fullscreen"],
                )
                gr.Markdown(
                    "_Кнопка с рамкой — полноэкранный просмотр. Обработка идёт **в фоне**: можно переключать вкладки, "
                    "вернуться на «Запуск» — статус и лог обновятся сами._"
                )
                run_progress = gr.Textbox(
                    label="Лог обработки",
                    lines=14,
                    interactive=False,
                    value=_init_log,
                    elem_id=RUN_LOG_TEXTBOX_ELEM_ID,
                    elem_classes=["run-log-textbox"],
                    info="Новые строки внизу. Если прокрутили вверх — лог не «прыгает»; у нижнего края — следует за концом.",
                )
                with gr.Row():
                    btn_start = gr.Button("▶ Запустить", variant="primary")
                    btn_stop = gr.Button("⏹ Остановить", variant="stop")
                    btn_refresh_status = gr.Button("Обновить статус", variant="secondary")
                run_status_timer = gr.Timer(2, active=True)
                run_pool_busy_block = gr.Checkbox(
                    value=True,
                    label="Не запускать обработку, если пул Ollama перегружен",
                    info="Смотрит `/_ollama_queue/status` (URL из Настроек, обычно :11435). Снимите галку, чтобы игнорировать.",
                )
                with gr.Accordion("Поставить произвольную CLI-команду в очередь пула", open=False):
                    gr.Markdown(
                        "**cwd** — обычно корень приложения (папка с `run.py`). **argv** — JSON-массив строк, "
                        "например `[\"python\",\"run.py\",\"--feed\",\"path/to/feed.yml\",\"--limit\",\"20\",\"--no-interactive\"]`."
                    )
                    enc_title = gr.Textbox(label="Название задачи", value="image_desc CLI")
                    enc_cwd = gr.Textbox(label="cwd", value=str(Path(__file__).resolve().parent))
                    enc_argv = gr.Textbox(
                        label="argv (JSON)",
                        value='["python", "run.py", "--help"]',
                    )
                    enc_submit = gr.Button("В очередь пула", variant="secondary")
                    enc_result = gr.Markdown("")

                    def _enqueue_custom_cli(title: str, cwd: str, argv_json: str) -> str:
                        try:
                            argv = json.loads(argv_json)
                        except json.JSONDecodeError as e:
                            return f"Невалидный JSON: {e}"
                        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
                            return "argv должен быть JSON-массивом строк."
                        g = pm.get_global_settings()
                        url = g.get("ollama_url", pm.GLOBAL_DEFAULTS["ollama_url"])
                        ok, msg = pool_jobs_client.enqueue_cli(
                            url,
                            title=(title or "cli").strip() or "cli",
                            project="image_description",
                            cwd=(cwd or "").strip() or str(Path(__file__).resolve().parent),
                            argv=argv,
                        )
                        return f"Поставлено в очередь, id: **{msg}**" if ok else f"Ошибка: {msg}"

                    enc_submit.click(_enqueue_custom_cli, inputs=[enc_title, enc_cwd, enc_argv], outputs=[enc_result])

                with gr.Accordion("📊 Тестовый прогон: сравнение моделей", open=False):
                    gr.Markdown(
                        "По очереди прогоняет **одни и те же** offer_id из **БД результатов** текущего проекта. "
                        "Используются текущие направления и промпт. "
                        "CLI: `python scripts/bench_vision_models.py -p ИМЯ --limit 10`."
                    )
                    model_bench_pick = gr.CheckboxGroup(
                        label="Модели для сравнения",
                        choices=_bench_model_choices,
                        value=["qwen3.5:9b", "qwen3.5:35b", "qwen3.5:35b-a3b"],
                    )
                    model_bench_n = gr.Number(label="Карточек из БД", value=10, precision=0, minimum=1, maximum=200)
                    with gr.Row():
                        btn_model_bench = gr.Button(
                            "▶ Запустить сравнение моделей",
                            variant="primary",
                        )
                    model_bench_status = gr.Markdown(value="_Статус появится после запуска; обновление каждые 2 с._")

        def load_categories(selected: list | None):
            db = _cache_db()
            if not db or not db.exists():
                return gr.update(choices=[], value=[])
            cats = sorted(fc.get_categories(db).keys())
            prev = [x for x in (selected or []) if x in cats]
            return gr.update(choices=cats, value=prev)

        def filter_categories(search_text: str, selected: list | None):
            db = _cache_db()
            if not db or not db.exists():
                return gr.update()
            cats = sorted(fc.get_categories(db).keys())
            q = (search_text or "").strip().lower()
            filtered = [c for c in cats if q in c.lower()] if q else cats
            valid_selected = [x for x in (selected or []) if x in filtered]
            return gr.update(choices=filtered, value=valid_selected)

        def run_processing(selected_cats, limit, all_feed, force_reprocess=False):
            global _run_log, _run_stop_event
            _run_stop_event.clear()
            _run_log = []

            name = _proj_name()
            if not name:
                _run_log.append("Ошибка: выберите проект на вкладке «Проекты».")
                yield "❌ Проект не выбран", None, _run_log_text_for_ui(_run_log)
                return
            db = _cache_db()
            if not db or not db.exists():
                _run_log.append("Ошибка: загрузите фид на вкладке «Фид».")
                yield "❌ Кэш не загружен", None, _run_log_text_for_ui(_run_log)
                return

            cats = None if all_feed else (selected_cats or None)
            cats_log = _format_categories_for_log(cats)
            limit_n = _coerce_run_limit(limit)
            rdb = pm.results_db_path(name)
            if force_reprocess:
                _run_log.append("Режим перезаписи: все офферы будут обработаны заново.")
            offers, skipped, resumed, resume_msg, fp = _resolve_offers_for_run(
                name, db, cats, limit_n, bool(all_feed), force_reprocess, rdb
            )
            total = len(offers)
            if resume_msg:
                _run_log.append(resume_msg)
            if skipped and not resumed:
                _run_log.append(f"Уже в результатах (пропущено при подборе): {skipped}")

            if total == 0:
                pm.clear_pending_run(name)
                _run_log.append("=== ГОТОВО ===")
                _run_log.append("Все офферы из фида уже в результатах. Новых к обработке нет.")
                yield "✅ Все уже обработаны", None, _run_log_text_for_ui(_run_log)
                return

            cache_dir = pm.image_cache_dir(name)
            global_settings = pm.get_global_settings()
            config_with_cache = pm.strip_legacy_prompt_from_config(
                {**global_settings, **_proj(), "image_cache_dir": str(cache_dir)}
            )
            lastp = pm.load_run_prompt_last()
            config_with_cache["task_instruction"] = (lastp.get("task_instruction") or "").strip()
            config_with_cache["task_constraints"] = (lastp.get("task_constraints") or "").strip()
            config_with_cache["task_examples"] = (lastp.get("task_examples") or "").strip()
            _apply_lastp_targets_to_config(config_with_cache, lastp)
            config_with_cache["dynamic_clothing_attributes"] = bool(
                config_with_cache.get("dynamic_clothing_attributes", True)
            )
            config_with_cache["use_full_prompt_edit"] = bool(lastp.get("use_full_prompt_edit"))
            config_with_cache["full_prompt_text"] = (lastp.get("full_prompt_text") or "").strip()
            model = global_settings.get("model", "?")
            ollama_url = global_settings.get("ollama_url", "?")
            max_size = global_settings.get("image_max_size", 1024)
            raw_bw_lp = int(global_settings.get("batch_offer_workers", 1) or 1)
            batch_workers_lp = max(1, min(16, raw_bw_lp))
            mpv_run_lp = int(global_settings.get("max_parallel_vision", 0) or 0)
            mpv_run_txt_lp = "без лимита (все задачи сразу)" if mpv_run_lp <= 0 else str(mpv_run_lp)

            _run_log.append("=== СТАРТ ОБРАБОТКИ ===")
            _run_log.append(f"Проект: {name}")
            _run_log.append(f"Категории: {cats_log}")
            _run_log.append(f"Офферов к обработке: {total}")
            _run_log.append(f"Модель: {model}  |  Ollama: {ollama_url}")
            _run_log.append(
                f"**Разных фото одновременно** (как в настройках): **{batch_workers_lp}** | "
                f"**Запросов к модели на одно фото**: **{mpv_run_txt_lp}**."
                + (
                    " _Этот упрощённый поток всегда по одному фото; пул ×N — в полном «Запуск»._"
                    if batch_workers_lp > 1
                    else ""
                )
            )
            _run_log.append("---")
            for _ln in _run_prompt_summary_for_log(config_with_cache):
                _run_log.append(_ln)
            _run_log.append("---")
            check_url = normalize_ollama_url(ollama_url or "")
            try:
                r = requests.get(
                    check_url.rstrip("/") + "/",
                    timeout=ollama_root_health_timeout_s(ollama_url or ""),
                )
                r.raise_for_status()
            except Exception:
                _run_log.append("⚠ Ollama не отвечает на " + check_url + ". Запустите приложение Ollama или перезапустите run.bat.")
                _run_log.append("")
            _run_log.append("Прогрев модели (первый запрос может занять до минуты)...")
            yield "Прогрев модели...", None, _run_log_text_for_ui(_run_log)
            try:
                _warmup_run_models(config_with_cache)
                _run_log.append("Прогрев выполнен (основная модель и при необходимости — модель надписей).")
            except Exception:
                _run_log.append("(прогрев не выполнен, продолжаем)")
            _run_log.append("")
            proj_run = _proj()
            dedupe_mode = dedupe_mode_from_config(proj_run)
            dedupe_infer = dedupe_mode in ("url", "phash")
            groups = (
                group_offers_by_picture_dedupe(
                    offers,
                    dedupe_mode,
                    Path(cache_dir),
                    int(max_size),
                    ensure_image_cached,
                    stop_event=_run_stop_event,
                )
                if dedupe_infer
                else [[o] for o in offers]
            )
            n_unique = len(groups)
            if dedupe_infer:
                how = "по URL" if dedupe_mode == "url" else "по dHash кэша (разные URL — один файл)"
                _run_log.append(
                    f"**Дедуп до vision ({how}):** уникальных **{n_unique}** при **{total}** офферах "
                    f"(повторный анализ не вызывается — копия результата по SKU)."
                )
            _mpv_con_lp = "без_лимита" if mpv_run_lp <= 0 else str(mpv_run_lp)
            print(
                f"\n[Запуск] проект={name}  к_обработке={total}  "
                f"пропущено_уже_в_результатах={skipped}  категории={cats_log}  модель={model}"
                + (f"  уникальных_картинок={n_unique}  dedupe={dedupe_mode}" if dedupe_infer else "")
                + f"  одновременно_разных_фото={batch_workers_lp}  запросов_к_модели_на_одно_фото={_mpv_con_lp}"
            )

            processed = 0
            errors = 0
            saved_ids: set[str] = set()

            def _emit_offer_log(offer: dict, n_disp: int, same_pic_copy: bool):
                oid = offer.get("offer_id", "")
                oname = (offer.get("name") or "")[:50]
                pic_url = (offer.get("picture_urls") or [None])[0]
                _run_log.append(f"[{n_disp}/{total}] Оффер {oid} — {oname}")
                _run_log.append(
                    f"      offer_id={oid}  picture_url={ (pic_url or '')[:100] }{'...' if (pic_url or '') and len(pic_url or '') > 100 else ''}"
                )
                if same_pic_copy:
                    _run_log.append("      → тот же URL фото, что у представителя группы — **vision не вызываем**")

            for group in groups:
                if _run_stop_event.is_set():
                    _run_log.append("")
                    _run_log.append("⏹ Остановлено пользователем.")
                    print("[Обработка] Остановлено пользователем")
                    break
                rep = group[0]
                rep_id = str(rep.get("offer_id", ""))
                pic_url = (rep.get("picture_urls") or [None])[0]
                img_path = ensure_image_cached(pic_url, cache_dir, max_size) if pic_url else None
                if img_path:
                    _run_log.append(f"      Картинка (представитель группы): путь: {Path(img_path).resolve()}")
                else:
                    _run_log.append("      Картинка: не загружена (представитель группы)")

                status_md = f"Обработано: **{processed}/{total}** | Ошибок: {errors}"
                yield status_md, str(img_path) if img_path else None, _run_log_text_for_ui(_run_log)

                try:
                    result = analyze_offer(rep, config_with_cache)
                    result["model"] = (config_with_cache.get("model") or "").strip()
                    for o in group:
                        if _run_stop_event.is_set():
                            break
                        oid = str(o.get("offer_id", ""))
                        n_disp = processed + 1
                        same_copy = oid != rep_id and dedupe_infer
                        _emit_offer_log(o, n_disp, same_copy)
                        pic_o = (o.get("picture_urls") or [None])[0]
                        img_o = ensure_image_cached(pic_o, cache_dir, max_size) if pic_o else None
                        to_save = _clone_analyze_result_for_offer(result, o) if oid != rep_id else result
                        to_save["model"] = result.get("model", "")
                        _save_result(rdb, to_save)
                        conf = to_save.get("avg_confidence", 0)
                        err = to_save.get("error")
                        if err:
                            errors += 1
                            _run_log.append(f"      Анализ: ошибка — {err[:100]}")
                            for _pl in _profile_run_log_lines(to_save):
                                _run_log.append(_pl)
                            print(f"  [{n_disp}/{total}] {oid}  Ошибка: {err[:60]}")
                        else:
                            _run_log.append(f"      Анализ: готово. Уверенность: {conf}%")
                            for _pl in _profile_run_log_lines(to_save):
                                _run_log.append(_pl)
                            _glossary = pm.load_attribute_glossary()
                            _key_to_label = _key_to_label_map(config_with_cache)
                            parts = []
                            for _dir_id, attrs in (to_save.get("direction_attributes") or {}).items():
                                if attrs and isinstance(attrs, dict):
                                    short = ", ".join(
                                        f"{_key_to_label.get(k, k)}={pm.translate_attribute_value(v.get('value', ''), _glossary)}"
                                        for k, v in attrs.items() if k != "error" and isinstance(v, dict)
                                    )
                                    if short:
                                        parts.append(short)
                            if parts:
                                _run_log.append("      Результат: " + " | ".join(parts)[:200])
                            print(
                                f"  [{n_disp}/{total}] {oid}  OK, уверенность {conf}%  "
                                f"{_console_result_line(to_save, config_with_cache)}"
                            )
                        saved_ids.add(oid)
                        processed += 1
                        _run_log.append("")
                        status_md = f"Обработано: **{processed}/{total}** | Ошибок: {errors}"
                        yield status_md, str(img_o) if img_o else None, _run_log_text_for_ui(_run_log)
                except Exception as e:
                    for o in group:
                        if _run_stop_event.is_set():
                            break
                        oid = str(o.get("offer_id", ""))
                        n_disp = processed + 1
                        _emit_offer_log(o, n_disp, False)
                        err_msg = str(e).strip()
                        if len(err_msg) > 100:
                            err_msg = err_msg[:97] + "..."
                        errors += 1
                        _run_log.append(f"      Анализ: ошибка — {err_msg}")
                        print(f"  [{n_disp}/{total}] {oid}  Ошибка: {err_msg[:60]}")
                        processed += 1
                        _run_log.append("")
                        pic_o = (o.get("picture_urls") or [None])[0]
                        img_o = ensure_image_cached(pic_o, cache_dir, max_size) if pic_o else None
                        status_md = f"Обработано: **{processed}/{total}** | Ошибок: {errors}"
                        yield status_md, str(img_o) if img_o else None, _run_log_text_for_ui(_run_log)

            if _run_stop_event.is_set():
                remaining = [str(o.get("offer_id", "")) for o in offers if o.get("offer_id") and str(o.get("offer_id")) not in saved_ids]
                if remaining:
                    pm.save_pending_run(name, fp, remaining)
                    _run_log.append(
                        f"💾 Очередь сохранена: осталось **{len(remaining)}** офферов. "
                        "С теми же лимитом/категориями следующий **Запуск** продолжит эту выборку."
                    )
            else:
                pm.clear_pending_run(name)

            _run_log.append("=== ГОТОВО ===")
            _run_log.append(f"Обработано офферов: {processed}. Ошибок: {errors}.")
            _run_log.append("Результаты смотрите на вкладке «Результаты».")
            done_msg = f"✅ Готово! Обработано: **{processed}** | Ошибок: **{errors}**"
            print(f"\n[Готово] Обработано: {processed}, ошибок: {errors}. Результаты — вкладка «Результаты».")
            yield done_msg, None, _run_log_text_for_ui(_run_log)

        def run_processing_thread(
            selected_cats,
            limit,
            all_feed,
            force_reprocess=False,
            task_instruction_override=None,
            task_constraints_run="",
            task_examples_run="",
            task_target_attribute_run="",
            use_full_prompt_edit=False,
            full_prompt_text_run="",
        ):
            """Фоновый поток: та же логика, что run_processing, но пишет состояние в run_state.json."""
            global _run_stop_event
            _run_stop_event.clear()
            name = _proj_name()
            if not name:
                _save_run_state({
                    "is_running": False,
                    "log": ["Ошибка: выберите проект на вкладке «Проекты»."],
                    "processed": 0, "errors": 0, "total": 0,
                    "status": "❌ Проект не выбран", "current_image": None,
                })
                return
            db = _cache_db()
            if not db or not db.exists():
                _save_run_state({
                    "is_running": False,
                    "log": ["Ошибка: загрузите фид на вкладке «Фид»."],
                    "processed": 0, "errors": 0, "total": 0,
                    "status": "❌ Кэш не загружен", "current_image": None,
                })
                return

            cats = None if all_feed else (selected_cats or None)
            cats_log = _format_categories_for_log(cats)
            limit_n = _coerce_run_limit(limit)
            rdb = pm.results_db_path(name)
            log_lines: list[str] = []
            if force_reprocess:
                log_lines.append("Режим перезаписи: все офферы будут обработаны заново.")
            offers, skipped, resumed, resume_msg, fp = _resolve_offers_for_run(
                name, db, cats, limit_n, bool(all_feed), force_reprocess, rdb
            )
            total = len(offers)
            if resume_msg:
                log_lines.append(resume_msg)
                log_lines.append(
                    "_Новая выборка по текущим полям:_ измените категории, лимит, «весь фид» или «перезапись» — "
                    "очередь сбросится; либо удалите **pending_run.json** в папке проекта."
                )
            if skipped and not resumed:
                log_lines.append(f"Уже в результатах (пропущено при подборе): {skipped}")

            cache_dir = pm.image_cache_dir(name)
            global_settings = pm.get_global_settings()
            config_with_cache = pm.strip_legacy_prompt_from_config(
                {**global_settings, **_proj(), "image_cache_dir": str(cache_dir)}
            )
            config_with_cache["task_instruction"] = (task_instruction_override or "").strip()
            config_with_cache["dynamic_clothing_attributes"] = bool(
                config_with_cache.get("dynamic_clothing_attributes", True)
            )
            config_with_cache["task_constraints"] = (task_constraints_run or "").strip()
            config_with_cache["task_examples"] = (task_examples_run or "").strip()
            _apply_lastp_targets_to_config(
                config_with_cache,
                {
                    "task_target_attribute": (task_target_attribute_run or "").strip(),
                    "task_target_attributes": [x.strip() for x in (task_target_attribute_run or "").split("\n") if x.strip()],
                },
            )
            config_with_cache["use_full_prompt_edit"] = bool(use_full_prompt_edit)
            config_with_cache["full_prompt_text"] = (full_prompt_text_run or "").strip()
            model = global_settings.get("model", "?")
            ollama_url = global_settings.get("ollama_url", "?")
            max_size = int(global_settings.get("image_max_size", 1024) or 1024)
            raw_bw = int(global_settings.get("batch_offer_workers", 1) or 1)
            batch_workers = max(1, min(16, raw_bw))
            mpv_run = int(global_settings.get("max_parallel_vision", 0) or 0)
            mpv_run_txt = "без лимита (все задачи сразу)" if mpv_run <= 0 else str(mpv_run)

            n_form_cats = len([x for x in (selected_cats or []) if str(x).strip()])
            lim_disp = limit_n if limit_n > 0 else "без лимита"
            form_line = (
                f"С формы: отмечено категорий={n_form_cats if not all_feed else '— (весь фид)'}, "
                f"лимит={lim_disp}, весь_фид={bool(all_feed)}, перезапись={bool(force_reprocess)}"
            )
            if resumed:
                form_line += f" | **сейчас берётся сохранённая очередь:** {total} офферов (не новый подбор по лимиту)"

            log_lines = log_lines + [
                "=== СТАРТ ОБРАБОТКИ ===",
                f"Проект: {name}",
                form_line,
                f"Категории (в подборе): {cats_log}",
                f"Офферов к обработке: {total}{' — продолжение очереди' if resumed else ''}",
                f"Модель: {model}  |  Ollama: {ollama_url}",
                f"**Разных фото одновременно** (ускорение фида): **{batch_workers}** | "
                f"**Запросов к модели на одно фото**: **{mpv_run_txt}**.",
            ]
            if config_with_cache.get("use_full_prompt_edit") and (config_with_cache.get("full_prompt_text") or "").strip():
                log_lines.append("Промпт: **отредактированный текст целиком** (не автосборка из полей).")
            log_lines.append("---")
            for _ln in _run_prompt_summary_for_log(config_with_cache):
                log_lines.append(_ln)
            log_lines.append("---")
            if total == 0:
                if not resumed:
                    pm.clear_pending_run(name)
                    log_lines.append("=== ГОТОВО ===")
                    log_lines.append("Все офферы из фида уже в результатах. Новых к обработке нет.")
                    _st = "✅ Все уже обработаны"
                else:
                    log_lines.append("=== ГОТОВО ===")
                    if resume_msg:
                        log_lines.append(resume_msg)
                    else:
                        log_lines.append("Очередь продолжения пуста.")
                    _st = "⚠ Нет офферов для продолжения (см. лог)"
                _save_run_state({
                    "is_running": False, "log": log_lines,
                    "processed": 0, "errors": 0, "total": 0,
                    "status": _st, "current_image": None,
                })
                return

            processed = 0
            errors = 0
            saved_ids: set[str] = set()

            _save_run_state({
                "is_running": True, "log": log_lines,
                "processed": 0, "errors": 0, "total": total,
                "status": "Прогрев модели...", "current_image": None,
            })

            try:
                check_url = normalize_ollama_url(ollama_url or "")
                try:
                    r = requests.get(
                        check_url.rstrip("/") + "/",
                        timeout=ollama_root_health_timeout_s(ollama_url or ""),
                    )
                    r.raise_for_status()
                except Exception:
                    log_lines.append("⚠ Ollama не отвечает. Запустите приложение Ollama или перезапустите run.bat.")
                    log_lines.append("")
                log_lines.append("Прогрев модели (первый запрос может занять до минуты)...")
                _save_run_state({
                    "is_running": True, "log": log_lines,
                    "processed": 0, "errors": 0, "total": total,
                    "status": "Прогрев модели...", "current_image": None,
                })

                try:
                    _warmup_run_models(config_with_cache)
                    log_lines.append("✅ Прогрев выполнен — модель в памяти.")
                except Exception as _wu_e:
                    log_lines.append(f"⚠ Прогрев не удался ({_wu_e}), продолжаем.")
                log_lines.append("")
                _save_run_state({
                    "is_running": True, "log": log_lines,
                    "processed": 0, "errors": 0, "total": total,
                    "status": "Запуск обработки…", "current_image": None,
                })

                sk_msg = skipped if not resumed else 0
                proj_run = _proj()
                dedupe_mode = dedupe_mode_from_config(proj_run)
                dedupe_infer = dedupe_mode in ("url", "phash")
                if dedupe_infer:
                    how = "URL" if dedupe_mode == "url" else "dHash кэша"
                    log_lines.append(f"Дедуп картинок ({how}): подготовка групп…")
                    _save_run_state({
                        "is_running": True, "log": log_lines,
                        "processed": 0, "errors": 0, "total": total,
                        "status": f"Дедуп ({how})…", "current_image": None,
                    })
                groups = (
                    group_offers_by_picture_dedupe(
                        offers,
                        dedupe_mode,
                        Path(cache_dir),
                        int(max_size),
                        ensure_image_cached,
                        stop_event=_run_stop_event,
                    )
                    if dedupe_infer
                    else [[o] for o in offers]
                )
                n_unique = len(groups)
                if dedupe_infer:
                    log_lines.append(
                        f"**Дедуп до vision ({how}):** уникальных **{n_unique}** при **{total}** офферах."
                    )
                    if _run_stop_event.is_set():
                        log_lines.append(
                            "⏹ Дедуп **прерван** по «Остановить»: хвост списка идёт **по одному офферу в группе** "
                            "(без слияния по картинке для ещё не сгруппированных)."
                        )
                    _save_run_state({
                        "is_running": True, "log": log_lines,
                        "processed": 0, "errors": 0, "total": total,
                        "status": f"Обработка {total} офферов, {n_unique} уникальных…", "current_image": None,
                    })
                _mpv_con = "без_лимита" if mpv_run <= 0 else str(mpv_run)
                print(
                    f"\n[Запуск] проект={name}  к_обработке={total}  "
                    f"пропущено_уже_в_результатах={sk_msg}  категории={cats_log}  модель={model}"
                    + (f"  продолжение_очереди={'да' if resumed else 'нет'}")
                    + (f"  уникальных_картинок={n_unique}  dedupe={dedupe_mode}" if dedupe_infer else "")
                    + f"  одновременно_разных_фото={batch_workers}  запросов_к_модели_на_одно_фото={_mpv_con}"
                )

                if batch_workers <= 1:

                    def _emit_thr(offer: dict, n_disp: int, same_pic_copy: bool):
                        oid = offer.get("offer_id", "")
                        oname = (offer.get("name") or "")[:50]
                        pic_url = (offer.get("picture_urls") or [None])[0]
                        log_lines.append(f"[{n_disp}/{total}] Оффер {oid} — {oname}")
                        log_lines.append(
                            f"      offer_id={oid}  picture_url={ (pic_url or '')[:100] }{'...' if (pic_url or '') and len(pic_url or '') > 100 else ''}"
                        )
                        if same_pic_copy:
                            log_lines.append("      → тот же URL фото — **vision не вызываем**")

                    for group in groups:
                        if _run_stop_event.is_set():
                            log_lines.append("")
                            log_lines.append("⏹ Остановлено пользователем.")
                            print("[Обработка] Остановлено пользователем")
                            break
                        rep = group[0]
                        rep_id = str(rep.get("offer_id", ""))
                        pic_url = (rep.get("picture_urls") or [None])[0]
                        img_path = ensure_image_cached(pic_url, cache_dir, max_size) if pic_url else None
                        if img_path:
                            log_lines.append(f"      Картинка (представитель группы): путь: {Path(img_path).resolve()}")
                        else:
                            log_lines.append("      Картинка: не загружена (представитель группы)")

                        status_md = f"Обработано: **{processed}/{total}** | Ошибок: {errors}"
                        _save_run_state({
                            "is_running": True, "log": log_lines,
                            "processed": processed, "errors": errors, "total": total,
                            "status": status_md, "current_image": str(Path(img_path).resolve()) if img_path else None,
                        })

                        try:
                            result = analyze_offer(rep, config_with_cache)
                            result["model"] = (config_with_cache.get("model") or "").strip()
                            for o in group:
                                if _run_stop_event.is_set():
                                    break
                                oid = str(o.get("offer_id", ""))
                                n_disp = processed + 1
                                same_copy = oid != rep_id and dedupe_infer
                                _emit_thr(o, n_disp, same_copy)
                                pic_o = (o.get("picture_urls") or [None])[0]
                                img_o = ensure_image_cached(pic_o, cache_dir, max_size) if pic_o else None
                                to_save = _clone_analyze_result_for_offer(result, o) if oid != rep_id else result
                                to_save["model"] = result.get("model", "")
                                _save_result(rdb, to_save)
                                conf = to_save.get("avg_confidence", 0)
                                err = to_save.get("error")
                                if err:
                                    errors += 1
                                    log_lines.append(f"      Анализ: ошибка — {err[:100]}")
                                    for _pl in _profile_run_log_lines(to_save):
                                        log_lines.append(_pl)
                                    print(f"  [{n_disp}/{total}] {oid}  Ошибка: {err[:60]}")
                                else:
                                    log_lines.append(f"      Анализ: готово. Уверенность: {conf}%")
                                    for _pl in _profile_run_log_lines(to_save):
                                        log_lines.append(_pl)
                                    _glossary = pm.load_attribute_glossary()
                                    _key_to_label = _key_to_label_map(config_with_cache)
                                    parts = []
                                    for _dir_id, attrs in (to_save.get("direction_attributes") or {}).items():
                                        if attrs and isinstance(attrs, dict):
                                            short = ", ".join(
                                                f"{_key_to_label.get(k, k)}={pm.translate_attribute_value(v.get('value', ''), _glossary)}"
                                                for k, v in attrs.items() if k != "error" and isinstance(v, dict)
                                            )
                                            if short:
                                                parts.append(short)
                                    if parts:
                                        log_lines.append("      Результат: " + " | ".join(parts)[:200])
                                    print(
                                        f"  [{n_disp}/{total}] {oid}  OK, уверенность {conf}%  "
                                        f"{_console_result_line(to_save, config_with_cache)}"
                                    )
                                saved_ids.add(oid)
                                processed += 1
                                log_lines.append("")
                                status_md = f"Обработано: **{processed}/{total}** | Ошибок: {errors}"
                                _save_run_state({
                                    "is_running": True, "log": log_lines,
                                    "processed": processed, "errors": errors, "total": total,
                                    "status": status_md,
                                    "current_image": str(Path(img_o).resolve()) if img_o else None,
                                })
                        except Exception as e:
                            for o in group:
                                if _run_stop_event.is_set():
                                    break
                                oid = str(o.get("offer_id", ""))
                                n_disp = processed + 1
                                _emit_thr(o, n_disp, False)
                                err_msg = str(e).strip()
                                if len(err_msg) > 100:
                                    err_msg = err_msg[:97] + "..."
                                errors += 1
                                log_lines.append(f"      Анализ: ошибка — {err_msg}")
                                print(f"  [{n_disp}/{total}] {oid}  Ошибка: {err_msg[:60]}")
                                processed += 1
                                log_lines.append("")
                                pic_o = (o.get("picture_urls") or [None])[0]
                                img_o = ensure_image_cached(pic_o, cache_dir, max_size) if pic_o else None
                                status_md = f"Обработано: **{processed}/{total}** | Ошибок: {errors}"
                                _save_run_state({
                                    "is_running": True, "log": log_lines,
                                    "processed": processed, "errors": errors, "total": total,
                                    "status": status_md,
                                    "current_image": str(Path(img_o).resolve()) if img_o else None,
                                })
                else:
                    _seq_lock = threading.Lock()
                    _seq_n = [0]

                    def _next_display():
                        with _seq_lock:
                            _seq_n[0] += 1
                            return _seq_n[0]

                    def _group_job(grp):
                        try:
                            return _run_single_group_batch(
                                grp,
                                total=total,
                                rdb=rdb,
                                cache_dir=cache_dir,
                                max_size=max_size,
                                dedupe_infer=dedupe_infer,
                                config_with_cache=config_with_cache,
                                stop_event=_run_stop_event,
                                next_offer_display_num=_next_display,
                            )
                        except Exception as _job_e:
                            msg = str(_job_e).strip()
                            if len(msg) > 120:
                                msg = msg[:117] + "..."
                            return (
                                [f"      ❌ Ошибка группы: {msg}"],
                                0,
                                0,
                                set(),
                                None,
                            )

                    _merge_lock = threading.Lock()
                    _exec_futures: list = []
                    _mpv_pool = "без лимита" if mpv_run <= 0 else str(mpv_run)
                    log_lines.append(
                        f"**Пул ×{batch_workers}:** столько **разных фото** считается **одновременно** (примерно в столько раз быстрее фид, если VRAM хватает). "
                        f"На **каждое** фото отдельно к той же модели может идти до **{_mpv_pool}** параллельных запросов (второе поле в настройках)."
                    )
                    log_lines.append(
                        "_Подсказка:_ консоль `[Пул] …`; детальный лог — `IMAGE_DESC_POOL_VERBOSE=1`."
                    )
                    print(
                        f"\n[Пул] одновременно_разных_фото={batch_workers}  запросов_на_одно_фото={_mpv_pool}",
                        flush=True,
                    )
                    executor = ThreadPoolExecutor(max_workers=batch_workers)
                    try:
                        for group in groups:
                            if _run_stop_event.is_set():
                                log_lines.append("")
                                log_lines.append("⏹ Остановлено пользователем.")
                                print("[Обработка] Остановлено пользователем")
                                break
                            _exec_futures.append(executor.submit(_group_job, group))
                        for fut in as_completed(_exec_futures):
                            if _run_stop_event.is_set():
                                for f in _exec_futures:
                                    if not f.done():
                                        f.cancel()
                            try:
                                chunk, pi, ei, sv, cur_img = fut.result()
                            except CancelledError:
                                continue
                            except Exception as _fe:
                                chunk = [f"      ❌ Сбой задачи: {_fe}"]
                                pi = 0
                                ei = 0
                                sv = set()
                                cur_img = None
                            with _merge_lock:
                                log_lines.extend(chunk)
                                processed += pi
                                errors += ei
                                saved_ids |= sv
                                status_md = _run_progress_status_md(
                                    processed, total, errors, batch_workers=batch_workers
                                )
                                _save_run_state({
                                    "is_running": True, "log": log_lines,
                                    "processed": processed, "errors": errors, "total": total,
                                    "status": status_md,
                                    "current_image": cur_img,
                                    "batch_offer_workers": batch_workers,
                                })
                    finally:
                        executor.shutdown(wait=False, cancel_futures=True)

                if _run_stop_event.is_set():
                    remaining = [str(o.get("offer_id", "")) for o in offers if o.get("offer_id") and str(o.get("offer_id")) not in saved_ids]
                    if remaining:
                        pm.save_pending_run(name, fp, remaining)
                        log_lines.append(
                            f"💾 Очередь сохранена: осталось {len(remaining)} офферов — следующий запуск с теми же параметрами продолжит."
                        )
                else:
                    pm.clear_pending_run(name)

                stopped_by_user = _run_stop_event.is_set()
                _pool_done = f" | ×{batch_workers} фото параллельно" if batch_workers > 1 else ""
                if stopped_by_user:
                    log_lines.append("=== ОСТАНОВЛЕНО ===")
                    log_lines.append(f"Обработано офферов: {processed}. Ошибок: {errors}.")
                    log_lines.append(
                        "Уже **запущенные** вызовы модели могут ещё завершиться на стороне Ollama (1–N групп); "
                        "новые группы не ставятся в очередь."
                    )
                    log_lines.append("Результаты смотрите на вкладке «Результаты».")
                    done_msg = f"⏹ Остановлено. Обработано: **{processed}** | Ошибок: **{errors}**{_pool_done}"
                    print(f"\n[Остановлено] Обработано: {processed}, ошибок: {errors}. Результаты — вкладка «Результаты».")
                else:
                    log_lines.append("=== ГОТОВО ===")
                    log_lines.append(f"Обработано офферов: {processed}. Ошибок: {errors}.")
                    log_lines.append("Результаты смотрите на вкладке «Результаты».")
                    done_msg = f"✅ Готово! Обработано: **{processed}** | Ошибок: **{errors}**{_pool_done}"
                    print(f"\n[Готово] Обработано: {processed}, ошибок: {errors}. Результаты — вкладка «Результаты».")
                _done_state = {
                    "is_running": False, "log": log_lines,
                    "processed": processed, "errors": errors, "total": total,
                    "status": done_msg, "current_image": None,
                }
                if batch_workers > 1:
                    _done_state["batch_offer_workers"] = batch_workers
                _save_run_state(_done_state)

            except Exception as _run_exc:
                log_lines.append(f"❌ Ошибка обработки: {_run_exc}")
                for _tl in traceback.format_exc().splitlines()[-14:]:
                    log_lines.append(_tl)
                _save_run_state({
                    "is_running": False, "log": log_lines,
                    "processed": processed, "errors": errors, "total": total,
                    "status": f"❌ Ошибка: {_run_exc}", "current_image": None,
                })
                print("[Обработка] Фатальная ошибка:", _run_exc)
                return

        def start_run(
            selected_cats,
            limit,
            all_feed,
            force_reprocess,
            task_instruction_from_run,
            task_constraints_from_run,
            task_examples_from_run,
            ta0,
            ta1,
            ta2,
            ta3,
            ta4,
            use_full_prompt_from_run,
            full_prompt_text_from_run,
            block_if_pool_busy,
        ):
            global _run_worker_thread
            if block_if_pool_busy:
                _pu = pm.get_global_settings().get("ollama_url", pm.GLOBAL_DEFAULTS["ollama_url"])
                if pool_jobs_client.pool_backlog_severe(_pu):
                    return (
                        "⚠️ Пул Ollama выглядит перегруженным (много ожидающих запросов или jobs). "
                        "Подождите или снимите галку «Не запускать обработку…».",
                        None,
                        "",
                        gr.update(active=True),
                    )
            packed = _pack_run_target_attrs(ta0, ta1, ta2, ta3, ta4)
            ta_joined = "\n".join(packed)
            pm.save_run_prompt_last(
                task_instruction_from_run or "",
                task_constraints_from_run or "",
                task_examples_from_run or "",
                ta_joined,
                bool(use_full_prompt_from_run),
                full_prompt_text_from_run or "",
                task_target_attributes=packed,
            )
            with _run_start_lock:
                state = _load_run_state() or {}
                alive = _run_worker_thread is not None and _run_worker_thread.is_alive()
                if alive:
                    return (
                        "⏳ Уже идёт обработка (в т.ч. дедуп большого фида или запросы к Ollama). "
                        "Лог обновляется по таймеру; **Остановить** прерывает дедуп и новые группы. "
                        "Повторный «Запустить» возможен после завершения потока.",
                        None, "", gr.update(active=True),
                    )
                if state.get("is_running") and not alive:
                    # Процесс перезапустили или поток упал — в JSON остался «идёт обработка».
                    _save_run_state(
                        {
                            "is_running": False,
                            "log": state.get("log") or [],
                            "processed": state.get("processed", 0),
                            "errors": state.get("errors", 0),
                            "total": state.get("total", 0),
                            "status": "⚠️ Сброшен зависший статус (поток не активен). Можно снова нажать «Запустить».",
                            "current_image": state.get("current_image"),
                        }
                    )
                t = threading.Thread(
                    target=run_processing_thread,
                    args=(
                        selected_cats,
                        limit,
                        all_feed,
                        force_reprocess,
                        (task_instruction_from_run or "").strip(),
                        (task_constraints_from_run or "").strip(),
                        (task_examples_from_run or "").strip(),
                        ta_joined,
                        bool(use_full_prompt_from_run),
                        (full_prompt_text_from_run or "").strip(),
                    ),
                    daemon=True,
                )
                _run_worker_thread = t
                t.start()
            _g0 = pm.get_global_settings()
            _bw0 = max(1, min(16, int(_g0.get("batch_offer_workers", 1) or 1)))
            _mpv0 = int(_g0.get("max_parallel_vision", 0) or 0)
            _mpv0_txt = "без лимита" if _mpv0 <= 0 else str(_mpv0)
            return (
                "🚀 **Запущено в фоне.** Лог и картинка обновляются каждые 2 секунды. "
                f"Параллельно уникальных картинок: **{_bw0}**, лимит vision внутри оффера: **{_mpv0_txt}**.",
                None,
                "Ожидание первого обновления…",
                gr.update(active=True),
            )

        def refresh_run_status(_tick_value=None):
            """Обновляет статус/картинку/лог. Возвращает также active-флаг таймера."""
            state = _load_run_state()
            if not state:
                return "_Ожидание_", None, "", gr.update(active=True)
            is_running = bool(state.get("is_running"))
            alive = _run_worker_thread is not None and _run_worker_thread.is_alive()
            if is_running and not alive:
                note = "⚠️ В JSON остался «идёт обработка», но поток не жив (перезапуск приложения или сбой) — сброшено."
                log = list(state.get("log") or [])
                log.append(note)
                state = {
                    "is_running": False,
                    "log": log,
                    "processed": state.get("processed", 0),
                    "errors": state.get("errors", 0),
                    "total": state.get("total", 0),
                    "status": note,
                    "current_image": state.get("current_image"),
                }
                _save_run_state(state)
                is_running = False
            status = state.get("status") or "_Ожидание_"
            current_image = state.get("current_image")
            log = state.get("log") or []
            log_text = _run_log_text_for_ui(list(log))
            # когда обработка завершена — уменьшаем частоту до 10 сек (не выключаем совсем,
            # чтобы статус отразился после перезапуска процесса)
            timer_update = gr.update(active=True) if is_running else gr.update(active=False)
            return status, current_image, log_text, timer_update

        def stop_run():
            _run_stop_event.set()
            return (
                "⏹ Запрос на остановку отправлен. Дедуп и постановка новых групп в пул остановятся; "
                "уже идущие запросы к Ollama дождутся конца сами."
            )

        def refresh_ollama_run_tab():
            _url = pm.get_global_settings().get("ollama_url", pm.GLOBAL_DEFAULTS["ollama_url"])
            _hint = _run_model_hint_text()
            out = [
                _ollama_status_html(),
                format_ollama_pool_status_html(_url),
                gr.update(value=_hint, visible=bool(_hint)),
            ]
            if header_model_badge is not None:
                out.append(_get_model_in_memory_badge())
            return out

        _check_outputs = [ollama_status_html, ollama_pool_status_html, run_model_hint]
        if header_model_badge is not None:
            _check_outputs.append(header_model_badge)
        btn_check_ollama.click(
            refresh_ollama_run_tab,
            outputs=_check_outputs,
        )
        btn_load_cats.click(load_categories, inputs=[categories_check], outputs=categories_check)
        cat_search_box.change(filter_categories, inputs=[cat_search_box, categories_check], outputs=categories_check)
        btn_limit_10.click(lambda: 10, outputs=limit_box)
        btn_limit_50.click(lambda: 50, outputs=limit_box)
        btn_limit_100.click(lambda: 100, outputs=limit_box)
        btn_limit_500.click(lambda: 500, outputs=limit_box)
        btn_limit_0.click(lambda: 0, outputs=limit_box)

        def _ta_field_updates_from_lines(lines: list[str], n: int):
            lines = (list(lines) + [""] * 5)[:5]
            n = max(1, min(5, int(n)))
            return (
                gr.update(value=lines[0]),
                gr.update(value=lines[1], visible=n >= 2),
                gr.update(value=lines[2], visible=n >= 3),
                gr.update(value=lines[3], visible=n >= 4),
                gr.update(value=lines[4], visible=n >= 5),
                n,
            )

        def _ta_add_row(n, a0, a1, a2, a3, a4):
            n2 = min(5, int(n) + 1)
            return _ta_field_updates_from_lines([a0, a1, a2, a3, a4], n2)

        def _ta_remove_row(n, a0, a1, a2, a3, a4):
            n2 = max(1, int(n) - 1)
            return _ta_field_updates_from_lines([a0, a1, a2, a3, a4], n2)

        btn_ta_add.click(
            _ta_add_row,
            inputs=[run_ta_visible_n, run_task_target_attr_0, run_task_target_attr_1, run_task_target_attr_2, run_task_target_attr_3, run_task_target_attr_4],
            outputs=[
                run_task_target_attr_0,
                run_task_target_attr_1,
                run_task_target_attr_2,
                run_task_target_attr_3,
                run_task_target_attr_4,
                run_ta_visible_n,
            ],
        )
        btn_ta_remove.click(
            _ta_remove_row,
            inputs=[run_ta_visible_n, run_task_target_attr_0, run_task_target_attr_1, run_task_target_attr_2, run_task_target_attr_3, run_task_target_attr_4],
            outputs=[
                run_task_target_attr_0,
                run_task_target_attr_1,
                run_task_target_attr_2,
                run_task_target_attr_3,
                run_task_target_attr_4,
                run_ta_visible_n,
            ],
        )

        def refresh_preset_dropdown():
            ch = _run_preset_choices()
            return gr.update(choices=ch, value=(ch[0] if len(ch) == 1 else None))

        btn_refresh_preset_list.click(refresh_preset_dropdown, outputs=[run_prompt_preset])

        def _lines_n_from_task_string(s: str) -> tuple[list[str], int]:
            parts = [x.strip() for x in (s or "").split("\n")]
            nonempty = [x for x in parts if x]
            lines = nonempty + [""] * 5
            lines = lines[:5]
            n = max(1, min(5, len(nonempty) or 1))
            return lines, n

        def _preset_fields_from_last_expanded():
            last = pm.load_run_prompt_last()
            lines, n = _lines_n_from_task_string(last.get("task_target_attribute") or "")
            return (
                last.get("task_instruction") or "",
                last.get("task_constraints") or "",
                last.get("task_examples") or "",
                lines[0],
                lines[1],
                lines[2],
                lines[3],
                lines[4],
                bool(last.get("use_full_prompt_edit")),
                last.get("full_prompt_text") or "",
                n,
            )

        def on_preset_dropdown_change(preset_name):
            """Выбор пресета — подставить значения и скрыть редактор; сброс — показать поля из последнего сохранения."""
            proj = _proj_name()
            if not proj:
                ti, tc, te, l0, l1, l2, l3, l4, ufp, fpt, n = _preset_fields_from_last_expanded()
                return (
                    gr.update(visible=True),
                    ti,
                    tc,
                    te,
                    gr.update(value=l0),
                    gr.update(value=l1, visible=n >= 2),
                    gr.update(value=l2, visible=n >= 3),
                    gr.update(value=l3, visible=n >= 4),
                    gr.update(value=l4, visible=n >= 5),
                    ufp,
                    fpt,
                    gr.update(value=""),
                    "⚠ Сначала выберите проект на вкладке «Проекты».",
                    n,
                )
            if not preset_name:
                ti, tc, te, l0, l1, l2, l3, l4, ufp, fpt, n = _preset_fields_from_last_expanded()
                return (
                    gr.update(visible=True),
                    ti,
                    tc,
                    te,
                    gr.update(value=l0),
                    gr.update(value=l1, visible=n >= 2),
                    gr.update(value=l2, visible=n >= 3),
                    gr.update(value=l3, visible=n >= 4),
                    gr.update(value=l4, visible=n >= 5),
                    ufp,
                    fpt,
                    gr.update(value=""),
                    "_Выберите пресет или нажмите «Новый пресет»._",
                    n,
                )
            for t in pm.load_project_prompt_presets(proj):
                if t.get("name") == preset_name:
                    lines, n = _lines_n_from_task_string(t.get("task_target_attribute") or "")
                    return (
                        gr.update(visible=False),
                        t.get("instruction") or "",
                        t.get("task_constraints") or "",
                        t.get("task_examples") or "",
                        gr.update(value=lines[0]),
                        gr.update(value=lines[1], visible=n >= 2),
                        gr.update(value=lines[2], visible=n >= 3),
                        gr.update(value=lines[3], visible=n >= 4),
                        gr.update(value=lines[4], visible=n >= 5),
                        bool(t.get("use_full_prompt_edit")),
                        t.get("full_prompt_text") or "",
                        gr.update(value=preset_name),
                        f"✅ Активен пресет **{preset_name}** — поля скрыты. **Изменить пресет** — чтобы править и перезаписать.",
                        n,
                    )
            ti, tc, te, l0, l1, l2, l3, l4, ufp, fpt, n = _preset_fields_from_last_expanded()
            return (
                gr.update(visible=True),
                ti,
                tc,
                te,
                gr.update(value=l0),
                gr.update(value=l1, visible=n >= 2),
                gr.update(value=l2, visible=n >= 3),
                gr.update(value=l3, visible=n >= 4),
                gr.update(value=l4, visible=n >= 5),
                ufp,
                fpt,
                gr.update(value=""),
                "⚠ Пресет не найден в проекте.",
                n,
            )

        run_prompt_preset.change(
            on_preset_dropdown_change,
            inputs=[run_prompt_preset],
            outputs=[
                prompt_main_group,
                run_task_instruction,
                run_task_constraints,
                run_task_examples,
                run_task_target_attr_0,
                run_task_target_attr_1,
                run_task_target_attr_2,
                run_task_target_attr_3,
                run_task_target_attr_4,
                run_use_full_prompt,
                run_full_prompt_text,
                preset_save_name,
                run_prefs_saved,
                run_ta_visible_n,
            ],
        )

        def show_preset_editor(preset_name):
            """Открыть редактор и подставить имя выбранного пресета — «Сохранить как пресет» перезапишет его."""
            return gr.update(visible=True), gr.update(value=(preset_name or ""))

        btn_edit_preset.click(
            show_preset_editor,
            inputs=[run_prompt_preset],
            outputs=[prompt_main_group, preset_save_name],
        )

        def new_preset_flow():
            return (
                gr.update(visible=True),
                gr.update(value=None),
                "",
                "",
                "",
                gr.update(value=""),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                False,
                "",
                gr.update(value=""),
                "Новый пресет: заполните поля и **Сохранить как пресет**.",
                1,
            )

        btn_new_preset.click(
            new_preset_flow,
            outputs=[
                prompt_main_group,
                run_prompt_preset,
                run_task_instruction,
                run_task_constraints,
                run_task_examples,
                run_task_target_attr_0,
                run_task_target_attr_1,
                run_task_target_attr_2,
                run_task_target_attr_3,
                run_task_target_attr_4,
                run_use_full_prompt,
                run_full_prompt_text,
                preset_save_name,
                run_prefs_saved,
                run_ta_visible_n,
            ],
        )

        def reset_clothing_only_mode():
            pm.save_run_prompt_last("", "", "", "", False, "", task_target_attributes=[])
            ch = _run_preset_choices()
            u0, u1, u2, u3, u4, n = _ta_field_updates_from_lines(["", "", "", "", ""], 1)
            return (
                gr.update(visible=True),
                gr.update(choices=ch, value=None),
                "",
                "",
                "",
                u0,
                u1,
                u2,
                u3,
                u4,
                False,
                "",
                gr.update(value=""),
                "✅ **Только одежда:** задание, строки атрибутов и пресет сброшены (`run_prompt_last.json`). "
                "**Запустить** дальше опирается на направления из **Настроек** проекта.",
                n,
            )

        btn_reset_clothing_run.click(
            reset_clothing_only_mode,
            outputs=[
                prompt_main_group,
                run_prompt_preset,
                run_task_instruction,
                run_task_constraints,
                run_task_examples,
                run_task_target_attr_0,
                run_task_target_attr_1,
                run_task_target_attr_2,
                run_task_target_attr_3,
                run_task_target_attr_4,
                run_use_full_prompt,
                run_full_prompt_text,
                preset_save_name,
                run_prefs_saved,
                run_ta_visible_n,
            ],
        )

        def save_current_as_preset(preset_name, ti, tc, te, ta0, ta1, ta2, ta3, ta4, use_fp, fp_txt):
            proj = _proj_name()
            if not proj:
                return "⚠ Выберите проект на вкладке «Проекты».", gr.update(), gr.update()
            n = (preset_name or "").strip()
            if not n:
                return "⚠ Введите имя пресета.", gr.update(), gr.update()
            dyn = bool(_proj().get("dynamic_clothing_attributes", True))
            packed = _pack_run_target_attrs(ta0, ta1, ta2, ta3, ta4)
            ta_joined = "\n".join(packed)
            pm.add_project_prompt_preset(
                proj,
                n,
                instruction=ti or "",
                task_constraints=tc or "",
                task_examples=te or "",
                task_target_attribute=ta_joined,
                full_prompt_text=fp_txt or "",
                use_full_prompt_edit=bool(use_fp),
                dynamic_clothing_attributes=dyn,
            )
            ch = _run_preset_choices()
            return (
                f"✅ Пресет **{n}** записан в проекте **{proj}** (`prompt_presets.json`) — новый или обновлённый по этому имени.",
                gr.update(choices=ch, value=n),
                gr.update(visible=False),
            )

        btn_save_preset.click(
            save_current_as_preset,
            inputs=[
                preset_save_name,
                run_task_instruction,
                run_task_constraints,
                run_task_examples,
                run_task_target_attr_0,
                run_task_target_attr_1,
                run_task_target_attr_2,
                run_task_target_attr_3,
                run_task_target_attr_4,
                run_use_full_prompt,
                run_full_prompt_text,
            ],
            outputs=[run_prefs_saved, run_prompt_preset, prompt_main_group],
        )

        def delete_selected_preset(preset_name):
            proj = _proj_name()
            if not proj:
                return "⚠ Выберите проект.", gr.update()
            if not preset_name:
                return "⚠ Выберите пресет в списке.", gr.update()
            tid = None
            for t in pm.load_project_prompt_presets(proj):
                if t.get("name") == preset_name:
                    tid = t.get("id")
                    break
            if not tid:
                return "⚠ Пресет не найден.", gr.update()
            pm.delete_project_prompt_preset(proj, tid)
            ch = _run_preset_choices()
            return f"✅ Пресет **{preset_name}** удалён.", gr.update(choices=ch, value=(ch[0] if ch else None))

        btn_delete_preset.click(
            delete_selected_preset,
            inputs=[run_prompt_preset],
            outputs=[run_prefs_saved, run_prompt_preset],
        )

        def remember_fields_only(ti, tc, te, ta0, ta1, ta2, ta3, ta4, use_fp, fp_txt):
            packed = _pack_run_target_attrs(ta0, ta1, ta2, ta3, ta4)
            ta_joined = "\n".join(packed)
            pm.save_run_prompt_last(
                ti, tc, te, ta_joined, bool(use_fp), fp_txt or "", task_target_attributes=packed
            )
            return "✅ Поля сохранены — при следующем открытии вкладки подставятся автоматически."

        def compose_prompt_from_fields(ti, tc, te, ta0, ta1, ta2, ta3, ta4):
            proj = _proj()
            v = (proj.get("vertical") or "").strip()
            dirs = proj.get("directions") or []
            dname = (dirs[0].get("name") or "") if dirs else ""
            attrs = _pack_run_target_attrs(ta0, ta1, ta2, ta3, ta4)
            rkeys: list[str] = []
            try:
                from attribute_detector import prepare_visual_analysis_plan, resolve_attributes_for_prompt

                cfg_prev = {
                    **proj,
                    "task_instruction": (ti or "").strip(),
                    "task_target_attributes": attrs,
                    "task_target_attribute": "\n".join(attrs),
                    "use_full_prompt_edit": False,
                    "full_prompt_text": "",
                }
                plan = prepare_visual_analysis_plan(cfg_prev)
                use_pt = bool((ti or "").strip() and plan["vertical"] != "Одежда")
                for d in plan["directions"]:
                    ba = d.get("attributes") or []
                    custom = (d.get("custom_prompt") or "").strip() or ((ti or "").strip() if use_pt else "")
                    for a in resolve_attributes_for_prompt(
                        ba, plan["task_target_list"], custom, ""
                    ):
                        k = a.get("key")
                        if k:
                            rkeys.append(k)
            except Exception:
                pass
            body = compose_task_prompt_blocks(
                (ti or "").strip(),
                vertical=v or None,
                direction_name=dname,
                product_name="{product_name}",
                user_constraints=(tc or "").strip(),
                user_examples=(te or "").strip(),
                target_attribute="",
                target_attributes=attrs if attrs else None,
                required_json_keys=sorted(set(rkeys)) or None,
            )
            if not (body or "").strip():
                return (
                    gr.update(),
                    "⚠ Заполните поле «Задание».",
                )
            return (
                gr.update(value=body),
                "✅ Предпросмотр подставлен. На **Запустить** по-прежнему влияют поля выше, пока в «Дополнительно» не включена подмена целого текста.",
            )

        btn_save_last_only.click(
            remember_fields_only,
            inputs=[
                run_task_instruction,
                run_task_constraints,
                run_task_examples,
                run_task_target_attr_0,
                run_task_target_attr_1,
                run_task_target_attr_2,
                run_task_target_attr_3,
                run_task_target_attr_4,
                run_use_full_prompt,
                run_full_prompt_text,
            ],
            outputs=[run_prefs_saved],
        )
        btn_compose_prompt.click(
            compose_prompt_from_fields,
            inputs=[
                run_task_instruction,
                run_task_constraints,
                run_task_examples,
                run_task_target_attr_0,
                run_task_target_attr_1,
                run_task_target_attr_2,
                run_task_target_attr_3,
                run_task_target_attr_4,
            ],
            outputs=[run_full_prompt_text, run_prefs_saved],
        )
        btn_start.click(
            start_run,
            inputs=[
                categories_check,
                limit_box,
                process_all,
                force_reprocess,
                run_task_instruction,
                run_task_constraints,
                run_task_examples,
                run_task_target_attr_0,
                run_task_target_attr_1,
                run_task_target_attr_2,
                run_task_target_attr_3,
                run_task_target_attr_4,
                run_use_full_prompt,
                run_full_prompt_text,
                run_pool_busy_block,
            ],
            outputs=[run_status, run_current_image, run_progress, run_status_timer],
        )
        btn_refresh_status.click(
            refresh_run_status,
            outputs=[run_status, run_current_image, run_progress, run_status_timer],
        )
        run_status_timer.tick(
            refresh_run_status,
            outputs=[run_status, run_current_image, run_progress, run_status_timer],
        )
        btn_stop.click(stop_run, outputs=run_status)

        btn_model_bench.click(
            start_model_bench,
            inputs=[model_bench_pick, model_bench_n],
            outputs=[model_bench_status],
        )
        model_bench_timer = gr.Timer(2, active=True)
        model_bench_timer.tick(poll_model_bench, outputs=[model_bench_status])

        return ollama_status_html, ollama_pool_status_html, run_model_hint


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Results
# ═══════════════════════════════════════════════════════════════════════════════

def tab_results():
    with gr.Tab("Результаты") as results_tab:
        gr.Markdown("## Результаты обработки")
        with gr.Row():
            conf_min = gr.Slider(0, 100, value=0, step=5, label="Уверенность от (%)", interactive=True)
            conf_max = gr.Slider(0, 100, value=100, step=5, label="Уверенность до (%)", interactive=True)
            cat_filter = gr.Dropdown(label="Категория", choices=["Все"], value="Все")
            results_model_filter = gr.Dropdown(
                label="Модель",
                choices=["Все"],
                value="Все",
                allow_custom_value=False,
                info="По полю **model_name** в сохранённых результатах. **(не указано)** — старые строки без модели.",
            )
            btn_refresh_results = gr.Button("Обновить", variant="primary")
            btn_export_csv = gr.Button("Выгрузить в CSV", variant="secondary")
            btn_export_csv_light = gr.Button("Выгрузить узкий CSV", variant="secondary")
        hide_dup_photos = gr.Checkbox(
            label="Скрыть дубли по URL картинки (нормализованный URL без query; одна карточка — макс. уверенность)",
            value=False,
        )
        _filter_attr_values, _filter_attr_labels = _result_filter_attr_choices(_proj())
        _ch_f = list(_filter_attr_labels)
        _filter_attr_default = _filter_attr_labels[0] if _filter_attr_labels else None
        with gr.Row():
            results_attr_filter = gr.Dropdown(
                label="Ползунки уверенности применять к",
                choices=_ch_f,
                value=_filter_attr_default,
                allow_custom_value=True,
                info=(
                    "Первая опция — фильтр по **средней** уверенности карточки. "
                    "Список строится из **конфига проекта** и дополняется атрибутами, которые **реально есть в сохранённых результатах** "
                    "(например `metal_color` в ювелирке, даже если в шаблоне направления числится одежда). "
                    "Для **надписей на фото** — только товары, где модель реально нашла текст (есть `text_found` или непустой список `texts`); дальше — диапазон уверенности по этому блоку."
                ),
            )
            results_attr_value_filter = gr.Dropdown(
                label="Значение атрибута (сужение списка)",
                choices=[_RESULT_ATTR_VALUE_ANY_LABEL],
                value=_RESULT_ATTR_VALUE_ANY_LABEL,
                allow_custom_value=True,
                interactive=False,
                info=(
                    "После выбора **конкретного атрибута** (не «средняя по карточке») список подставляется из текущей выборки; "
                    "можно выбрать значение или **ввести свой текст** — поиск **по подстроке** в сыром значении и в переводе (RU). "
                    "Для **надписей** — подстрока в объединённом тексте с фото."
                ),
            )
        with gr.Row():
            btn_results_prev = gr.Button("◀  Назад", min_width=110, scale=0)
            results_page = gr.Number(
                value=1,
                minimum=1,
                precision=0,
                label="Стр.",
                scale=0,
                min_width=80,
                container=True,
            )
            btn_results_next = gr.Button("Вперёд  ▶", min_width=110, scale=0)
        with gr.Row():
            csv_download = gr.DownloadButton(label="↓ Скачать CSV", visible=True, scale=0)
            csv_download_light = gr.DownloadButton(label="↓ Скачать узкий CSV (id, атрибут, значение)", visible=True, scale=0)
            export_status = gr.Markdown(value="")

        _g_rr = pm.get_global_settings()
        _proj_keys = []
        for _d in _proj().get("directions", []):
            for _a in _d.get("attributes", []):
                _k = (_a.get("key") or "").strip()
                if _k and _k not in _proj_keys:
                    _proj_keys.append(_k)
        _rr_attr_choices = ["__text__ (надписи на фото)"] + [f"{_k}" for _k in _proj_keys]
        rr_reprocess_queue = gr.State([])  # offer_id — без гигантского multiselect в UI
        with gr.Accordion("Повторная обработка (другая модель / только выбранные атрибуты)", open=False):
            gr.Markdown(
                "Использует **текущий фильтр** (ползунки, категория, модель, атрибут/значение, дедуп в списке). "
                "Кнопка **«В список: все по текущему фильтру»** — **над карточками** (рядом с галочками). "
                "Очередь — **компактная сводка** (первые id), без прокрутки тысяч строк. "
                "Добавление: «все по фильтру / странице / ниже порога», **«В очередь: отмеченные галочками»** под карточками, плюс галочки объединяются с очередью при **«Повторить выбранные»**. "
                "Если в «атрибуты для прогона» пусто — полный повторный анализ; иначе в БД **обновятся только** выбранные поля. "
                "Прогресс — в терминале (`[Reprocess]`)."
            )
            rr_reanalyze_attrs = gr.Dropdown(
                label="Атрибуты для частичного прогона (пусто = полностью)",
                choices=_rr_attr_choices,
                value=[],
                multiselect=True,
                allow_custom_value=False,
            )
            with gr.Row():
                rr_threshold = gr.Slider(
                    0,
                    100,
                    value=80,
                    step=1,
                    label="Порог: в список — офферы со средней уверенностью строго ниже (%)",
                )
                btn_rr_select_below = gr.Button("В список: все ниже порога (по средней)", variant="secondary")
            with gr.Row():
                btn_rr_select_page = gr.Button("В список: все на текущей странице", variant="secondary")
                btn_rr_clear_list = gr.Button("Очистить список офферов", variant="secondary")
            rr_queue_summary = gr.Markdown(value=_rr_queue_summary_markdown([]))
            with gr.Row():
                rr_model = gr.Dropdown(
                    label="Модель для повторного прогона",
                    choices=_global_model_choices(_g_rr.get("ollama_url")),
                    value=_g_rr.get("model", "qwen3.5:35b"),
                    allow_custom_value=True,
                )
                btn_rr_refresh_models = gr.Button("Обновить список моделей", size="sm", variant="secondary")
            btn_rr_start = gr.Button("▶ Повторить выбранные с этой моделью", variant="primary")
            rr_status = gr.Markdown(value="")

        results_stats = gr.Markdown(value="— загрузка при открытии вкладки —")
        with gr.Accordion("Подсказка: как править атрибуты (можно писать по-русски)", open=False):
            gr.Markdown(_correction_hint_md())
        results_state = gr.State([])  # полный отфильтрованный список (все страницы)
        selected_offer_ids = gr.State([])

        empty_pick = gr.update(visible=False, value=False)

        # Строка 1: выбор карточек + добавить все по фильтру в очередь повтора
        with gr.Row():
            btn_pick_all_filtered_cards = gr.Button("☑ Все по фильтру", variant="secondary", scale=0, size="sm")
            btn_pick_all_page_cards = gr.Button("☑ Страница", variant="secondary", scale=0, size="sm")
            btn_pick_none_page_cards = gr.Button("☐ Снять", variant="secondary", scale=0, size="sm")
            cb_page_select_all = gr.Checkbox(
                label="Вся страница",
                value=False,
                scale=0,
                elem_classes=["result-page-select-all"],
            )
            btn_rr_select_filtered = gr.Button("📥 В повтор: все по фильтру", variant="primary", scale=0, size="sm")
        # Строка 2: действия с отмеченными галочками
        with gr.Row():
            btn_rr_add_checked_cards = gr.Button("📋 В повтор: отмеченные", variant="secondary", scale=0, size="sm")
            btn_ft_queue_add = gr.Button("📌 В дообучение: отмеченные", variant="secondary", scale=0, size="sm")
            btn_ft_queue_clear = gr.Button("Очистить дообучение", variant="stop", scale=0, size="sm")
            btn_delete_all_picked = gr.Button("🗑 Удалить отмеченные", variant="stop", scale=0, size="sm")
        ft_queue_status = gr.Markdown(value="")

        def _add_picked_to_finetune_queue(selected_ids):
            name = _proj_name()
            if not name:
                return "❌ Выберите проект на вкладке «Проекты»."
            ids = [str(x).strip() for x in (selected_ids or []) if str(x).strip()]
            if not ids:
                return "❌ Отметьте галочками карточки на странице результатов."
            added, total = pm.append_finetune_queue_offer_ids(name, ids)
            return (
                f"✅ В очередь дообучения добавлено **{added}** новых offer_id (всего **{total}**). "
                f"Дальше: вкладка **«Дообучение»** → «Собрать датасет» с включённой галочкой очереди."
            )

        def _clear_finetune_queue():
            name = _proj_name()
            if not name:
                return "❌ Проект не выбран."
            pm.clear_finetune_queue_offer_ids(name)
            return "✅ Очередь дообучения очищена."

        btn_ft_queue_add.click(_add_picked_to_finetune_queue, inputs=[selected_offer_ids], outputs=[ft_queue_status])
        btn_ft_queue_clear.click(_clear_finetune_queue, outputs=[ft_queue_status])

        # Карточки: галочка, HTML, кнопки «править» и «удалить», блок редактирования под карточкой
        card_picks: list[gr.Checkbox] = []
        card_slots: list[gr.HTML] = []
        edit_buttons: list[gr.Button] = []
        delete_buttons: list[gr.Button] = []
        correction_accordions: list[gr.Accordion] = []
        corr_offer_ids: list[gr.Textbox] = []
        corr_attributes_jsons: list[gr.Textbox] = []
        corr_texts: list[gr.Textbox] = []
        btn_save_corrections: list[gr.Button] = []
        corr_statuses: list[gr.Markdown] = []

        for _ in range(MAX_RESULT_CARDS):
            with gr.Column():
                with gr.Row(elem_classes=["result-card-row"]):
                    card_picks.append(
                        gr.Checkbox(
                            value=False,
                            label="",
                            show_label=False,
                            visible=False,
                            scale=0,
                            min_width=0,
                            elem_classes=["result-card-pick"],
                        )
                    )
                    card_slots.append(gr.HTML(value="", visible=False))
                    with gr.Column(scale=0, min_width=96, elem_classes=["card-actions-col"]):
                        with gr.Row(elem_classes=["card-actions-inner"]):
                            edit_buttons.append(
                                gr.Button(
                                    "✏️",
                                    size="sm",
                                    visible=False,
                                    min_width=44,
                                    elem_classes=["card-edit-btn"],
                                )
                            )
                            delete_buttons.append(
                                gr.Button(
                                    "🗑️",
                                    size="sm",
                                    visible=False,
                                    min_width=44,
                                    elem_classes=["card-delete-btn"],
                                    variant="stop",
                                )
                            )
                acc = gr.Accordion("Правка атрибутов (для дообучения)", open=False, visible=False)
                correction_accordions.append(acc)
                with acc:
                    corr_offer_ids.append(gr.Textbox(label="Offer ID"))
                    corr_attributes_jsons.append(gr.Textbox(
                        label="Атрибуты (JSON). Пишите ключи и значения по-русски — см. подсказку выше.",
                        placeholder='{"Длина рукава": "длинный", "Воротник": "поло"}',
                        lines=3,
                    ))
                    corr_texts.append(gr.Textbox(label="Текст на одежде (пусто = не менять)"))
                    btn_save_corrections.append(gr.Button("Сохранить правку", variant="secondary"))
                    corr_statuses.append(gr.Markdown())

        results_rest_html = gr.HTML(value="", visible=False)

        def refresh_results(
            conf_min_val,
            conf_max_val,
            category,
            model_filter_val,
            hide_dup_val,
            filter_attr_val,
            attr_value_val,
            page_val,
            selected_ids,
            rr_queue_in,
        ):
            safe_dropdown = lambda choices, val=None: gr.update(choices=choices, value=(val if val is not None and val in choices else (choices[0] if choices else "Все")))
            empty_html = gr.update(value="", visible=False)
            empty_btn = gr.update(interactive=False, visible=False)
            sel_norm = _normalize_offer_id_list(selected_ids)
            rr_q = list(rr_queue_in or [])
            proj = _proj()
            fval_list_cfg, flab_list_cfg = _result_filter_attr_choices(proj)
            fkey = _resolve_result_filter_attr_key(filter_attr_val, fval_list_cfg, flab_list_cfg)
            if fkey not in fval_list_cfg and fval_list_cfg:
                fkey = fval_list_cfg[0]
            val_dd_reset = gr.update(
                choices=[_RESULT_ATTR_VALUE_ANY_LABEL],
                value=_RESULT_ATTR_VALUE_ANY_LABEL,
                interactive=False,
            )

            def fail_pack(
                msg,
                cats_list,
                cat_val,
                rest_html,
                fkey_canon,
                choices_for_dd=None,
                model_ch=None,
                model_val=None,
            ):
                ch = choices_for_dd if choices_for_dd is not None else flab_list_cfg
                fv_dd = _result_filter_label_for_key(fkey_canon, fval_list_cfg, flab_list_cfg)
                mch = model_ch if model_ch is not None else ["Все"]
                mva = (
                    model_val
                    if model_val is not None and model_val in mch
                    else (mch[0] if mch else "Все")
                )
                picks = [empty_pick] * MAX_RESULT_CARDS
                return (
                    msg,
                    *picks,
                    *[empty_html] * MAX_RESULT_CARDS,
                    *[empty_btn] * MAX_RESULT_CARDS,
                    *[empty_btn] * MAX_RESULT_CARDS,
                    *[gr.update(open=False, visible=False)] * MAX_RESULT_CARDS,
                    gr.update(value=rest_html, visible=bool(rest_html)),
                    [],
                    safe_dropdown(cats_list or ["Все"], cat_val),
                    safe_dropdown(mch, mva),
                    gr.update(value=1),
                    gr.update(choices=ch, value=fv_dd),
                    val_dd_reset,
                    gr.update(value=False),
                    sel_norm,
                    gr.update(value=_rr_queue_summary_markdown(rr_q)),
                    rr_q,
                )

            try:
                name = _proj_name()
                rdb = _results_db()
                if not name or not rdb:
                    return fail_pack(
                        "Проект не выбран",
                        ["Все"],
                        "Все",
                        "<p style='color:#666; padding:1rem;'>Выберите проект на вкладке <strong>«Проекты»</strong>, затем загрузите фид на вкладке <strong>«Фид»</strong> и запустите обработку на вкладке <strong>«Запуск»</strong>. После этого здесь появятся обработанные офферы.</p>",
                        fkey,
                    )

                min_c = int(conf_min_val) if conf_min_val is not None else 0
                max_c = int(conf_max_val) if conf_max_val is not None else 100
                cats = _results_categories(rdb)
                if category not in cats:
                    category = "Все"
                model_choices = _results_model_choices(rdb, category)
                m_pick = (model_filter_val or "").strip()
                if m_pick not in model_choices:
                    m_pick = "Все"

                (
                    results,
                    fkey,
                    loaded_count,
                    value_choices,
                    n_pre,
                    fval_list,
                    flab_list,
                ) = _results_tab_filtered_list(
                    rdb,
                    proj,
                    min_c,
                    max_c,
                    category,
                    bool(hide_dup_val),
                    filter_attr_val,
                    attr_value_val,
                    m_pick,
                )
                needle = _normalize_attr_value_filter_pick(attr_value_val)

                if not results:
                    picks = [empty_pick] * MAX_RESULT_CARDS
                    stat_empty = f"Записей нет (фильтр {min_c}–{max_c}% по выбранному режиму)"
                    hint_empty = "<p style='color:#666; padding:1rem;'>По выбранным фильтрам записей не найдено.</p>"
                    if fkey != "__avg__" and loaded_count > 0:
                        stat_empty += f" — в БД **{loaded_count}** строк по категории, но ни одна не подошла под атрибут `{html.escape(fkey)}`"
                        hint_empty = (
                            "<p style='color:#666; padding:1rem;'>В базе есть записи, но для выбранного атрибута нет совпадений "
                            f"(нет поля в результате или уверенность не в диапазоне {min_c}–{max_c}%). "
                            "Переключите <b>Ползунки уверенности применять к</b> на "
                            "<b>— По средней уверенности карточки —</b>.</p>"
                        )
                    if needle and fkey not in ("__avg__",) and n_pre > 0:
                        stat_empty += " — по **значению** после фильтра уверенности оставались карточки, но **ни одна не совпала** с введённым значением"
                        hint_empty = (
                            "<p style='color:#666; padding:1rem;'>Сбросьте значение на <b>— любое значение —</b> "
                            "или введите другую подстроку (поиск без учёта регистра).</p>"
                        )
                    sel_out = _prune_selection_to_visible_results(results, sel_norm)
                    return (
                        stat_empty,
                        *picks,
                        *[empty_html] * MAX_RESULT_CARDS,
                        *[empty_btn] * MAX_RESULT_CARDS,
                        *[empty_btn] * MAX_RESULT_CARDS,
                        *[gr.update(open=False, visible=False)] * MAX_RESULT_CARDS,
                        gr.update(value=hint_empty, visible=True),
                        [],
                        safe_dropdown(cats, category),
                        safe_dropdown(model_choices, m_pick),
                        gr.update(value=1),
                        gr.update(
                            choices=flab_list,
                            value=_result_filter_label_for_key(fkey, fval_list, flab_list),
                        ),
                        _results_attr_value_dropdown_update(attr_value_val, value_choices, fkey),
                        gr.update(value=False),
                        sel_out,
                        gr.update(value=_rr_queue_summary_markdown(rr_q)),
                        rr_q,
                    )

                good = sum(1 for r in results if r["avg_confidence"] >= 80)
                mid = sum(1 for r in results if 50 <= r["avg_confidence"] < 80)
                bad = sum(1 for r in results if r["avg_confidence"] < 50)
                n = len(results)
                pages = max(1, math.ceil(n / RESULTS_PAGE_SIZE))
                p = _safe_results_page_index(page_val)
                p = max(1, min(p, pages))
                start = (p - 1) * RESULTS_PAGE_SIZE
                page_slice = results[start : start + RESULTS_PAGE_SIZE]
                sel_out = _prune_selection_to_visible_results(results, sel_norm)
                sel_set = set(sel_out)

                stats = (
                    f"По фильтру: **{n}** | страница **{p}** / **{pages}** "
                    f"(по **{RESULTS_PAGE_SIZE}** карточек) | "
                    f"🟢 ≥80%: **{good}** | 🟡 50–79%: **{mid}** | 🔴 ниже 50%: **{bad}**"
                )
                if hide_dup_val and fkey == "__avg__" and loaded_count > n:
                    stats += f" _(скрыто **{loaded_count - n}** дублей по URL фото)_"

                out_picks: list = []
                out_cards = []
                out_btns = []
                out_del_btns = []
                for i in range(MAX_RESULT_CARDS):
                    if i < len(page_slice):
                        oid = str(page_slice[i].get("offer_id", "")).strip()
                        out_picks.append(gr.update(visible=True, value=(oid in sel_set)))
                        out_cards.append(
                            gr.update(
                                value=_result_card_html(page_slice[i], _config_for_result_card_badges()),
                                visible=True,
                            )
                        )
                        out_btns.append(gr.update(interactive=True, visible=True))
                        out_del_btns.append(gr.update(interactive=True, visible=True))
                    else:
                        out_picks.append(empty_pick)
                        out_cards.append(empty_html)
                        out_btns.append(empty_btn)
                        out_del_btns.append(empty_btn)

                out_acc = [gr.update(open=False, visible=(i < len(page_slice))) for i in range(MAX_RESULT_CARDS)]
                return (
                    stats,
                    *out_picks,
                    *out_cards,
                    *out_btns,
                    *out_del_btns,
                    *out_acc,
                    gr.update(value="", visible=False),
                    results,
                    safe_dropdown(cats, category),
                    safe_dropdown(model_choices, m_pick),
                    gr.update(value=p),
                    gr.update(
                        choices=flab_list,
                        value=_result_filter_label_for_key(fkey, fval_list, flab_list),
                    ),
                    _results_attr_value_dropdown_update(attr_value_val, value_choices, fkey),
                    gr.update(value=_selection_covers_all_on_page(sel_out, page_slice)),
                    sel_out,
                    gr.update(value=_rr_queue_summary_markdown(rr_q)),
                    rr_q,
                )
            except Exception as e:
                return fail_pack(
                    "Ошибка загрузки результатов",
                    ["Все"],
                    "Все",
                    f"<p style='color:#c00; padding:1rem;'>Не удалось загрузить результаты: {html.escape(str(e))}.</p>",
                    "__avg__",
                    choices_for_dd=flab_list_cfg,
                )

        def make_delete_result(index: int):
            def delete_and_refresh(
                state_list,
                page_val,
                conf_min_val,
                conf_max_val,
                category_val,
                model_filter_val,
                hide_dup_val,
                filter_attr_val,
                attr_value_val,
                selected_ids,
                rr_queue_in,
            ):
                new_sel = _normalize_offer_id_list(selected_ids)
                if state_list:
                    pi = _safe_results_page_index(page_val) - 1
                    gi = pi * RESULTS_PAGE_SIZE + index
                    if 0 <= gi < len(state_list):
                        oid = state_list[gi].get("offer_id")
                        if oid is not None:
                            oids = str(oid).strip()
                            if oids:
                                rdb = _results_db()
                                _delete_result_by_offer_id(rdb, oids)
                                new_sel = [x for x in new_sel if x != oids]
                return refresh_results(
                    conf_min_val,
                    conf_max_val,
                    category_val,
                    model_filter_val,
                    hide_dup_val,
                    filter_attr_val,
                    attr_value_val,
                    1,
                    new_sel,
                    rr_queue_in,
                )

            return delete_and_refresh

        def make_open_correction(index: int):
            def open_correction(state_list, page_val):
                if not state_list:
                    accordion_updates = [gr.update(open=False) for _ in range(MAX_RESULT_CARDS)]
                    return "", "", "", *accordion_updates
                pi = _safe_results_page_index(page_val) - 1
                gi = pi * RESULTS_PAGE_SIZE + index
                if gi >= len(state_list):
                    accordion_updates = [gr.update(open=False) for _ in range(MAX_RESULT_CARDS)]
                    return "", "", "", *accordion_updates
                proj = _proj()
                glossary = pm.load_attribute_glossary()
                oid, attrs_json, text = _result_to_correction_form(state_list[gi], proj, glossary)
                accordion_updates = [gr.update(open=(j == index)) for j in range(MAX_RESULT_CARDS)]
                return oid, attrs_json, text, *accordion_updates

            return open_correction

        def make_toggle_pick(index: int):
            def on_pick(checked, page_val, full_list, selected_list):
                s = _normalize_offer_id_list(selected_list)
                p = _safe_results_page_index(page_val) - 1
                start = p * RESULTS_PAGE_SIZE
                if not full_list or start + index >= len(full_list):
                    return s
                oid = str(full_list[start + index].get("offer_id", "")).strip()
                if not oid:
                    return s
                if checked and oid not in s:
                    s.append(oid)
                elif not checked and oid in s:
                    s = [x for x in s if x != oid]
                return s

            return on_pick

        def pick_all_checkboxes_on_page(state_list, page_val, selected_ids):
            """Добавить в выбор все offer_id с текущей страницы и обновить галочки."""
            if not state_list:
                return tuple(
                    [_normalize_offer_id_list(selected_ids)]
                    + [gr.update() for _ in range(MAX_RESULT_CARDS)]
                    + [gr.update(value=False)]
                )
            p = _safe_results_page_index(page_val) - 1
            start = p * RESULTS_PAGE_SIZE
            sl = state_list[start : start + RESULTS_PAGE_SIZE]
            s = _normalize_offer_id_list(selected_ids)
            for r in sl:
                oid = str(r.get("offer_id", "")).strip()
                if oid and oid not in s:
                    s.append(oid)
            out: list = []
            for i in range(MAX_RESULT_CARDS):
                if i < len(sl):
                    out.append(gr.update(visible=True, value=True))
                else:
                    out.append(empty_pick)
            sync = bool(sl) and _selection_covers_all_on_page(s, sl)
            return tuple([s] + out + [gr.update(value=sync)])

        def pick_all_checkboxes_filtered(state_list, page_val, selected_ids):
            """Все offer_id из полного отфильтрованного списка; галочки на текущей странице — включены."""
            if not state_list:
                return tuple(
                    [_normalize_offer_id_list(selected_ids)]
                    + [gr.update() for _ in range(MAX_RESULT_CARDS)]
                    + [gr.update(value=False)]
                )
            s = _normalize_offer_id_list(selected_ids)
            for r in state_list:
                oid = str(r.get("offer_id", "")).strip()
                if oid and oid not in s:
                    s.append(oid)
            p = _safe_results_page_index(page_val) - 1
            start = p * RESULTS_PAGE_SIZE
            sl = state_list[start : start + RESULTS_PAGE_SIZE]
            out: list = []
            for i in range(MAX_RESULT_CARDS):
                if i < len(sl):
                    out.append(gr.update(visible=True, value=True))
                else:
                    out.append(empty_pick)
            sync = bool(sl) and _selection_covers_all_on_page(s, sl)
            return tuple([s] + out + [gr.update(value=sync)])

        def pick_none_checkboxes_on_page(state_list, page_val, selected_ids):
            """Убрать из выбора только offer_id текущей страницы; галочки на странице снять."""
            if not state_list:
                return tuple(
                    [_normalize_offer_id_list(selected_ids)]
                    + [gr.update() for _ in range(MAX_RESULT_CARDS)]
                    + [gr.update(value=False)]
                )
            p = _safe_results_page_index(page_val) - 1
            start = p * RESULTS_PAGE_SIZE
            sl = state_list[start : start + RESULTS_PAGE_SIZE]
            page_oids = {str(r.get("offer_id", "")).strip() for r in sl if r.get("offer_id")}
            s = [x for x in _normalize_offer_id_list(selected_ids) if x not in page_oids]
            out: list = []
            for i in range(MAX_RESULT_CARDS):
                if i < len(sl):
                    out.append(gr.update(visible=True, value=False))
                else:
                    out.append(empty_pick)
            return tuple([s] + out + [gr.update(value=False)])

        def toggle_page_select_all(checked, state_list, page_val, selected_ids):
            if checked:
                return pick_all_checkboxes_on_page(state_list, page_val, selected_ids)
            return pick_none_checkboxes_on_page(state_list, page_val, selected_ids)

        def delete_all_checked_results(
            selected_ids,
            conf_min_val,
            conf_max_val,
            category_val,
            model_filter_val,
            hide_dup_val,
            filter_attr_val,
            attr_value_val,
            page_val,
            rr_queue_in,
        ):
            ids = _normalize_offer_id_list(selected_ids)
            if not ids:
                return refresh_results(
                    conf_min_val,
                    conf_max_val,
                    category_val,
                    model_filter_val,
                    hide_dup_val,
                    filter_attr_val,
                    attr_value_val,
                    page_val,
                    selected_ids,
                    rr_queue_in,
                )
            rdb = _results_db()
            if rdb and rdb.exists():
                _delete_results_batch(rdb, ids)
            return refresh_results(
                conf_min_val,
                conf_max_val,
                category_val,
                model_filter_val,
                hide_dup_val,
                filter_attr_val,
                attr_value_val,
                1,
                [],
                rr_queue_in,
            )

        def save_correction(oid, attrs_json, text_val):
            name = _proj_name()
            if not name:
                return "❌ Проект не выбран"
            if not oid:
                return "❌ Введите Offer ID"

            corrected_attrs = {}
            if attrs_json and attrs_json.strip():
                try:
                    raw = json.loads(attrs_json)
                    if isinstance(raw, dict):
                        proj = _proj()
                        glossary = pm.load_attribute_glossary()
                        corrected_attrs = pm.normalize_correction_attrs(raw, proj, glossary)
                except json.JSONDecodeError as e:
                    return f"❌ Неверный JSON: {e}"

            rdb = _results_db()
            proj = _proj()
            glossary = pm.load_attribute_glossary()
            pic_url = ""
            if rdb and rdb.exists():
                con = fc.sqlite_connect(rdb)
                row = con.execute("SELECT picture_url FROM results WHERE offer_id=?", (oid,)).fetchone()
                con.close()
                if row:
                    pic_url = row[0] or ""

            has_attrs = bool(corrected_attrs)
            has_text = (text_val or "").strip() != ""
            if not has_attrs and not has_text:
                return "❌ Нечего сохранить: пустой JSON атрибутов и пустой текст"

            before_edit = None
            if has_attrs and rdb and rdb.exists():
                from fine_tune.dataset_builder import direction_attributes_to_flat_answer

                ref_id = str(oid).strip()
                fb0 = _load_result_by_offer_id(rdb, ref_id)
                if fb0 and isinstance(fb0.get("direction_attributes"), dict):
                    td0 = fb0.get("text_detection")
                    before_edit = {
                        "attributes_flat": direction_attributes_to_flat_answer(fb0.get("direction_attributes")),
                        "text_detection": td0 if isinstance(td0, dict) else {},
                    }

            dup_ids = _offer_ids_sharing_normalized_picture(rdb, pic_url) if rdb and rdb.exists() else []
            if not dup_ids:
                dup_ids = [str(oid)]
            elif str(oid) not in dup_ids:
                dup_ids = [str(oid)] + dup_ids

            n_saved_db = 0
            if rdb and rdb.exists():
                for dup_oid in dup_ids:
                    base = _load_result_by_offer_id(rdb, dup_oid)
                    if not base:
                        continue
                    updated = _apply_correction_to_stored_result(
                        base,
                        corrected_attrs,
                        text_val,
                        proj,
                        glossary,
                    )
                    _save_result(rdb, updated)
                    n_saved_db += 1
                    corr_pic = (base.get("picture_url") or "").strip()
                    corr_entry = {
                        "offer_id": dup_oid,
                        "picture_url": corr_pic,
                        "corrected_attributes": corrected_attrs,
                        "corrected_text": (
                            {"texts": [text_val], "text_found": bool((text_val or "").strip())}
                            if (text_val or "").strip()
                            else {}
                        ),
                    }
                    if before_edit is not None:
                        corr_entry["before_edit"] = before_edit
                    pm.save_correction(name, corr_entry)

            if n_saved_db == 0:
                correction = {
                    "offer_id": oid,
                    "picture_url": pic_url,
                    "corrected_attributes": corrected_attrs,
                    "corrected_text": (
                        {"texts": [text_val], "text_found": bool((text_val or "").strip())}
                        if (text_val or "").strip()
                        else {}
                    ),
                }
                if before_edit is not None:
                    correction["before_edit"] = before_edit
                pm.save_correction(name, correction)
                return f"✅ Правка записана в corrections.json для {oid} (строка в results не найдена)"

            extra = ""
            if len(dup_ids) > 1:
                extra = f" — та же картинка (нормализованный URL): **{len(dup_ids)}** offer_id"
            return f"✅ Правка применена к **{n_saved_db}** карточкам в результатах{extra}"

        def export_csv(
            conf_min_val,
            conf_max_val,
            category_val,
            model_filter_val,
            hide_dup_val,
            filter_attr_val,
            attr_value_val,
        ):
            name = _proj_name()
            rdb = _results_db()
            if not name or not rdb or not rdb.exists():
                return None, "❌ Выберите проект и загрузите результаты."
            proj = _proj()
            glossary = pm.load_attribute_glossary()
            min_c = int(conf_min_val) if conf_min_val is not None else 0
            max_c = int(conf_max_val) if conf_max_val is not None else 100
            cat = (category_val or "Все").strip()
            if cat not in _results_categories(rdb):
                cat = "Все"
            mch = _results_model_choices(rdb, cat)
            m_pick = (model_filter_val or "").strip()
            if m_pick not in mch:
                m_pick = "Все"
            rows, _fk, _hint, _vc, _npre, _fv, _fl = _results_tab_filtered_list(
                rdb,
                proj,
                min_c,
                max_c,
                cat,
                bool(hide_dup_val),
                filter_attr_val,
                attr_value_val,
                m_pick,
            )
            path = _export_results_to_csv(rows, proj, glossary, out_dir=rdb.parent)
            if path:
                return (
                    str(path),
                    f"✅ Файл готов: **{path.name}** — **{len(rows)}** карточек по текущим фильтрам — нажмите «Скачать CSV».",
                )
            return None, "❌ Нет данных по выбранным фильтрам или ошибка экспорта."

        btn_export_csv.click(
            export_csv,
            inputs=[
                conf_min,
                conf_max,
                cat_filter,
                results_model_filter,
                hide_dup_photos,
                results_attr_filter,
                results_attr_value_filter,
            ],
            outputs=[csv_download, export_status],
        )

        def export_csv_light(
            conf_min_val,
            conf_max_val,
            category_val,
            model_filter_val,
            hide_dup_val,
            filter_attr_val,
            attr_value_val,
        ):
            name = _proj_name()
            rdb = _results_db()
            if not name or not rdb or not rdb.exists():
                return None, "❌ Выберите проект и загрузите результаты."
            proj = _proj()
            glossary = pm.load_attribute_glossary()
            min_c = int(conf_min_val) if conf_min_val is not None else 0
            max_c = int(conf_max_val) if conf_max_val is not None else 100
            cat = (category_val or "Все").strip()
            if cat not in _results_categories(rdb):
                cat = "Все"
            mch = _results_model_choices(rdb, cat)
            m_pick = (model_filter_val or "").strip()
            if m_pick not in mch:
                m_pick = "Все"
            rows, _fk, _hint, _vc, _npre, _fv, _fl = _results_tab_filtered_list(
                rdb,
                proj,
                min_c,
                max_c,
                cat,
                bool(hide_dup_val),
                filter_attr_val,
                attr_value_val,
                m_pick,
            )
            path = _export_results_to_csv_light(rows, proj, glossary, out_dir=rdb.parent)
            if path:
                return (
                    str(path),
                    f"✅ Узкий CSV: **{path.name}** — **{len(rows)}** карточек. Колонки: external_id, attribute_name, attribute_value.",
                )
            return None, "❌ Нет данных по выбранным фильтрам или ошибка экспорта."

        btn_export_csv_light.click(
            export_csv_light,
            inputs=[
                conf_min,
                conf_max,
                cat_filter,
                results_model_filter,
                hide_dup_photos,
                results_attr_filter,
                results_attr_value_filter,
            ],
            outputs=[csv_download_light, export_status],
        )

        def select_below_threshold(state_list, thr_val, rr_queue):
            if not state_list:
                return (
                    gr.update(),
                    list(rr_queue or []),
                    "❌ Сначала нажмите **«Обновить»** — список офферов пуст.",
                )
            t = int(thr_val) if thr_val is not None else 80
            new_ids: list[str] = []
            for r in state_list:
                ac = r.get("avg_confidence") if r.get("avg_confidence") is not None else 0
                try:
                    if int(ac) >= t:
                        continue
                except (TypeError, ValueError):
                    continue
                oid = r.get("offer_id")
                if oid is None:
                    continue
                s = str(oid).strip()
                if s:
                    new_ids.append(s)
            merged = _merge_offer_ids_into_queue(rr_queue, new_ids)
            return (
                gr.update(value=_rr_queue_summary_markdown(merged)),
                merged,
                f"✅ В очередь добавлено **{len(new_ids)}** офферов с уверенностью **ниже** {t}% (всего в очереди: **{len(merged)}**).",
            )

        def refresh_rr_model_choices():
            g = pm.get_global_settings()
            ch = _global_model_choices(g.get("ollama_url"))
            cur = (g.get("model") or "qwen3.5:35b").strip()
            val = cur if cur in ch else (ch[0] if ch else cur)
            return gr.update(choices=ch, value=val)

        def refresh_results_page1(
            conf_min_val,
            conf_max_val,
            category,
            model_filter_val,
            hide_dup_val,
            filter_attr_val,
            attr_value_val,
            selected_ids,
            rr_queue_in,
        ):
            return refresh_results(
                conf_min_val,
                conf_max_val,
                category,
                model_filter_val,
                hide_dup_val,
                filter_attr_val,
                attr_value_val,
                1,
                selected_ids,
                rr_queue_in,
            )

        def select_all_page_rr(state_list, page_val, rr_queue):
            if not state_list:
                return (
                    gr.update(),
                    list(rr_queue or []),
                    "❌ Сначала **«Обновить»**.",
                )
            p = _safe_results_page_index(page_val) - 1
            sl = state_list[p * RESULTS_PAGE_SIZE : p * RESULTS_PAGE_SIZE + RESULTS_PAGE_SIZE]
            new_ids: list[str] = []
            for r in sl:
                oid = r.get("offer_id")
                if oid is None:
                    continue
                s = str(oid).strip()
                if s:
                    new_ids.append(s)
            merged = _merge_offer_ids_into_queue(rr_queue, new_ids)
            return (
                gr.update(value=_rr_queue_summary_markdown(merged)),
                merged,
                f"✅ В очередь добавлено **{len(new_ids)}** с текущей страницы (всего в очереди: **{len(merged)}**).",
            )

        def select_all_filtered_rr(state_list, rr_queue):
            if not state_list:
                return (
                    gr.update(),
                    list(rr_queue or []),
                    "❌ Сначала **«Обновить»**.",
                )
            new_ids: list[str] = []
            for r in state_list:
                oid = r.get("offer_id")
                if oid is None:
                    continue
                s = str(oid).strip()
                if s:
                    new_ids.append(s)
            merged = _merge_offer_ids_into_queue(rr_queue, new_ids)
            return (
                gr.update(value=_rr_queue_summary_markdown(merged)),
                merged,
                f"✅ В очередь добавлено **{len(new_ids)}** по текущему фильтру (всего в очереди: **{len(merged)}**).",
            )

        def clear_rr_list():
            return (
                gr.update(value=_rr_queue_summary_markdown([])),
                [],
                "Очередь повторной обработки очищена.",
            )

        def add_checked_to_rr_queue(selected_ids, rr_queue):
            ids = _normalize_offer_id_list(selected_ids)
            if not ids:
                return (
                    gr.update(),
                    list(rr_queue or []),
                    "❌ Нет отмеченных галочками: нажмите «Выбрать все отфильтрованные» или отметьте карточки вручную.",
                )
            merged = _merge_offer_ids_into_queue(rr_queue, ids)
            return (
                gr.update(value=_rr_queue_summary_markdown(merged)),
                merged,
                f"✅ В очередь добавлено **{len(ids)}** отмеченных offer_id (всего в очереди: **{len(merged)}**).",
            )

        def _reanalyze_keys_from_ui(selected: list | None) -> list[str] | None:
            if not selected:
                return None
            out: list[str] = []
            for s in selected:
                t = str(s).strip()
                if t.startswith("__text__"):
                    out.append("__text__")
                else:
                    out.append(t)
            return out or None

        def start_reprocess(rr_queue, model_val, rr_keys_vals, selected_ids):
            q = _normalize_offer_id_list(rr_queue)
            sid = _normalize_offer_id_list(selected_ids)
            merged = list(dict.fromkeys(q + sid))
            if not merged:
                return (
                    "❌ Не выбран ни один оффер: добавьте их в **очередь повторной обработки** "
                    "(кнопки «В список…» / «В очередь: отмеченные галочками») и/или отметьте галочками на карточках."
                )
            m = (model_val or "").strip()
            if not m:
                return "❌ Укажите модель."
            rkeys = _reanalyze_keys_from_ui(rr_keys_vals)
            mode = "полный прогон" if not rkeys else f"частично: {rkeys!r}"
            threading.Thread(
                target=lambda: _reprocess_results_worker(merged, m, rkeys),
                daemon=True,
            ).start()
            return (
                f"▶ Запущено: **{len(merged)}** офферов, модель `{m}`, **{mode}**. "
                f"Прогресс в терминале (`[Reprocess] …`). Потом **«Обновить»**."
            )

        refresh_outputs = (
            [results_stats]
            + card_picks
            + card_slots
            + edit_buttons
            + delete_buttons
            + correction_accordions
            + [
                results_rest_html,
                results_state,
                cat_filter,
                results_model_filter,
                results_page,
                results_attr_filter,
                results_attr_value_filter,
                cb_page_select_all,
                selected_offer_ids,
                rr_queue_summary,
                rr_reprocess_queue,
            ]
        )
        refresh_inputs = [
            conf_min,
            conf_max,
            cat_filter,
            results_model_filter,
            hide_dup_photos,
            results_attr_filter,
            results_attr_value_filter,
            results_page,
            selected_offer_ids,
            rr_reprocess_queue,
        ]
        btn_refresh_results.click(refresh_results, inputs=refresh_inputs, outputs=refresh_outputs)
        results_tab.select(refresh_results, inputs=refresh_inputs, outputs=refresh_outputs)
        _refresh_page1_inputs = [
            conf_min,
            conf_max,
            cat_filter,
            results_model_filter,
            hide_dup_photos,
            results_attr_filter,
            results_attr_value_filter,
            selected_offer_ids,
            rr_reprocess_queue,
        ]
        hide_dup_photos.change(
            refresh_results_page1,
            inputs=_refresh_page1_inputs,
            outputs=refresh_outputs,
        )
        conf_min.change(
            refresh_results_page1,
            inputs=_refresh_page1_inputs,
            outputs=refresh_outputs,
        )
        conf_max.change(
            refresh_results_page1,
            inputs=_refresh_page1_inputs,
            outputs=refresh_outputs,
        )
        results_attr_filter.change(
            refresh_results_page1,
            inputs=_refresh_page1_inputs,
            outputs=refresh_outputs,
        )
        results_attr_value_filter.change(
            refresh_results_page1,
            inputs=_refresh_page1_inputs,
            outputs=refresh_outputs,
        )
        cat_filter.change(
            refresh_results_page1,
            inputs=_refresh_page1_inputs,
            outputs=refresh_outputs,
        )
        results_model_filter.change(
            refresh_results_page1,
            inputs=_refresh_page1_inputs,
            outputs=refresh_outputs,
        )
        results_page.change(refresh_results, inputs=refresh_inputs, outputs=refresh_outputs)

        def _results_page_bump(
            delta,
            conf_min_val,
            conf_max_val,
            category,
            model_filter_val,
            hide_dup_val,
            filter_attr_val,
            attr_value_val,
            page_val,
            selected_ids,
            rr_queue_in,
        ):
            cur = _safe_results_page_index(page_val)
            return refresh_results(
                conf_min_val,
                conf_max_val,
                category,
                model_filter_val,
                hide_dup_val,
                filter_attr_val,
                attr_value_val,
                cur + delta,
                selected_ids,
                rr_queue_in,
            )

        def _page_prev_wrapper(
            conf_min_val,
            conf_max_val,
            category,
            model_filter_val,
            hide_dup_val,
            filter_attr_val,
            attr_value_val,
            page_val,
            selected_ids,
            rr_queue_in,
        ):
            return _results_page_bump(
                -1,
                conf_min_val,
                conf_max_val,
                category,
                model_filter_val,
                hide_dup_val,
                filter_attr_val,
                attr_value_val,
                page_val,
                selected_ids,
                rr_queue_in,
            )

        def _page_next_wrapper(
            conf_min_val,
            conf_max_val,
            category,
            model_filter_val,
            hide_dup_val,
            filter_attr_val,
            attr_value_val,
            page_val,
            selected_ids,
            rr_queue_in,
        ):
            return _results_page_bump(
                1,
                conf_min_val,
                conf_max_val,
                category,
                model_filter_val,
                hide_dup_val,
                filter_attr_val,
                attr_value_val,
                page_val,
                selected_ids,
                rr_queue_in,
            )

        btn_results_prev.click(_page_prev_wrapper, inputs=refresh_inputs, outputs=refresh_outputs)
        btn_results_next.click(_page_next_wrapper, inputs=refresh_inputs, outputs=refresh_outputs)

        _pick_page_inputs = [results_state, results_page, selected_offer_ids]
        _pick_page_outputs = [selected_offer_ids] + card_picks + [cb_page_select_all]
        btn_pick_all_filtered_cards.click(
            pick_all_checkboxes_filtered, inputs=_pick_page_inputs, outputs=_pick_page_outputs
        )
        btn_pick_all_page_cards.click(pick_all_checkboxes_on_page, inputs=_pick_page_inputs, outputs=_pick_page_outputs)
        btn_pick_none_page_cards.click(pick_none_checkboxes_on_page, inputs=_pick_page_inputs, outputs=_pick_page_outputs)
        cb_page_select_all.change(
            toggle_page_select_all,
            inputs=[cb_page_select_all, results_state, results_page, selected_offer_ids],
            outputs=_pick_page_outputs,
        )
        btn_delete_all_picked.click(
            delete_all_checked_results,
            inputs=[
                selected_offer_ids,
                conf_min,
                conf_max,
                cat_filter,
                results_model_filter,
                hide_dup_photos,
                results_attr_filter,
                results_attr_value_filter,
                results_page,
                rr_reprocess_queue,
            ],
            outputs=refresh_outputs,
        )
        btn_rr_add_checked_cards.click(
            add_checked_to_rr_queue,
            inputs=[selected_offer_ids, rr_reprocess_queue],
            outputs=[rr_queue_summary, rr_reprocess_queue, rr_status],
        )

        for i in range(MAX_RESULT_CARDS):
            edit_buttons[i].click(
                make_open_correction(i),
                inputs=[results_state, results_page],
                outputs=[corr_offer_ids[i], corr_attributes_jsons[i], corr_texts[i]] + correction_accordions,
            )
            delete_buttons[i].click(
                make_delete_result(i),
                inputs=[
                    results_state,
                    results_page,
                    conf_min,
                    conf_max,
                    cat_filter,
                    results_model_filter,
                    hide_dup_photos,
                    results_attr_filter,
                    results_attr_value_filter,
                    selected_offer_ids,
                    rr_reprocess_queue,
                ],
                outputs=refresh_outputs,
            )
            btn_save_corrections[i].click(
                save_correction,
                inputs=[corr_offer_ids[i], corr_attributes_jsons[i], corr_texts[i]],
                outputs=[corr_statuses[i]],
            )
            card_picks[i].change(
                make_toggle_pick(i),
                inputs=[card_picks[i], results_page, results_state, selected_offer_ids],
                outputs=[selected_offer_ids],
            )

        btn_rr_select_below.click(
            select_below_threshold,
            inputs=[results_state, rr_threshold, rr_reprocess_queue],
            outputs=[rr_queue_summary, rr_reprocess_queue, rr_status],
        )
        btn_rr_select_page.click(
            select_all_page_rr,
            inputs=[results_state, results_page, rr_reprocess_queue],
            outputs=[rr_queue_summary, rr_reprocess_queue, rr_status],
        )
        btn_rr_select_filtered.click(
            select_all_filtered_rr,
            inputs=[results_state, rr_reprocess_queue],
            outputs=[rr_queue_summary, rr_reprocess_queue, rr_status],
        )
        btn_rr_clear_list.click(
            clear_rr_list,
            outputs=[rr_queue_summary, rr_reprocess_queue, rr_status],
        )
        btn_rr_refresh_models.click(refresh_rr_model_choices, outputs=[rr_model])
        btn_rr_start.click(
            start_reprocess,
            inputs=[rr_reprocess_queue, rr_model, rr_reanalyze_attrs, selected_offer_ids],
            outputs=[rr_status],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Fine-tune
# ═══════════════════════════════════════════════════════════════════════════════

def tab_finetune():
    with gr.Tab("Дообучение"):
        gr.Markdown("## Дообучение модели (LoRA)")
        finetune_step1_info = gr.Markdown(value=get_finetune_dashboard_markdown())
        with gr.Row():
            btn_refresh_finetune_info = gr.Button("Обновить сводку", size="sm", variant="secondary")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Шаг 1. Собрать датасет")
                external_jsonl_path = gr.Textbox(
                    label="Путь к JSONL",
                    value="data/deepfashion/train.jsonl",
                    info="Обычно не менять.",
                )
                build_max_examples = gr.Number(
                    label="Макс. примеров для обучения (0 = все)",
                    value=0,
                    precision=0,
                    info="Сколько примеров из JSONL взять в датасет. 0 — все.",
                )
                build_skip_first = gr.Number(
                    label="Продолжить с примера (0 = с начала)",
                    value=0,
                    precision=0,
                    info="После остановки сборки подставьте сюда число «Обработано примеров» — следующая сборка продолжится с этого места.",
                )
                include_images_cb = gr.Checkbox(label="Включить картинки в датасет", value=True)
                include_auto_good_cb = gr.Checkbox(
                    label="Автодобавление уверенных карточек из БД результатов",
                    value=True,
                )
                include_results_queue_cb = gr.Checkbox(
                    label="Дополнительно: офферы из ручной очереди (кнопка на «Результаты»)",
                    value=False,
                )
                auto_good_min_conf = gr.Slider(
                    50, 100, value=90, step=1, label="Порог средней уверенности для авто-примеров (%)"
                )
                auto_good_max_n = gr.Number(
                    label="Макс. авто-примеров за одну сборку (0 = не ограничивать)",
                    value=300,
                    precision=0,
                    minimum=0,
                )
                dedupe_train_urls_cb = gr.Checkbox(
                    label="Дедупликация train.jsonl по нормализованному URL (рекомендуется)",
                    value=True,
                )
                rebuild_eval_anchors_cb = gr.Checkbox(
                    label="Обновить eval_anchors.jsonl (якорь для «Оценка до/после»)",
                    value=True,
                )
                eval_anchors_max_n = gr.Number(
                    label="Макс. картинок в eval_anchors.jsonl",
                    value=36,
                    precision=0,
                    minimum=4,
                )
                export_low_conf_review_cb = gr.Checkbox(
                    label="Обновить review_low_confidence.json (пул для разметки)",
                    value=True,
                )
                low_conf_max_conf = gr.Slider(
                    0, 100, value=78, step=1, label="Порог «низкая уверенность» для пула (строго ниже, %)"
                )
                low_conf_max_count = gr.Number(
                    label="Макс. строк в пуле низкой уверенности",
                    value=250,
                    precision=0,
                    minimum=0,
                )
                with gr.Row():
                    btn_build_dataset = gr.Button("Собрать датасет", variant="primary")
                    btn_stop_build = gr.Button("Остановить сборку", variant="stop")
                dataset_status = gr.Markdown()
                dataset_progress = gr.Markdown(visible=False)

            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Шаг 2. Обучение")
                ft_base_model = gr.Dropdown(
                    label="Базовая модель (HuggingFace)",
                    choices=[
                        "── Qwen3.5 — нативный multimodal (картинки + текст) ──",
                        "Qwen/Qwen3.5-0.8B-Instruct",
                        "Qwen/Qwen3.5-2B-Instruct",
                        "Qwen/Qwen3.5-4B-Instruct",
                        "Qwen/Qwen3.5-9B-Instruct",
                        "Qwen/Qwen3.5-27B-Instruct",
                        "Qwen/Qwen3.5-35B-A3B",
                        "── Qwen3-VL (предыдущее поколение VL) ──",
                        "Qwen/Qwen3-VL-4B-Instruct",
                        "Qwen/Qwen3-VL-8B-Instruct",
                        "Qwen/Qwen3-VL-32B-Instruct",
                        "── Qwen2.5-VL ──",
                        "Qwen/Qwen2.5-VL-3B-Instruct",
                        "Qwen/Qwen2.5-VL-7B-Instruct",
                        "Qwen/Qwen2.5-VL-72B-Instruct",
                    ],
                    value="Qwen/Qwen3.5-9B-Instruct",
                    allow_custom_value=True,
                )
                with gr.Accordion("💡 Памятка: объём GPU / LoRA vs инференс", open=False):
                    gr.Markdown(
                        "**Инференс в Ollama** (GGUF Q4) и **LoRA-обучение** — разные нагрузки. "
                        "Ollama qwen3.5:35b умещается на 24 GB; LoRA тянет полные веса HF + градиенты → на 24 GB безопаснее **4B/9B**. "
                        "**27B/35B** — пробуйте при необходимости; при нехватке памяти шаг упадёт с ошибкой.\n\n"
                        "Адаптер привязан к выбранной базе. Папка: `lora_out_<модель>` (например `lora_out_35B-A3B`)."
                    )
                _grad_accum = 4
                def _steps_from_examples(max_ex, batch):
                    if max_ex is None or (isinstance(max_ex, (int, float)) and max_ex <= 0):
                        return 200
                    b = max(1, int(batch or 2))
                    from math import ceil
                    return max(1, ceil(int(max_ex) / (b * _grad_accum)))
                with gr.Row():
                    ft_max_steps = gr.Number(label="Max steps", value=200, precision=0)
                    ft_lora_rank = gr.Number(label="LoRA rank", value=16, precision=0)
                    ft_batch_size = gr.Number(label="Batch size", value=2, precision=0)
                ft_time_estimate = gr.Markdown("_Примерное время: ~15–30 мин (зависит от модели и GPU)._")
                ft_max_train_examples = gr.Number(
                    label="Макс. примеров для этого запуска (0 = все из собранного)",
                    value=0,
                    precision=0,
                    info="Объём выборки. Steps подстраиваются под него (steps = примеры / (batch×4)).",
                )
                def on_max_examples_or_batch(max_ex, batch):
                    return _steps_from_examples(max_ex, batch)

                def update_time_estimate(steps):
                    s = int(steps) if steps is not None else 0
                    if s <= 0:
                        return "_Примерное время: зависит от числа шагов._"
                    # VL: ~3–10 сек/шаг (9B ~3s, 35B ~8s)
                    min_min = max(1, int(s * 3 / 60))
                    max_min = max(min_min, int(s * 10 / 60))
                    return f"_Примерное время: **~{min_min}–{max_min} мин** (зависит от модели и GPU)._"

                def on_steps_or_examples(max_ex, batch):
                    steps = _steps_from_examples(max_ex, batch)
                    return steps, update_time_estimate(steps)

                ft_max_train_examples.change(
                    on_steps_or_examples,
                    inputs=[ft_max_train_examples, ft_batch_size],
                    outputs=[ft_max_steps, ft_time_estimate],
                )
                ft_batch_size.change(
                    on_steps_or_examples,
                    inputs=[ft_max_train_examples, ft_batch_size],
                    outputs=[ft_max_steps, ft_time_estimate],
                )
                ft_max_steps.change(update_time_estimate, inputs=ft_max_steps, outputs=ft_time_estimate)
                ft_save_to_same_folder = gr.Checkbox(
                    label="Сохранять в ту же папку, что и адаптер (дозапись в один адаптер)",
                    value=False,
                )
                ft_resume_from_adapter = gr.Textbox(
                    label="Дообучить с предыдущего адаптера (пусто = с нуля)",
                    placeholder="fine_tune/lora_out/lora_adapter",
                    value="",
                    info="При указании пути базовая модель берётся из конфига адаптера (вариант в выпадающем списке выше игнорируется).",
                )
                with gr.Row():
                    btn_fill_resume = gr.Button("Подставить адаптер для выбранной модели", size="sm")
                def fill_last_adapter(base_model):
                    name = _proj_name()
                    path = pm.get_last_lora_path_for_base(name, base_model or "") if name else ""
                    if not path:
                        return "", False
                    p = Path(path)
                    adapter_sub = p / "lora_adapter"
                    path_str = str(adapter_sub) if adapter_sub.exists() else path
                    return path_str, True
                btn_fill_resume.click(fill_last_adapter, inputs=[ft_base_model], outputs=[ft_resume_from_adapter, ft_save_to_same_folder])
                _train_params_default = ("Qwen/Qwen3.5-9B-Instruct", 200, 16, 2, 0, "", False)

                def sync_train_params(base_model, max_steps, lora_rank, batch_size, max_train_examples, resume_from_adapter, save_to_same_folder):
                    return (
                        (base_model or "").strip() or _train_params_default[0],
                        int(max_steps) if max_steps is not None else _train_params_default[1],
                        int(lora_rank) if lora_rank is not None else _train_params_default[2],
                        int(batch_size) if batch_size is not None else _train_params_default[3],
                        int(max_train_examples) if max_train_examples is not None else _train_params_default[4],
                        (resume_from_adapter or "").strip() if resume_from_adapter is not None else _train_params_default[5],
                        bool(save_to_same_folder) if save_to_same_folder is not None else _train_params_default[6],
                    )

                train_params_cache = gr.State(_train_params_default)
                _train_inputs = [ft_base_model, ft_max_steps, ft_lora_rank, ft_batch_size, ft_max_train_examples, ft_resume_from_adapter, ft_save_to_same_folder]
                for comp in _train_inputs:
                    comp.change(sync_train_params, inputs=_train_inputs, outputs=[train_params_cache])
                gr.Markdown("_При нажатии «Остановить» обучение прервётся после текущего шага; **адаптер сохранится** — можно дообучить с него дальше (указать путь выше и запустить снова)._")
                with gr.Row():
                    btn_train = gr.Button("▶ Запустить LoRA обучение", variant="primary")
                    btn_stop_train = gr.Button("⏹ Остановить обучение", variant="stop")
                    btn_reset_used = gr.Button("🔄 Сбросить счётчик использованных")
                train_progress = gr.Slider(0, 100, value=0, step=0.5, label="Прогресс (%)", interactive=False)
                train_log = gr.Textbox(label="Лог обучения (подробный)", lines=16, interactive=False, max_lines=30)
                train_status = gr.Markdown()

                gr.Markdown("### Mini-проверка пайплайна")
                mini_test_status = gr.Markdown()
                mini_test_log = gr.Textbox(label="Лог mini-теста", lines=10, interactive=False, max_lines=60)
                btn_mini_test = gr.Button("▶ Тест пайплайна (mini train)", variant="secondary")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Шаг 3. Экспорт в Ollama")
                ft_adapter_path = gr.Textbox(
                    label="Путь к адаптеру",
                    placeholder="Пусто = последний обученный адаптер",
                )
                with gr.Row():
                    ft_ollama_name = gr.Textbox(
                        label="Имя модели в Ollama",
                        value="clothes-detector-v1",
                        scale=1,
                    )
                    ft_ollama_overwrite = gr.Dropdown(
                        label="Записать поверх модели",
                        choices=[],
                        value=None,
                        scale=1,
                    )
                with gr.Row():
                    btn_refresh_ollama_export = gr.Button("Обновить список моделей Ollama", size="sm")
                    btn_export = gr.Button("Экспортировать в Ollama", variant="secondary")
                _export_params_default = ("", "clothes-detector-v1")

                def sync_export_params(adapter_path, ollama_name):
                    return ((adapter_path or "").strip() or _export_params_default[0], (ollama_name or "").strip() or _export_params_default[1])

                export_params_cache = gr.State(_export_params_default)
                ft_adapter_path.change(sync_export_params, inputs=[ft_adapter_path, ft_ollama_name], outputs=[export_params_cache])
                ft_ollama_name.change(sync_export_params, inputs=[ft_adapter_path, ft_ollama_name], outputs=[export_params_cache])
                def fill_ollama_overwrite_choices():
                    g = pm.get_global_settings()
                    url = g.get("ollama_url", "http://127.0.0.1:11435")
                    models = ollama_list_models(url, timeout=3)
                    return gr.update(choices=models if models else [])
                btn_refresh_ollama_export.click(fill_ollama_overwrite_choices, outputs=ft_ollama_overwrite)
                ft_ollama_overwrite.change(lambda x: x or "", inputs=ft_ollama_overwrite, outputs=ft_ollama_name)
                export_status = gr.Markdown()
                export_log = gr.Textbox(label="Лог экспорта", lines=12, interactive=False, visible=True)

        _project_root = Path(__file__).resolve().parent
        _default_eval_jsonl = _project_root / "eval_sample.jsonl"
        if not _default_eval_jsonl.exists():
            _default_eval_jsonl.write_text('{"image_path": "", "attributes": {}}\n', encoding="utf-8")
        with gr.Accordion("Оценка до/после", open=False):
            def _eval_model_choices():
                g = pm.get_global_settings()
                url = g.get("ollama_url", "http://127.0.0.1:11435")
                # Сначала модели для извлечения (qwen3.5, свои из Ollama), потом VL для дообучения
                extraction_first = ["qwen3.5:4b", "qwen3.5:9b", "qwen3.5:35b"]
                vl_optional = ["qwen2.5-vl:3b", "qwen2.5-vl:7b", "qwen2.5-vl:72b"]
                from_ollama = ollama_list_models(url, timeout=3)
                seen = set()
                order = []
                for m in extraction_first + from_ollama + vl_optional:
                    if m not in seen:
                        seen.add(m)
                        order.append(m)
                return order if order else extraction_first + vl_optional
            _eval_choices = _eval_model_choices()
            _def_saved = pm.get_global_settings().get("model", "qwen3.5:35b")
            _def_before = "qwen3.5:35b"
            _def_after = _def_saved
            with gr.Row():
                eval_images_path = gr.Textbox(label="JSONL", value=str(_default_eval_jsonl))
                eval_out_dir = gr.Textbox(label="Папка результатов", value=str(_project_root / "results"))
                eval_max_examples = gr.Number(label="Макс. картинок (0 = все)", value=50, precision=0, minimum=0)
            with gr.Row():
                eval_model_before = gr.Dropdown(label="Модель «до» (для извлечения — базовая)", choices=_eval_choices, value=_def_before, allow_custom_value=True)
                eval_model_after = gr.Dropdown(label="Модель «после» (для извлечения — ваша дообученная)", choices=_eval_choices, value=_def_after, allow_custom_value=True)
                btn_refresh_eval_models = gr.Button("Обновить список моделей", size="sm")
            with gr.Row():
                btn_eval_compare = gr.Button("Сравнить до/после", variant="primary")
                btn_eval_stop = gr.Button("Остановить оценку (пауза)", variant="stop")
            eval_status = gr.Markdown()
            eval_report = gr.Textbox(label="Лог / Отчёт (здесь обновляется прогресс и итог)", lines=12, interactive=False)

        _eval_log_holder = []
        _eval_done_holder = [None]  # [None] or [(status_str, report_str)]
        _eval_running = [False]
        _eval_stop_event = threading.Event()

        def _resolve_eval_path(raw: str) -> Path:
            p = Path((raw or "").strip())
            if not p.is_absolute():
                p = _project_root / p
            return p.resolve()

        def _run_eval_thread(images_path, out_dir, phase, model_override, max_examples):
            try:
                from scripts.eval_before_after import run_eval_ui
                from scripts.eval_before_after import _collect_image_paths
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from scripts.eval_before_after import run_eval_ui, _collect_image_paths
            resolved_jsonl = _resolve_eval_path(images_path or "")
            resolved_out = _resolve_eval_path(out_dir or "results") if (out_dir or "").strip() else _project_root / "results"
            if phase in ("before", "after"):
                if not resolved_jsonl.exists():
                    _eval_done_holder[0] = (f"❌ Файл не найден: {resolved_jsonl}", "")
                    _eval_running[0] = False
                    return
                images_list = _collect_image_paths(str(resolved_jsonl))
                if not images_list:
                    _eval_done_holder[0] = ("❌ В файле нет строк с image_path.", "")
                    _eval_running[0] = False
                    return
            try:
                config_override = {"model": model_override} if model_override else None
                r = run_eval_ui(
                    str(resolved_jsonl) if phase in ("before", "after") else "",
                    str(resolved_out),
                    phase,
                    config_override=config_override,
                    progress_callback=lambda m: _eval_log_holder.append(m),
                    stop_event=_eval_stop_event,
                    max_examples=int(max_examples or 0),
                )
                status = f"✅ {r['message']}" if r["success"] else f"❌ {r['message']}"
                report = r.get("report_text") or ""
                _eval_done_holder[0] = (status, report)
            except Exception as e:
                _eval_done_holder[0] = (f"❌ Ошибка: {e}", "")
            finally:
                _eval_running[0] = False

        def _run_full_compare_thread(images_path, out_dir, max_examples, model_before, model_after):
            """Запуск по порядку: до (model_before) → после (model_after) → сравнить. Один поток — одна кнопка."""
            try:
                from scripts.eval_before_after import run_eval_ui
                from scripts.eval_before_after import _collect_image_paths
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from scripts.eval_before_after import run_eval_ui, _collect_image_paths
            try:
                resolved_jsonl = _resolve_eval_path(images_path or "")
                resolved_out = _resolve_eval_path(out_dir or "results") if (out_dir or "").strip() else _project_root / "results"
                if not resolved_jsonl.exists():
                    _eval_done_holder[0] = (f"❌ Файл не найден: {resolved_jsonl}", "")
                    _eval_running[0] = False
                    return
                images_list = _collect_image_paths(str(resolved_jsonl))
                if not images_list:
                    _eval_done_holder[0] = ("❌ В файле нет строк с image_path.", "")
                    _eval_running[0] = False
                    return
                max_n = int(max_examples or 0)
                cb = lambda m: _eval_log_holder.append(m)
                # 1) До
                r1 = run_eval_ui(str(resolved_jsonl), str(resolved_out), "before", {"model": model_before}, cb, _eval_stop_event, max_n)
                if not r1["success"]:
                    _eval_done_holder[0] = (f"❌ До: {r1['message']}", "")
                    _eval_running[0] = False
                    return
                if _eval_stop_event.is_set():
                    _eval_done_holder[0] = ("🛑 Остановлено.", "")
                    _eval_running[0] = False
                    return
                # 2) После
                r2 = run_eval_ui(str(resolved_jsonl), str(resolved_out), "after", {"model": model_after}, cb, _eval_stop_event, max_n)
                if not r2["success"]:
                    _eval_done_holder[0] = (f"❌ После: {r2['message']}", "")
                    _eval_running[0] = False
                    return
                if _eval_stop_event.is_set():
                    _eval_done_holder[0] = ("🛑 Остановлено.", "")
                    _eval_running[0] = False
                    return
                # 3) Сравнить
                r3 = run_eval_ui("", str(resolved_out), "compare", None, cb, None, 0)
                status = f"✅ {r3['message']}" if r3["success"] else f"❌ {r3['message']}"
                report = r3.get("report_text") or ""
                _eval_done_holder[0] = (status, report)
            except Exception as e:
                _eval_done_holder[0] = (f"❌ Ошибка: {e}", "")
            finally:
                _eval_running[0] = False

        def poll_eval_status(current_report):
            if _eval_done_holder[0] is not None:
                status, report = _eval_done_holder[0]
                _eval_done_holder[0] = None
                _eval_log_holder.clear()
                return status, report
            if _eval_log_holder:
                log_text = "\n".join(_eval_log_holder)
                return "⏳ Идёт оценка — лог ниже в блоке «Лог / Отчёт».", log_text
            return gr.update(), gr.update()

        def refresh_finetune_info_with_adapter():
            info = get_finetune_step1_info()
            path = pm.get_last_lora_path(_proj_name()) or ""
            return info, path

        btn_refresh_finetune_info.click(
            refresh_finetune_info_with_adapter,
            outputs=[finetune_step1_info, ft_adapter_path],
        )

        progress_holder = []
        stop_build_event = threading.Event()

        def build_dataset_gen(
            include_imgs,
            external_path,
            max_examples,
            skip_first,
            use_auto_good,
            use_queue,
            auto_min_conf,
            auto_max_n,
            dedupe_urls,
            rebuild_eval,
            eval_anchors_max,
            export_low_review,
            low_conf_thr,
            low_conf_max,
        ):
            name = _proj_name()
            if not name:
                yield "❌ Выберите проект вкладкой «Проекты»", gr.update(visible=True)
                return
            stop_build_event.clear()
            from fine_tune.dataset_builder import (
                build_dataset as _build,
                build_from_external,
                build_eval_anchors_jsonl,
                build_train_results_jsonl,
                dedupe_train_jsonl_file,
                export_low_confidence_review,
                merge_datasets,
            )
            out_dir = pm.project_dir(name) / "fine_tune_dataset"
            out_dir.mkdir(parents=True, exist_ok=True)
            train_path = out_dir / "train.jsonl"
            ext_jsonl = out_dir / "train_external.jsonl"
            train_results_path = out_dir / "train_results.jsonl"
            rdb = pm.results_db_path(name)
            progress_holder.clear()
            progress_holder.append(0)
            result_holder = {}
            max_ex = int(max_examples) if max_examples is not None else 0
            skip_n = int(skip_first) if skip_first is not None and skip_first else 0

            def do_build():
                parts = []
                corr_path = pm.corrections_path(name)
                if corr_path.exists():
                    r = _build(
                        corrections_path=corr_path,
                        output_dir=out_dir,
                        include_images=include_imgs,
                        min_corrections=0,
                    )
                    if r.get("error"):
                        result_holder["error"] = r["error"]
                        return
                    if r.get("valid_examples", 0):
                        parts.append(r["valid_examples"])
                else:
                    train_path.parent.mkdir(parents=True, exist_ok=True)
                    train_path.write_text("", encoding="utf-8")
                ext_path = (external_path or "").strip()
                if ext_path and Path(ext_path).exists():
                    ext_result = build_from_external(
                        ext_path, out_dir, include_images=include_imgs,
                        max_examples=max_ex if max_ex > 0 else 0,
                        progress_callback=lambda n: progress_holder.__setitem__(0, n),
                        stop_check=lambda: stop_build_event.is_set(),
                        skip_first_n=skip_n,
                    )
                    if ext_result.get("error"):
                        result_holder["error"] = ext_result["error"]
                        return
                    result_holder["stopped"] = ext_result.get("stopped", False)
                    if train_path.exists() and ext_jsonl.exists():
                        merge_datasets(train_path, ext_jsonl, out_dir)
                    elif ext_jsonl.exists():
                        import shutil
                        shutil.copy(ext_jsonl, train_path)
                use_q = bool(use_queue)
                use_a = bool(use_auto_good)
                extras: list = []
                if use_q or use_a:
                    if rdb and rdb.exists():
                        q_ids = pm.load_finetune_queue_offer_ids(name) if use_q else []
                        auto_min = int(auto_min_conf) if use_a else None
                        auto_max = int(auto_max_n) if auto_max_n is not None else 300
                        rr = build_train_results_jsonl(
                            rdb,
                            train_results_path,
                            include_images=include_imgs,
                            queued_offer_ids=q_ids,
                            skip_offer_ids=pm.correction_offer_ids(name),
                            auto_min_confidence=auto_min,
                            auto_max_examples=auto_max,
                        )
                        if rr.get("error"):
                            result_holder["error"] = rr["error"]
                            return
                        n_res = rr.get("valid_examples", 0)
                        if n_res:
                            parts.append(n_res)
                            extras.append(train_results_path)
                if extras:
                    merge_datasets(train_path, None, out_dir, extra_jsonl_paths=extras)
                result_holder["post_notes"] = []
                if bool(dedupe_urls) and train_path.exists():
                    dd = dedupe_train_jsonl_file(train_path)
                    if dd.get("dropped", 0) > 0:
                        result_holder["post_notes"].append(
                            f"Дедуп по URL: убрано **{dd['dropped']}** дублей, осталось **{dd['kept']}** примеров."
                        )
                if bool(rebuild_eval) and rdb and rdb.exists():
                    gset = pm.get_global_settings()
                    mx = int(gset.get("image_max_size") or 1024)
                    ev_path = out_dir / "eval_anchors.jsonl"
                    be = build_eval_anchors_jsonl(
                        rdb,
                        pm.load_corrections(name),
                        pm.image_cache_dir(name),
                        ev_path,
                        max_total=int(eval_anchors_max) if eval_anchors_max is not None else 36,
                        image_max_size=mx,
                    )
                    if be.get("valid_examples", 0):
                        result_holder["post_notes"].append(
                            f"Якорный eval: **{be['valid_examples']}** картинок → `{ev_path.name}` (подставьте путь в «Оценка до/после»)."
                        )
                if bool(export_low_review) and rdb and rdb.exists():
                    lowp = out_dir / "review_low_confidence.json"
                    er = export_low_confidence_review(
                        rdb,
                        lowp,
                        max_confidence=int(low_conf_thr) if low_conf_thr is not None else 78,
                        limit=int(low_conf_max) if low_conf_max is not None else 250,
                        skip_offer_ids=pm.correction_offer_ids(name),
                    )
                    if er.get("count", 0):
                        result_holder["post_notes"].append(
                            f"Пул низкой уверенности: **{er['count']}** офферов → `{lowp.name}`."
                        )
                result_holder["parts"] = parts
                result_holder["done"] = True

            t = threading.Thread(target=do_build, daemon=True)
            t.start()
            while t.is_alive():
                n = progress_holder[0] if progress_holder else 0
                yield (
                    "⏳ Идёт сборка датасета… Подождите. Нажмите «Остановить сборку», чтобы сохранить то, что уже есть.",
                    gr.update(value=f"Обработано примеров: **{n}**", visible=True),
                    gr.update(),
                )
                time.sleep(2)
            if result_holder.get("error"):
                yield (f"❌ {result_holder['error']}", gr.update(visible=True), get_finetune_step1_info())
                return
            if not train_path.exists() or train_path.stat().st_size == 0:
                yield (
                    "❌ Нет данных: нужны **результаты в БД** (прогон) + включённые **авто-уверенные**, и/или **правки**, и/или внешний **JSONL**. "
                    "Ручная очередь не обязательна.",
                    gr.update(visible=True),
                    get_finetune_step1_info(),
                )
                return
            with open(train_path, encoding="utf-8") as f:
                total = sum(1 for _ in f if _.strip())
            info_after = get_finetune_step1_info()
            extra_notes = result_holder.get("post_notes") or []
            note_block = "\n\n" + "\n".join(extra_notes) if extra_notes else ""
            if result_holder.get("stopped"):
                yield (
                    f"🛑 Сборка остановлена. Сохранено примеров: **{total}**. Чтобы продолжить позже — подставьте {total} в поле «Продолжить с примера» и нажмите «Собрать датасет» снова."
                    + note_block,
                    gr.update(visible=False),
                    info_after,
                )
            else:
                yield (
                    f"✅ Датасет собран. Примеров: **{total}**. Файл: `{train_path}`. Переходите к Шагу 2 — запуск обучения."
                    + note_block,
                    gr.update(visible=False),
                    info_after,
                )

        def build_dataset(
            include_imgs,
            external_path,
            max_examples,
            skip_first,
            use_auto_good,
            use_queue,
            auto_min_conf,
            auto_max_n,
            dedupe_urls,
            rebuild_eval,
            eval_anchors_max,
            export_low_review,
            low_conf_thr,
            low_conf_max,
        ):
            for status, prog_vis, info_md in build_dataset_gen(
                include_imgs,
                external_path,
                max_examples,
                skip_first,
                use_auto_good,
                use_queue,
                auto_min_conf,
                auto_max_n,
                dedupe_urls,
                rebuild_eval,
                eval_anchors_max,
                export_low_review,
                low_conf_thr,
                low_conf_max,
            ):
                yield status, prog_vis, info_md

        def reset_used_counter():
            name = _proj_name()
            if not name:
                return "❌ Проект не выбран"
            used_offset_path = pm.project_dir(name) / "fine_tune_dataset" / "used_offset.txt"
            try:
                if used_offset_path.exists():
                    used_offset_path.unlink()
                return "✅ Счётчик использованных примеров сброшен. Следующий запуск возьмёт примеры с начала датасета."
            except OSError as e:
                return f"❌ Не удалось сбросить: {e}"

        def run_training_start(cached):
            global _train_state, _train_stop_event
            if not cached or (isinstance(cached, (list, tuple)) and len(cached) >= 1 and cached[0] is None):
                return "❌ Параметры не получены. Откройте вкладку «Дообучение», заполните форму и нажмите кнопку снова.", "", 0.0
            base_model, max_steps, lora_rank, batch_size, max_train_examples, resume_from_adapter, save_to_same_folder = (
                cached[0], cached[1], cached[2], cached[3], cached[4], cached[5], bool(cached[6]) if len(cached) > 6 else False
            )
            name = _proj_name()
            if not name:
                return "❌ Проект не выбран", "", 0.0
            dataset_path = pm.project_dir(name) / "fine_tune_dataset" / "train.jsonl"
            used_offset_path = pm.project_dir(name) / "fine_tune_dataset" / "used_offset.txt"
            if not dataset_path.exists():
                return "❌ Сначала соберите датасет", "", 0.0
            try:
                used_offset = int(used_offset_path.read_text(encoding="utf-8").strip()) if used_offset_path.exists() else 0
            except (ValueError, OSError):
                used_offset = 0

            from fine_tune.train import check_gpu, check_unsloth
            gpu_ok, gpu_msg = check_gpu()
            if not gpu_ok:
                return f"❌ {gpu_msg}", "", 0.0
            unsloth_ok, unsloth_msg = check_unsloth()
            if not unsloth_ok:
                return f"❌ {unsloth_msg}", "", 0.0

            log_lines = [f"GPU: {gpu_msg}", "Запуск обучения..."]
            base_dir = pm.project_dir(name)
            resume_path_raw = (resume_from_adapter or "").strip() or None
            rp = None
            if resume_path_raw:
                rp = Path(resume_path_raw)
                if not rp.is_absolute():
                    rp = base_dir / resume_path_raw
                if not rp.exists():
                    rp = _project_root / resume_path_raw
            # Имя папки по базовой модели: один адаптер = одна база (LoRA не универсален)
            _base_for_slug = base_model
            if rp and rp.exists():
                _adapter_dir = rp if (rp / "adapter_config.json").exists() else (rp / "lora_adapter" if (rp / "lora_adapter").exists() else rp)
                _cfg = _adapter_dir / "adapter_config.json"
                if _cfg.exists():
                    try:
                        _cfg_data = json.loads(_cfg.read_text(encoding="utf-8"))
                        _base_for_slug = _cfg_data.get("base_model_name_or_path") or _cfg_data.get("base_model") or _base_for_slug
                    except Exception:
                        pass
            _model_slug = (_base_for_slug or "").split("/")[-1].strip() if _base_for_slug else "default"
            _model_slug = re.sub(r"[^\w\-.]", "_", _model_slug)[:48] or "default"
            output_dir = base_dir / f"lora_out_{_model_slug}"
            if rp and rp.exists() and save_to_same_folder:
                output_dir = rp.parent if rp.name == "lora_adapter" else rp
                log_lines.append(f"Дозапись в ту же папку: {output_dir}")
            if not save_to_same_folder and (output_dir / "lora_adapter").exists():
                for i in range(2, 100):
                    candidate = base_dir / f"lora_out_{_model_slug}_{i}"
                    if not (candidate / "lora_adapter").exists():
                        output_dir = candidate
                        log_lines.append(f"Новая папка: {output_dir}")
                        break
            train_log_path = output_dir / "train.log"
            try:
                train_log_path.parent.mkdir(parents=True, exist_ok=True)
                train_log_path.write_text("", encoding="utf-8")
            except Exception:
                pass
            log_lines.append(f"Лог дообучения: {train_log_path}")
            resume_path = str(rp) if (rp and rp.exists()) else None

            result_holder = {}
            progress_holder = {"step": 0, "total": max(1, int(max_steps))}
            _train_stop_event.clear()

            def do_train():
                from fine_tune.train import train
                def cb(step, total, loss):
                    progress_holder["step"] = step
                    progress_holder["total"] = total
                    pct = int(round(100 * step / total)) if total else 0
                    log_lines.append(f"Step {step}/{total} ({pct}%)  loss={loss:.4f}")
                result_holder["r"] = train(
                    dataset_path=dataset_path,
                    base_model=base_model,
                    output_dir=output_dir,
                    max_steps=int(max_steps),
                    batch_size=int(batch_size),
                    lora_rank=int(lora_rank),
                    progress_callback=cb,
                    max_train_examples=int(max_train_examples or 0),
                    resume_from_adapter=resume_path,
                    log_file=train_log_path,
                    stop_event=_train_stop_event,
                    skip_first_n=used_offset,
                )

            t = threading.Thread(target=do_train, daemon=True)
            _train_state["progress_holder"] = progress_holder
            _train_state["log_lines"] = log_lines
            _train_state["result_holder"] = result_holder
            _train_state["thread"] = t
            _train_state["train_log_path"] = str(train_log_path)
            _train_state["project_name"] = name
            _train_state["used_offset"] = used_offset
            _train_state["used_offset_path"] = str(used_offset_path)
            t.start()

            return "⏳ Обучается... Шаг 0/" + str(progress_holder["total"]) + " (0%)", "\n".join(log_lines[-40:]), 0.0

        def poll_training_status():
            global _train_state
            ph = _train_state.get("progress_holder")
            log_lines = _train_state.get("log_lines", [])
            result_holder = _train_state.get("result_holder", {})
            t = _train_state.get("thread")
            train_log_path = _train_state.get("train_log_path", "")
            if not t:
                # Нет запущенного обучения — сохраняем последний вывод, не затираем
                return (
                    _train_state.get("last_status", ""),
                    _train_state.get("last_log", ""),
                    _train_state.get("last_pct", 0.0),
                )
            if t.is_alive():
                step = ph.get("step", 0) if ph else 0
                total = ph.get("total", 1) if ph else 1
                pct = min(100, round(100 * step / total)) if total else 0
                status_msg = f"⏳ Обучается... Шаг {step}/{total} ({pct}%)"
                log_text = "\n".join(log_lines[-40:])
                _train_state["last_status"] = status_msg
                _train_state["last_log"] = log_text
                _train_state["last_pct"] = float(pct)
                return status_msg, log_text, float(pct)
            r = result_holder.get("r", {})
            if r.get("success"):
                out_dir = r.get("output_dir", "")
                if out_dir:
                    try:
                        pm.save_last_lora_path(_train_state.get("project_name", ""), out_dir)
                    except Exception:
                        pass
                used_offset_path = _train_state.get("used_offset_path")
                if used_offset_path:
                    try:
                        prev = int(_train_state.get("used_offset", 0))
                        added = int(r.get("examples_used", 0))
                        Path(used_offset_path).write_text(str(prev + added), encoding="utf-8")
                    except (ValueError, OSError, TypeError):
                        pass
                status_msg = f"✅ Обучение завершено! Адаптер: `{out_dir}`. Полный лог: `{train_log_path}`"
            else:
                step = ph.get("step", 0) if ph else 0
                total = ph.get("total", 1) if ph else 1
                pct = (step / total * 100) if total else 0
                status_msg = f"❌ {r.get('error', 'Неизвестная ошибка')}. Лог: `{train_log_path}`"
            log_text = "\n".join(log_lines[-50:])
            _train_state["last_status"] = status_msg
            _train_state["last_log"] = log_text
            _train_state["last_pct"] = 100.0 if r.get("success") else float(ph.get("step", 0) / max(1, ph.get("total", 1)) * 100) if ph else 0.0
            _train_state["thread"] = None  # чтобы следующий poll не считал обучение активным
            return status_msg, log_text, _train_state["last_pct"]

        def request_stop_train():
            _train_stop_event.set()
            return "🛑 Запрос на остановку отправлен. Обучение прервётся после текущего шага."

        def run_mini_train_test(base_model: str):
            """Запуск mini-теста обучения в фоне (отдельный процесс), лог в mini_test_log."""
            global _mini_test_state
            if not base_model or "/" not in str(base_model):
                return "❌ Выберите базовую модель в списке.", ""

            _mini_test_state["log_lines"] = []
            _mini_test_state["result"] = None
            _mini_test_state["last_status"] = "⏳ Запускаю mini-тест..."
            _mini_test_state["thread"] = None

            def target():
                import subprocess
                import sys
                # Скрипт лежит в scripts/, запускаем отдельным процессом, чтобы не мешать основному Gradio.
                script_path = str(Path(__file__).resolve().parent / "scripts" / "run_mini_train_test.py")
                cmd = [
                    sys.executable,
                    script_path,
                    "--model",
                    str(base_model),
                    "--steps",
                    "1",
                    "--batch-size",
                    "1",
                    "--lora-rank",
                    "8",
                    "--max-train-examples",
                    "2",
                ]
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(Path(__file__).resolve().parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    _mini_test_state["log_lines"].append(line.rstrip("\n"))
                _mini_test_state["result"] = {"returncode": proc.returncode}
                _mini_test_state["thread"] = None

            t = threading.Thread(target=target, daemon=True)
            _mini_test_state["thread"] = t
            t.start()
            return _mini_test_state["last_status"], ""

        def poll_mini_train_test():
            global _mini_test_state
            t = _mini_test_state.get("thread")
            log_lines = _mini_test_state.get("log_lines", [])
            result = _mini_test_state.get("result")
            log_text = "\n".join(log_lines[-40:]) if log_lines else ""

            if t and t.is_alive():
                return "⏳ Mini-тест идёт...", log_text

            if result is None:
                return _mini_test_state.get("last_status", ""), log_text

            rc = int(result.get("returncode", -1))
            if rc == 0:
                _mini_test_state["last_status"] = "✅ Mini-тест успешно завершён."
            else:
                _mini_test_state["last_status"] = f"❌ Mini-тест упал (exit code {rc})."
            return _mini_test_state["last_status"], log_text

        def export_start(cached):
            global _export_state
            if not cached or (isinstance(cached, (list, tuple)) and len(cached) >= 2 and cached[0] is None and cached[1] is None):
                return "❌ Параметры не получены. Откройте вкладку «Дообучение» и нажмите кнопку экспорта снова.", ""
            adapter_path = (cached[0] or "").strip() if len(cached) > 0 else ""
            ollama_name = (cached[1] or "").strip() if len(cached) > 1 else "clothes-detector-v1"
            name = _proj_name()
            if not (adapter_path or "").strip():
                adapter_path = pm.get_last_lora_path(name) if name else None
                if not adapter_path and name:
                    adapter_path = str(pm.project_dir(name) / "lora_out" / "lora_adapter")
            if not adapter_path or not Path(adapter_path).exists():
                return "❌ Укажите путь к адаптеру или сначала запустите обучение.", ""
            _export_state["log_lines"] = []
            _export_state["result"] = None
            _export_state["ollama_name"] = ollama_name or "clothes-detector-v1"

            def run_export():
                import os
                os.environ.setdefault("HF_HOME", "C:\\ollama")
                os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ.get("HF_HOME", "C:\\ollama"), "hub"))
                from fine_tune.export import export_to_gguf
                def cb(msg):
                    _export_state["log_lines"].append(msg)
                _export_state["result"] = export_to_gguf(
                    adapter_dir=adapter_path,
                    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
                    output_dir=str(Path(adapter_path).parent.parent / "gguf_out"),
                    ollama_model_name=_export_state["ollama_name"],
                    progress_callback=cb,
                )
                _export_state["thread"] = None

            t = threading.Thread(target=run_export, daemon=True)
            _export_state["thread"] = t
            t.start()
            return "⏳ Экспорт запущен. Подождите 10–30 мин — лог обновляется ниже.", "Запуск экспорта...\n"

        def poll_export_status():
            global _export_state
            t = _export_state.get("thread")
            log_lines = _export_state.get("log_lines", [])
            result = _export_state.get("result")
            name = _export_state.get("ollama_name", "")
            log_text = "\n".join(log_lines[-30:]) if log_lines else ""
            if t and t.is_alive():
                return "⏳ **Экспорт идёт** (объединение модели → GGUF → регистрация в Ollama). Лог ниже обновляется каждые 2 сек.", log_text
            if result is None:
                return _export_state.get("last_status", ""), log_text
            if result.get("success"):
                _export_state["last_status"] = (
                    f"✅ Модель `{name}` зарегистрирована в Ollama!\nGGUF: `{result.get('gguf_path','')}`\n"
                    "В Настройках нажмите «Обновить» у списка моделей — новая модель появится в выборе. "
                    "Если модель не загружается — выберите другую или укажите путь к папке с дообученной моделью."
                )
                return _export_state["last_status"], log_text
            _export_state["last_status"] = f"❌ {result.get('error','')}"
            return _export_state["last_status"], log_text

        def _start_full_compare(images_path, out_dir, max_ex, model_before, model_after):
            _eval_stop_event.clear()
            _eval_log_holder.clear()
            _eval_done_holder[0] = None
            _eval_running[0] = True
            t = threading.Thread(
                target=_run_full_compare_thread,
                args=(images_path, out_dir, max_ex, model_before, model_after),
                daemon=True,
            )
            t.start()
            return "⏳ Запущено: сначала «до», потом «после», затем сравнение. Лог ниже.", ""

        def _refresh_eval_models():
            choices = _eval_model_choices()
            saved = pm.get_global_settings().get("model", "qwen3.5:35b")
            upd = gr.update(choices=choices, value=saved)
            return upd, upd

        btn_eval_compare.click(
            _start_full_compare,
            inputs=[eval_images_path, eval_out_dir, eval_max_examples, eval_model_before, eval_model_after],
            outputs=[eval_status, eval_report],
        )
        btn_refresh_eval_models.click(
            _refresh_eval_models,
            outputs=[eval_model_before, eval_model_after],
        )
        def request_stop_eval():
            _eval_stop_event.set()
            return "🛑 Остановка оценки запрошена. Текущая картинка дообработается, затем оценка прервётся."

        btn_eval_stop.click(request_stop_eval, outputs=[eval_status])

        eval_timer = gr.Timer(1)
        eval_timer.tick(
            poll_eval_status,
            inputs=[eval_report],
            outputs=[eval_status, eval_report],
        )

        def request_stop_build():
            stop_build_event.set()
            return "🛑 Остановка запрошена. Сборка прервётся через несколько секунд."

        btn_build_dataset.click(
            build_dataset,
            inputs=[
                include_images_cb,
                external_jsonl_path,
                build_max_examples,
                build_skip_first,
                include_auto_good_cb,
                include_results_queue_cb,
                auto_good_min_conf,
                auto_good_max_n,
                dedupe_train_urls_cb,
                rebuild_eval_anchors_cb,
                eval_anchors_max_n,
                export_low_conf_review_cb,
                low_conf_max_conf,
                low_conf_max_count,
            ],
            outputs=[dataset_status, dataset_progress, finetune_step1_info],
        )
        btn_stop_build.click(request_stop_build, outputs=[dataset_status])
        btn_train.click(
            run_training_start,
            inputs=[train_params_cache],
            outputs=[train_status, train_log, train_progress],
        )
        train_timer = gr.Timer(2)
        train_timer.tick(
            poll_training_status,
            outputs=[train_status, train_log, train_progress],
        )
        btn_stop_train.click(request_stop_train, outputs=[train_status])
        btn_reset_used.click(reset_used_counter, outputs=[train_status])
        btn_mini_test.click(
            run_mini_train_test,
            inputs=[ft_base_model],
            outputs=[mini_test_status, mini_test_log],
        )
        mini_test_timer = gr.Timer(2)
        mini_test_timer.tick(
            poll_mini_train_test,
            outputs=[mini_test_status, mini_test_log],
        )
        btn_export.click(
            export_start,
            inputs=[export_params_cache],
            outputs=[export_status, export_log],
        )
        export_timer = gr.Timer(2)
        export_timer.tick(poll_export_status, outputs=[export_status, export_log])

        return finetune_step1_info, ft_adapter_path


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: Settings
# ═══════════════════════════════════════════════════════════════════════════════

def tab_settings(create_project_vertical_dd: gr.Dropdown):
    with gr.Tab("Настройки") as settings_tab:
        gr.Markdown("## Глобальные настройки (модель, Ollama) и настройки проекта")
        gr.Markdown(
            "**Промпты и пресеты** — на вкладке **«Запуск»**. **Модель и Ollama** — общие для всех проектов. "
            "**Одежда:** умный список + **галочки стандартных атрибутов** (что вообще отправлять в модель) — ниже."
        )
        with gr.Row():
            with gr.Column():
                s_ollama_url = gr.Textbox(
                    label="Ollama URL (общий)",
                    value="http://127.0.0.1:11435",
                    info="По умолчанию :11435 — локальный пул (ollama-queue-proxy) к настоящему Ollama на :11434; "
                    "даёт очередь HTTP и отложенные задания. Прямой :11434 обходит пул.",
                )
                _g = pm.get_global_settings()
                s_app_public_host = gr.Textbox(
                    label="Адрес для доступа с другого ПК (Tailscale)",
                    placeholder="100.115.68.2",
                    value=(_g.get("app_public_host") or "").strip(),
                    info="IP или хост этой машины в Tailscale. С MacBook откройте: http://<этот-адрес>:7860",
                )
                _predefined_models = [
                    "qwen3.5:4b", "qwen3.5:9b", "qwen3.5:35b",
                    "qwen2.5-vl:3b", "qwen2.5-vl:7b", "qwen2.5-vl:72b",
                ]

                def _model_choices(ollama_url: str):
                    """Список моделей: стандартные + все из Ollama (ollama list), чтобы дообученная модель была в списке."""
                    from attribute_detector import ollama_list_models
                    url = (ollama_url or "").strip() or "http://127.0.0.1:11435"
                    from_ollama = ollama_list_models(url, timeout=3)
                    merged = sorted(set(_predefined_models) | set(from_ollama))
                    return merged if merged else _predefined_models

                s_model = gr.Dropdown(
                    label="Модель (общая для всех проектов) — выбор здесь",
                    choices=_model_choices(_g.get("ollama_url", "http://127.0.0.1:11435")),
                    value=_g.get("model", "qwen3.5:35b"),
                    allow_custom_value=True,
                )
                gr.Markdown("Выберите модель из списка или укажите путь к папке с дообученной моделью (например `...\\lora_out_2\\lora_adapter`).")
                gr.Markdown("### Что загружено в память Ollama")
                gr.Markdown("_Чтобы освободить VRAM перед другим проектом: выгрузите ненужную модель или «Освободить всё», затем выберите нужную модель выше и запустите обработку — она подгрузится._")
                ollama_memory_md = gr.Markdown(value="Список моделей в памяти подгружается при открытии настроек и по кнопке «Обновить список». Выберите модель из выпадающего списка и нажмите «Выгрузить выбранную».")
                ollama_unload_dropdown = gr.Dropdown(
                    label="Модель для выгрузки (освободить VRAM) — выберите из списка",
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                )
                with gr.Row():
                    btn_refresh_ollama = gr.Button("Обновить список", size="sm")
                    btn_unload_one = gr.Button("Выгрузить выбранную", size="sm")
                    btn_unload_all = gr.Button("Освободить всё", size="sm")
                s_max_size = gr.Slider(256, 2048, value=1024, step=128, label="Макс. размер картинки (px)")
                gr.Markdown("_Если картинка больше — уменьшается по длинной стороне (пропорционально), без обрезки._")
                gr.Markdown(
                    "**Хочешь быстрее гонять фид — смотри первое поле ниже** (сколько **разных** фото считать **одновременно**). "
                    "Второе поле — только про **одну** картинку (несколько запросов к той же модели по атрибутам/надписям); на скорость по всему фиду почти не влияет."
                )
                s_batch_offer_workers = gr.Number(
                    label="Сколько разных фото обрабатывать одновременно (ускорение фида)",
                    value=int(_g.get("batch_offer_workers", 1)),
                    precision=0,
                    minimum=1,
                    maximum=16,
                    info="**Это то, что даёт ~×2 скорости при значении 2** (две картинки параллельно — одна и та же модель в Ollama, два независимых прогона). "
                    "После дедупа «разное фото» = разная группа. При >1 в статусе запуска будет **«×N фото параллельно»**. "
                    "Если VRAM не хватает — уменьши число или второе поле ниже. Подробный лог пула: `IMAGE_DESC_POOL_VERBOSE=1`.",
                )
                s_max_parallel_vision = gr.Number(
                    label="Сколько запросов к модели одновременно на одно фото",
                    value=int(_g.get("max_parallel_vision", 0)),
                    precision=0,
                    minimum=0,
                    maximum=32,
                    info="Атрибуты/направления/надписи по **одной** картинке могут идти параллельными HTTP к Ollama. "
                    "**Не ускоряет обработку разных фото** — для этого верхнее поле. 0 = без лимита (всё сразу на один кадр); на одной GPU часто 1–2.",
                )
                s_pool_http_concurrency = gr.Number(
                    label="Слотов HTTP у пула Ollama (прокси :11435 → :11434)",
                    value=int(_g.get("ollama_pool_http_concurrency", 3)),
                    precision=0,
                    minimum=1,
                    maximum=8,
                    info="Сколько **тяжёлых** запросов одновременно пускает ollama-queue-proxy на настоящий Ollama. "
                    "При **Сохранить всё** значение уходит в **работающий** пул (`POST /_ollama_queue/http_capacity`) и в `pool_http_concurrency.txt`. "
                    "Если Ollama URL — прямой :11434, пункт не применяется. При OOM уменьши (2) или верхнее поле «разных фото».",
                )
                gr.Markdown(
                    "_Ориентир **~24 ГБ VRAM**, одна GPU — для **верхнего** поля (разные фото): **4B** до **4**; **7–8B** **3**; **9B** **2**; "
                    "**14B** **2**; **35B+** **1**. Если второе поле большое — верхнее лучше уменьшить._"
                )
                btn_apply_batch_workers = gr.Button("Подставить рекомендуемое «разных фото одновременно» по модели", size="sm")
                s_attr_llm_translate = gr.Checkbox(
                    label="Доперевод value через Ollama (если после глоссария осталась латиница)",
                    value=bool(_g.get("attribute_value_llm_translate", False)),
                    info="Батч-текст без картинки: строки с ≥3 латинских букв подряд. Модель — ниже или основная.",
                )
                _attr_tr_ch0 = _model_choices(_g.get("ollama_url", "http://127.0.0.1:11435"))
                _attr_tr_m0 = str(_g.get("attribute_value_translate_model") or "").strip()
                s_attr_translate_model = gr.Dropdown(
                    label="Модель для доперевода (пусто = та же, что «Модель» выше)",
                    choices=_attr_tr_ch0,
                    value=_attr_tr_m0 if _attr_tr_m0 else None,
                    allow_custom_value=True,
                    info="Можно выбрать лёгкую текстовую модель; путь к локальному адаптеру — как у основной модели.",
                )

            with gr.Column():
                _cfg_init = _proj() or {}
                _vinit = (_cfg_init.get("vertical") or "Одежда").strip()
                if _vinit and _vinit not in pm.get_vertical_choices():
                    pm.add_custom_vertical(_vinit)
                _vert_choices_settings = pm.get_vertical_choices()
                _cat_cl0 = pm.default_clothing_attribute_catalog()
                _keys_cl0 = [k for _, k in _cat_cl0]
                _ench0 = _cfg_init.get("clothing_standard_keys_enabled")
                _val_cl0 = (
                    _keys_cl0
                    if _ench0 is None
                    else [
                        k
                        for k in _keys_cl0
                        if k in {str(x).strip() for x in (_ench0 or []) if str(x).strip()}
                    ]
                )
                s_vertical_settings = gr.Dropdown(
                    label="Вертикаль проекта",
                    choices=_vert_choices_settings,
                    value=_vinit if _vinit in _vert_choices_settings else pm.VERTICAL_OTHER,
                    allow_custom_value=True,
                    info="Выберите из списка, **введите своё название в поле** (например «Аптека») или выберите **«Другое»** и укажите название в поле ниже — оно попадёт в **общий список** для всех проектов (`custom_verticals.json`). "
                    "«Одежда» — стандартные атрибуты одежды; любая другая сфера — задания с вкладки **Запуск**.",
                )
                s_vertical_other_settings = gr.Textbox(
                    label="Своё название вертикали (если в списке выбрано «Другое»)",
                    placeholder="Например: Аптека, БАДы, электроника",
                    visible=(_vinit == pm.VERTICAL_OTHER),
                    info="Сохраняется в общий список вертикалей и появится при **создании нового проекта** после «Сохранить всё».",
                )
                s_dynamic_clothing = gr.Checkbox(
                    label="Умный список атрибутов одежды",
                    value=bool(_cfg_init.get("dynamic_clothing_attributes", True)),
                    visible=(_vinit == pm.VERTICAL_CLOTHING),
                    info="Если включено — из **отмеченных ниже** стандартных ключей модель сама выбирает подходящие под тип изделия. Если выключено — в запрос уходят **все отмеченные** стандартные ключи (полный список в JSON). Только для «Одежда».",
                )
                s_clothing_standard_attrs = gr.CheckboxGroup(
                    label="Стандартные атрибуты одежды (что извлекать)",
                    choices=_cat_cl0,
                    value=_val_cl0,
                    visible=(_vinit == pm.VERTICAL_CLOTHING),
                    info="Снимите лишние — их **не будет** в промпте к модели. **Цвет (базовый)** и **Оттенок** — два отдельных ключа: сначала общая гамма (чёрный, синий, бежевый…), затем нюанс (фуксия, пыльная роза…). **Свои** поля в JSON направления «Одежда» (не из этого списка) не отключаются здесь. Пустой список — без шаблонных ключей, только свои атрибуты/custom_prompt.",
                )
                s_extract_inscriptions = gr.Checkbox(
                    label="Искать надписи на товаре",
                    value=bool(_cfg_init.get("extract_inscriptions", True)),
                    info="Если выключено — текст на фото не извлекается. Как именно — см. режим ниже.",
                )
                s_omit_offer_title = gr.Checkbox(
                    label="Не подставлять название оффера в vision-промпт",
                    value=bool(_cfg_init.get("omit_offer_title_in_prompt", False)),
                    info="Для **аптек, упаковок, этикеток**: иначе в промпт попадает title из фида («…»), модель часто **копирует его** вместо OCR. **Старые проекты** с вертикалью **не «Одежда»** без этого поля в `config.json` при открытии получают **вкл** автоматически. Одежда с несколькими вещами на фото — обычно **выкл**.",
                )
                _dup_init = dedupe_mode_from_config(_cfg_init)
                # Gradio: кортеж (подпись в UI, значение в Python). Раньше было наоборот — в конфиг уезжали русские строки и сбрасывались в off.
                s_process_unique_pictures_mode = gr.Radio(
                    label="Дедуп перед vision (не гонять модель на одинаковые снимки)",
                    choices=[
                        ("Выключено", "off"),
                        ("По URL первой картинки (быстро)", "url"),
                        (
                            "По содержимому файла в кэше (dHash; разные URL, тот же файл — один вызов; чуть дольше на группировку)",
                            "phash",
                        ),
                    ],
                    value=_dup_init if _dup_init in ("off", "url", "phash") else "off",
                    info="Режим phash: для каждого оффера один раз открывается уменьшенная копия из кэша (~9×8 px), без держания полного изображения в RAM. Совпадение hash строгое.",
                )
                s_inscription_mode = gr.Radio(
                    label="Режим надписей",
                    choices=[
                        ("Отдельный vision-запрос (можно другая модель ниже)", "separate_call"),
                        ("Та же модель, тот же JSON, что и атрибуты (один запрос на направление)", "same_prompt"),
                    ],
                    value=_canonical_inscription_mode_stored(_cfg_init.get("inscription_mode")),
                    info="Отдельный запрос — как раньше, параллельно атрибутам. «Тот же JSON» — быстрее, но ответ модели должен включать text_found / texts / text_read_confidence.",
                )
                _ins_ch0 = _global_model_choices(_g.get("ollama_url", "http://127.0.0.1:11435"))
                _ins_m0 = str(_cfg_init.get("inscription_model") or "").strip()
                s_inscription_model = gr.Dropdown(
                    label="Модель для надписей (только режим «отдельный запрос»; пусто = основная из глобальных настроек)",
                    choices=_ins_ch0,
                    value=_ins_m0 if _ins_m0 else None,
                    allow_custom_value=True,
                )
                s_conf_threshold = gr.Slider(0, 100, value=50, step=5, label="Порог уверенности (проект)")
                recommended_threshold_md = gr.Markdown(visible=False)
                btn_apply_recommended = gr.Button("Подставить рекомендуемый порог", size="sm")
                gr.Markdown(
                    "**Направления** — наборы атрибутов для анализа (например «Одежда», «Другое»). У каждого: **id**, **name**, **text_enabled**, **attributes**, **custom_prompt**. "
                    "Если не заданы — при загрузке проекта подставляются значения по умолчанию.\n\n"
                    "- **`omit_offer_title_in_prompt`** (направление, опционально): если `true` — для **этого** направления в запрос не передаётся название из фида (имеет смысл вместе с общей галочкой выше или вместо неё для одного направления).\n\n"
                    "- **text_enabled** — по-русски: *«искать на фото текст на товаре»* (надписи, принты, логотипы). Это **не** список базовых атрибутов одежды: базовые атрибуты задаются в **attributes**. "
                    "Для ювелирки и т.п. обычно ставьте `text_enabled`: false.\n\n"
                    "_Масштабирование:_ если в фиде есть не только одежда, но и сумки, бижутерия и т.д. — добавьте новое направление (например id: \"bags\", name: \"Сумки\", attributes: [...]). Модель будет определять атрибуты по фото для каждого направления."
                )
                s_directions_json = gr.Code(
                    label="Направления (проект)",
                    language="json",
                    lines=22,
                )

        def _on_settings_vertical_change(v):
            v = (v or "").strip()
            is_cl = v == pm.VERTICAL_CLOTHING
            is_other = v == pm.VERTICAL_OTHER
            return (
                gr.update(visible=is_cl),
                gr.update(visible=is_cl),
                gr.update(visible=is_other),
            )

        s_vertical_settings.change(
            _on_settings_vertical_change,
            inputs=[s_vertical_settings],
            outputs=[s_dynamic_clothing, s_clothing_standard_attrs, s_vertical_other_settings],
        )

        btn_load_settings = gr.Button("Загрузить настройки", size="sm")
        btn_save_settings = gr.Button("💾 Сохранить всё", variant="primary")
        settings_status = gr.Markdown()

        def _recommended_threshold_and_md():
            rdb = _results_db()
            if not rdb or not rdb.exists():
                return None, gr.update(visible=False)
            con = fc.sqlite_connect(rdb)
            rows = con.execute("SELECT avg_confidence FROM results WHERE avg_confidence > 0 ORDER BY avg_confidence").fetchall()
            con.close()
            if not rows:
                return None, gr.update(visible=False)
            values = [r[0] for r in rows]
            n = len(values)
            p25 = values[max(0, (n * 25) // 100)] if n else 50
            recommended = max(30, min(80, p25))
            return recommended, gr.update(value=f"_Рекомендуемый по результатам: **{recommended}**_", visible=True)

        def _ollama_memory_status(ollama_url: str):
            url = ollama_url or "http://127.0.0.1:11435"
            models = ollama_loaded_models(url)
            if not models:
                return "**В памяти Ollama:** ни одной модели. При первом запросе подгрузится выбранная выше модель.", []
            lines = []
            names = []
            for m in models:
                name = m.get("name") or "?"
                names.append(name)
                size_gb = (m.get("size_vram") or 0) / (1024**3)
                lines.append(f"- **{name}** — {size_gb:.1f} ГБ VRAM")
            return "**В памяти Ollama:**\n\n" + "\n".join(lines), names

        def refresh_ollama_memory(url):
            md, names = _ollama_memory_status(url)
            return md, gr.update(choices=names, value=names[0] if names else None)

        def unload_one_model(url, model):
            if not model:
                md, names = _ollama_memory_status(url or "http://127.0.0.1:11435")
                return md, gr.update(choices=names, value=names[0] if names else None)
            ok, msg = ollama_unload_model(url or "http://127.0.0.1:11435", model)
            md, names = _ollama_memory_status(url or "http://127.0.0.1:11435")
            return md + "\n\n" + ("✅ " if ok else "⚠️ ") + msg, gr.update(choices=names, value=names[0] if names else None)

        def unload_all_ollama(url):
            base = url or "http://127.0.0.1:11435"
            for m in ollama_loaded_models(base):
                n = m.get("name")
                if n:
                    ollama_unload_model(base, n)
            md, names = _ollama_memory_status(base)
            return md + "\n\n✅ Все модели выгружены.", gr.update(choices=names, value=None)

        btn_refresh_ollama.click(
            refresh_ollama_memory,
            inputs=[s_ollama_url],
            outputs=[ollama_memory_md, ollama_unload_dropdown],
        )
        btn_unload_one.click(
            unload_one_model,
            inputs=[s_ollama_url, ollama_unload_dropdown],
            outputs=[ollama_memory_md, ollama_unload_dropdown],
        )
        btn_unload_all.click(
            unload_all_ollama,
            inputs=[s_ollama_url],
            outputs=[ollama_memory_md, ollama_unload_dropdown],
        )

        def load_settings():
            g = pm.get_global_settings()
            cfg = _proj()
            rec_val, rec_md = _recommended_threshold_and_md()
            url = g.get("ollama_url", "http://127.0.0.1:11435")
            mem_md, mem_names = _ollama_memory_status(url)
            mem_dd = gr.update(choices=mem_names, value=mem_names[0] if mem_names else None)
            model_choices = _model_choices(url)
            saved_model = g.get("model", "qwen3.5:35b")
            s_model_up = gr.update(choices=model_choices, value=saved_model if saved_model in model_choices else (model_choices[0] if model_choices else saved_model))
            attr_tr_ch = _model_choices(url)
            attr_tr_saved = str(g.get("attribute_value_translate_model") or "").strip()
            attr_tr_dd = gr.update(
                choices=attr_tr_ch,
                value=attr_tr_saved if attr_tr_saved else None,
            )
            if not cfg:
                th = g.get("confidence_threshold", 50)
                eff_v = "Одежда"
            else:
                th = cfg.get("confidence_threshold", 50)
                eff_v = (cfg.get("vertical") or "Одежда").strip()
                if eff_v and eff_v not in pm.get_vertical_choices():
                    pm.add_custom_vertical(eff_v)
            dc_vis = eff_v == pm.VERTICAL_CLOTHING
            ins_choices = _global_model_choices(url)

            def _cl_std_load_update(c, eff_vertical):
                cat = pm.default_clothing_attribute_catalog()
                kall = [k for _, k in cat]
                vis = eff_vertical == pm.VERTICAL_CLOTHING and bool(c)
                if not c:
                    return gr.update(choices=cat, value=kall, visible=False)
                en = c.get("clothing_standard_keys_enabled")
                val = (
                    kall
                    if en is None
                    else [k for k in kall if k in {str(x).strip() for x in (en or []) if str(x).strip()}]
                )
                return gr.update(choices=cat, value=val, visible=vis)

            if not cfg:
                return (
                    url,
                    (g.get("app_public_host") or "").strip(),
                    s_model_up,
                    g.get("image_max_size", 1024),
                    int(g.get("batch_offer_workers", 1)),
                    int(g.get("max_parallel_vision", 0)),
                    int(g.get("ollama_pool_http_concurrency", 3)),
                    bool(g.get("attribute_value_llm_translate", False)),
                    attr_tr_dd,
                    th,
                    json.dumps(pm.DEFAULT_DIRECTIONS, ensure_ascii=False, indent=2),
                    "_Выберите проект, чтобы загрузить его направления и порог_",
                    rec_md,
                    mem_md,
                    mem_dd,
                    gr.update(value=True, visible=False),
                    _cl_std_load_update(None, eff_v),
                    gr.update(value=True),
                    gr.update(value=False),
                    gr.update(value="off"),
                    gr.update(value="separate_call"),
                    gr.update(choices=ins_choices, value=None),
                    gr.update(choices=pm.get_vertical_choices(), value=pm.VERTICAL_CLOTHING),
                    gr.update(value="", visible=False),
                    gr.update(choices=[CREATE_PROJECT_VERTICAL_PLACEHOLDER] + pm.get_vertical_choices()),
                )
            ins_saved = str(cfg.get("inscription_model") or "").strip()
            vert_choices_now = pm.get_vertical_choices()
            other_vis = eff_v == pm.VERTICAL_OTHER
            return (
                url,
                (g.get("app_public_host") or "").strip(),
                s_model_up,
                g.get("image_max_size", 1024),
                int(g.get("batch_offer_workers", 1)),
                int(g.get("max_parallel_vision", 0)),
                int(g.get("ollama_pool_http_concurrency", 3)),
                bool(g.get("attribute_value_llm_translate", False)),
                attr_tr_dd,
                th,
                json.dumps(cfg.get("directions", []), ensure_ascii=False, indent=2),
                f"Глобальные настройки + проект **{cfg.get('name','')}**",
                rec_md,
                mem_md,
                mem_dd,
                gr.update(
                    value=bool(cfg.get("dynamic_clothing_attributes", True)),
                    visible=dc_vis,
                ),
                _cl_std_load_update(cfg, eff_v),
                gr.update(value=bool(cfg.get("extract_inscriptions", True))),
                gr.update(value=bool(cfg.get("omit_offer_title_in_prompt", False))),
                gr.update(
                    value=(
                        dedupe_mode_from_config(cfg)
                        if dedupe_mode_from_config(cfg) in ("off", "url", "phash")
                        else "off"
                    )
                ),
                gr.update(value=_canonical_inscription_mode_stored(cfg.get("inscription_mode"))),
                gr.update(
                    choices=ins_choices,
                    value=ins_saved if ins_saved else None,
                ),
                gr.update(
                    choices=vert_choices_now,
                    value=eff_v if eff_v in vert_choices_now else pm.VERTICAL_OTHER,
                ),
                gr.update(value="", visible=other_vis),
                gr.update(choices=[CREATE_PROJECT_VERTICAL_PLACEHOLDER] + vert_choices_now),
            )

        def save_settings(
            url,
            app_public_host,
            model,
            max_size,
            batch_offer_workers,
            max_parallel_vision,
            pool_http_concurrency,
            attr_llm_translate,
            attr_translate_model,
            threshold,
            directions_str,
            dynamic_clothing,
            clothing_standard_pick,
            extract_inscriptions,
            omit_offer_title_in_prompt,
            process_unique_pictures_mode,
            inscription_mode,
            inscription_model,
            new_vertical,
            vertical_other_text,
        ):
            global _current_project
            _atm = (attr_translate_model or "")
            if _atm is None:
                _atm = ""
            _atm = str(_atm).strip()
            try:
                _phc = int(pool_http_concurrency)
            except (TypeError, ValueError):
                _phc = 3
            _phc = int(max(1, min(8, _phc)))
            pm.save_global_settings({
                "ollama_url": url,
                "model": model,
                "image_max_size": int(max_size),
                "max_parallel_vision": int(max_parallel_vision or 0),
                "batch_offer_workers": int(max(1, min(16, int(batch_offer_workers or 1)))),
                "ollama_pool_http_concurrency": _phc,
                "app_public_host": (app_public_host or "").strip(),
                "attribute_value_llm_translate": bool(attr_llm_translate),
                "attribute_value_translate_model": _atm,
            })
            _pool_line = ""
            _bu = (url or "").strip()
            if _bu:
                _pst = pool_jobs_client.fetch_pool_status(_bu, timeout_s=3.0)
                if _pst and _pst.get("service") == "ollama_pool":
                    _pok, _pmsg = pool_jobs_client.push_pool_http_capacity(_bu, _phc, timeout_s=10.0)
                    _pool_line = (
                        f"\n\n{'✅' if _pok else '⚠️'} **Пул Ollama:** {_pmsg}"
                    )
                else:
                    _pool_line = (
                        "\n\n_По этому URL не отвечает пул (`/_ollama_queue/status`) — число слотов сохранено только в настройках приложения._"
                    )
            _noop_dd = gr.update()
            name = _proj_name()
            if name:
                try:
                    directions = json.loads(directions_str) if directions_str else []
                except json.JSONDecodeError:
                    return ("❌ Ошибка в JSON направлений", _noop_dd, _noop_dd)
                if not isinstance(directions, list):
                    return ("❌ Направления должны быть массивом", _noop_dd, _noop_dd)
                cur = _proj()
                raw_v = (new_vertical or "").strip()
                if raw_v == pm.VERTICAL_OTHER:
                    eff_v = (vertical_other_text or "").strip()
                    if not eff_v:
                        return (
                            "❌ В списке выбрано **«Другое»** — введите **своё название вертикали** в поле под выпадашкой.",
                            _noop_dd,
                            _noop_dd,
                        )
                else:
                    eff_v = raw_v or (cur.get("vertical") or "Одежда").strip()
                if eff_v and eff_v not in pm.get_vertical_choices():
                    pm.add_custom_vertical(eff_v)
                dyn_val = (
                    bool(dynamic_clothing)
                    if eff_v == pm.VERTICAL_CLOTHING
                    else cur.get("dynamic_clothing_attributes", True)
                )
                imode = _canonical_inscription_mode_stored(inscription_mode)
                imod = (inscription_model or "")
                if imod is None:
                    imod = ""
                imod = str(imod).strip()
                dup_m = (process_unique_pictures_mode or "off").strip().lower()
                if dup_m not in ("off", "url", "phash"):
                    dup_m = "off"
                cfg = {
                    **_proj(),
                    "vertical": eff_v,
                    "confidence_threshold": int(threshold),
                    "directions": directions,
                    "dynamic_clothing_attributes": dyn_val,
                    "extract_inscriptions": bool(extract_inscriptions),
                    "omit_offer_title_in_prompt": bool(omit_offer_title_in_prompt),
                    "process_unique_pictures_mode": dup_m,
                    "inscription_mode": imode,
                    "inscription_model": imod,
                }
                if eff_v == pm.VERTICAL_CLOTHING:
                    keys_all = set(pm.default_clothing_standard_keys())
                    picked = [str(x).strip() for x in (clothing_standard_pick or []) if str(x).strip()]
                    cfg["clothing_standard_keys_enabled"] = None if set(picked) == keys_all else picked
                pm.save_project(cfg)
                _current_project = pm.load_project(name)
                msg = (
                    f"✅ Глобальные настройки сохранены. Настройки проекта **{name}** сохранены."
                    + _pool_line
                )
                vc = pm.get_vertical_choices()
                return (
                    msg,
                    gr.update(choices=[CREATE_PROJECT_VERTICAL_PLACEHOLDER] + vc, value=CREATE_PROJECT_VERTICAL_PLACEHOLDER),
                    gr.update(choices=vc, value=eff_v),
                )
            msg = (
                "✅ Глобальные настройки (модель, Ollama) сохранены. Выберите проект и сохраните ещё раз, чтобы записать направления."
                + _pool_line
            )
            return (msg, _noop_dd, _noop_dd)

        def apply_recommended():
            rec, _ = _recommended_threshold_and_md()
            return rec if rec is not None else 50

        def apply_recommended_batch_workers(model):
            n = pm.recommended_batch_offer_workers(str(model or ""))
            return int(max(1, min(16, n)))

        _settings_load_outputs = [
            s_ollama_url,
            s_app_public_host,
            s_model,
            s_max_size,
            s_batch_offer_workers,
            s_max_parallel_vision,
            s_pool_http_concurrency,
            s_attr_llm_translate,
            s_attr_translate_model,
            s_conf_threshold,
            s_directions_json,
            settings_status,
            recommended_threshold_md,
            ollama_memory_md,
            ollama_unload_dropdown,
            s_dynamic_clothing,
            s_clothing_standard_attrs,
            s_extract_inscriptions,
            s_omit_offer_title,
            s_process_unique_pictures_mode,
            s_inscription_mode,
            s_inscription_model,
            s_vertical_settings,
            s_vertical_other_settings,
            create_project_vertical_dd,
        ]
        btn_load_settings.click(load_settings, outputs=_settings_load_outputs)
        settings_tab.select(load_settings, outputs=_settings_load_outputs)
        btn_apply_recommended.click(apply_recommended, outputs=[s_conf_threshold])
        btn_apply_batch_workers.click(apply_recommended_batch_workers, inputs=[s_model], outputs=[s_batch_offer_workers])
        btn_save_settings.click(
            save_settings,
            inputs=[
                s_ollama_url,
                s_app_public_host,
                s_model,
                s_max_size,
                s_batch_offer_workers,
                s_max_parallel_vision,
                s_pool_http_concurrency,
                s_attr_llm_translate,
                s_attr_translate_model,
                s_conf_threshold,
                s_directions_json,
                s_dynamic_clothing,
                s_clothing_standard_attrs,
                s_extract_inscriptions,
                s_omit_offer_title,
                s_process_unique_pictures_mode,
                s_inscription_mode,
                s_inscription_model,
                s_vertical_settings,
                s_vertical_other_settings,
            ],
            outputs=[settings_status, create_project_vertical_dd, s_vertical_settings],
        )

        return ollama_memory_md, ollama_unload_dropdown


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _get_model_in_memory_badge() -> str:
    """Короткий бейдж для угла экрана: какая модель в памяти Ollama."""
    try:
        url = pm.get_global_settings().get("ollama_url", "http://127.0.0.1:11435")
        models = ollama_loaded_models(url, timeout=2)
        if not models:
            return "<span class=\"header-model-badge\" title=\"При первом запросе подгрузится модель из Настроек\">В памяти: —</span>"
        names = [m.get("name") or m.get("model") or "?" for m in models]
        text = ", ".join(names) if len(names) <= 2 else f"{names[0]} +{len(names)-1}"
        return f"<span class=\"header-model-badge\" title=\"{html.escape(', '.join(names))}\">В памяти: {html.escape(text)}</span>"
    except Exception:
        return "<span class=\"header-model-badge\" title=\"Ollama недоступна\">В памяти: ?</span>"


def _get_initial_ollama_memory():
    """При загрузке приложения заполнить блок «Что загружено в память» и выпадающий список."""
    url = pm.get_global_settings().get("ollama_url", "http://127.0.0.1:11435")
    models = ollama_loaded_models(url)
    if not models:
        return "**В памяти Ollama:** ни одной модели. Нажмите «Обновить список» после запуска модели.", gr.update(choices=[], value=None)
    lines = []
    names = []
    for m in models:
        name = m.get("name") or m.get("model") or "?"
        names.append(name)
        size_gb = (m.get("size_vram") or 0) / (1024**3)
        lines.append(f"- **{name}** — {size_gb:.1f} ГБ VRAM")
    md = "**В памяти Ollama:**\n\n" + "\n".join(lines)
    return md, gr.update(choices=names, value=names[0] if names else None)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Image analyzer") as app:
        with gr.Row(elem_classes=["header-row"]):
            gr.Markdown(
                "# 🖼️ Image analyzer\n"
                "Атрибуты товара и надписи на фото по фиду (одежда, ювелирка и др.)"
            )
            header_project = gr.Markdown("_Не выбран_", elem_id="header-project", elem_classes=["header-project-badge"])
            header_model_badge = gr.HTML(value="", elem_id="header-model-badge")
        create_project_vertical_dd = tab_projects(app, header_project)
        tab_feed()
        ollama_status_html, ollama_pool_status_html, run_model_hint = tab_run(header_model_badge)
        tab_results()
        finetune_step1_info, ft_adapter_path = tab_finetune()
        ollama_memory_md, ollama_unload_dropdown = tab_settings(create_project_vertical_dd)

        def _get_finetune_initial():
            return get_finetune_step1_info(), pm.get_last_lora_path(_proj_name()) or ""

        def _load_all_ollama_on_start():
            """Один HTTP-запрос к Ollama при загрузке страницы — заполняем все компоненты сразу."""
            g = pm.get_global_settings()
            url = g.get("ollama_url", "http://127.0.0.1:11435")
            configured_model = g.get("model", "?")
            check_url = normalize_ollama_url(url).rstrip("/") + "/"
            ollama_ok = False
            models = []
            try:
                r = requests.get(check_url, timeout=ollama_root_health_timeout_s(url))
                r.raise_for_status()
                ollama_ok = True
                models = ollama_loaded_models(url, timeout=5)
            except Exception:
                pass

            names = [m.get("name") or m.get("model") or "?" for m in models] if models else []

            # ── header badge ─────────────────────────────────────────────────
            if not names:
                badge = "<span class=\"header-model-badge\" title=\"В памяти пусто\">В памяти: —</span>"
            else:
                text = ", ".join(names) if len(names) <= 2 else f"{names[0]} +{len(names)-1}"
                badge = f"<span class=\"header-model-badge\" title=\"{html.escape(', '.join(names))}\">В памяти: {html.escape(text)}</span>"

            # ── ollama status html ────────────────────────────────────────────
            if not ollama_ok:
                status_html = (
                    "<div style='padding:8px 14px;border-radius:6px;background:#fee2e2;"
                    "border:1px solid #fca5a5;color:#991b1b;font-weight:500'>"
                    f"❌ Ollama недоступен ({url})<br>"
                    "<small>Установите Ollama: <a href='https://ollama.com/download' target='_blank'>"
                    "ollama.com/download</a> → запустите установщик → перезапустите run.bat</small></div>"
                )
            else:
                if names:
                    parts = [
                        f"<b>{html.escape(m.get('name', '?'))}</b> ({(m.get('size_vram') or 0) / (1024**3):.1f} ГБ)"
                        for m in models
                    ]
                    in_mem = "В памяти: " + ", ".join(parts) + ". Выгрузка — в Настройки."
                else:
                    in_mem = "В памяти пусто (модель подгрузится при первом запросе)."
                status_html = (
                    "<div style='padding:8px 14px;border-radius:6px;background:#dcfce7;"
                    "border:1px solid #86efac;color:#166534;font-weight:500'>"
                    f"✅ Ollama доступен — {url}<br><small>{in_mem}</small></div>"
                )

            # ── run model hint ────────────────────────────────────────────────
            in_mem_str = ", ".join(names) if names else "ни одной"
            if names and in_mem_str not in ("ни одной", "—"):
                hint = (
                    f"**Модель для обработки** (из Настроек): **{configured_model}** — при запуске запрос пойдёт именно в неё. "
                    f"Сейчас в памяти Ollama: **{in_mem_str}**. Если нужно обрабатывать другой моделью — смените в **«Настройки»**."
                )
            else:
                hint = (
                    f"**Модель для обработки** (из Настроек): **{configured_model}**. В памяти Ollama сейчас пусто — "
                    "при первом запросе подгрузится эта модель. Изменить — вкладка **«Настройки»** (qwen3.5:4b/9b/35b или своя)."
                )

            # ── settings memory block ─────────────────────────────────────────
            if not names:
                mem_md = "**В памяти Ollama:** ни одной модели. Нажмите «Обновить список» после запуска модели."
                mem_dd = gr.update(choices=[], value=None)
            else:
                lines = []
                for m in models:
                    n = m.get("name") or m.get("model") or "?"
                    size_gb = (m.get("size_vram") or 0) / (1024**3)
                    lines.append(f"- **{n}** — {size_gb:.1f} ГБ VRAM")
                mem_md = "**В памяти Ollama:**\n\n" + "\n".join(lines)
                mem_dd = gr.update(choices=names, value=names[0])

            pool_html = format_ollama_pool_status_html(url)
            return badge, status_html, pool_html, hint, mem_md, mem_dd

        app.load(_get_finetune_initial, outputs=[finetune_step1_info, ft_adapter_path])
        app.load(
            _load_all_ollama_on_start,
            outputs=[
                header_model_badge,
                ollama_status_html,
                ollama_pool_status_html,
                run_model_hint,
                ollama_memory_md,
                ollama_unload_dropdown,
            ],
        )
    return app


def _gradio_bind_host(server_name: str) -> str:
    """Хост для проверки/захвата порта так же, как у Gradio (0.0.0.0 = все интерфейсы)."""
    s = (server_name or "").strip()
    return "0.0.0.0" if s in ("", "0.0.0.0") else s


def _tcp_port_available(bind_host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((bind_host, port))
        return True
    except OSError:
        return False


def _pids_listening_on_port_tcp(port: int) -> list[int]:
    """PID процессов с TCP LISTEN на указанном порту (Windows: netstat; Linux/macOS: lsof)."""
    p = str(port)
    out: list[int] = []

    if sys.platform == "win32":
        kwargs: dict = {
            "args": ["netstat", "-ano", "-p", "TCP"],
            "capture_output": True,
            "text": True,
            "timeout": 20,
        }
        cf = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if cf:
            kwargs["creationflags"] = cf
        try:
            r = subprocess.run(**kwargs)
        except (OSError, subprocess.TimeoutExpired):
            return out
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("TCP"):
                continue
            if "LISTENING" not in line and "СЛУШАЕТ" not in line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            local = parts[1]
            _, _, port_part = local.rpartition(":")
            if port_part != p:
                continue
            try:
                pid = int(parts[-1])
            except ValueError:
                continue
            if pid > 0:
                out.append(pid)
        return sorted(set(out))

    try:
        r = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        return out
    if r.returncode != 0:
        return out
    for line in (r.stdout or "").strip().split():
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid > 0:
            out.append(pid)
    return sorted(set(out))


def _terminate_pids_listening_on_port(port: int) -> list[int]:
    """Завершает процессы, которые слушают ``port`` (кроме текущего). Нужно при повторном запуске Gradio."""
    import signal

    me = os.getpid()
    pids = [x for x in _pids_listening_on_port_tcp(port) if x != me]
    killed: list[int] = []
    for pid in pids:
        if sys.platform == "win32":
            kwargs = {
                "args": ["taskkill", "/PID", str(pid), "/F"],
                "capture_output": True,
                "timeout": 20,
            }
            cf = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if cf:
                kwargs["creationflags"] = cf
            try:
                r = subprocess.run(**kwargs)
                if r.returncode == 0:
                    killed.append(pid)
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except (OSError, ProcessLookupError):
                pass
    if killed:
        time.sleep(0.4)
    return killed


def _ensure_gradio_port_free(port: int, bind_host: str) -> None:
    """Если порт занят (часто — зависший прошлый запуск), освобождаем и ждём."""
    if _tcp_port_available(bind_host, port):
        return
    pids = [x for x in _pids_listening_on_port_tcp(port) if x != os.getpid()]
    if not pids:
        return
    print(f"[INFO] Порт {port} занят (PID: {', '.join(map(str, pids))}). Завершаем процесс(ы)...")
    _terminate_pids_listening_on_port(port)


def _find_free_port(start: int = 7860, end: int = 7870, server_name: str = "0.0.0.0") -> int:
    """Первый свободный порт в [start, end]; проверка bind на том же хосте, что и Gradio."""
    bind_host = _gradio_bind_host(server_name)
    for port in range(start, end + 1):
        if _tcp_port_available(bind_host, port):
            return port
    return start


# Скрипт: при обновлении значения лога не сбрасывать прокрутку вверх, если пользователь читает середину;
# у нижнего края — прокрутка к последней строке.
RUN_LOG_SCROLL_HEAD = """
<script>
(function () {
  const BOX_ID = "image-desc-run-log";
  const BOTTOM_EPS = 56;
  const TICK_MS = 100;
  function textarea() {
    var box = document.getElementById(BOX_ID);
    if (!box) return null;
    if (box.tagName === "TEXTAREA") return box;
    return box.querySelector("textarea");
  }
  function arm() {
    var ta = textarea();
    if (!ta || ta.dataset._runLogSmartScroll === "1") return;
    ta.dataset._runLogSmartScroll = "1";
    var lastVal = ta.value;
    var prevSt = 0, prevSh = 1, prevCh = 1;
    setInterval(function () {
      var el = textarea();
      if (!el) return;
      if (el.value === lastVal) {
        prevSt = el.scrollTop;
        prevSh = el.scrollHeight;
        prevCh = el.clientHeight;
        return;
      }
      var osh = prevSh, ost = prevSt, och = prevCh;
      var wasAtBottom = osh - ost - och < BOTTOM_EPS;
      var omax = Math.max(1, osh - och);
      var ratio = omax > 0 ? Math.min(1, Math.max(0, ost / omax)) : 0;
      lastVal = el.value;
      requestAnimationFrame(function () {
        var nsh = el.scrollHeight, nch = el.clientHeight, nmax = Math.max(0, nsh - nch);
        if (wasAtBottom || nmax <= 0) {
          el.scrollTop = nsh;
        } else {
          el.scrollTop = ratio * nmax;
        }
      });
    }, TICK_MS);
  }
  arm();
  new MutationObserver(arm).observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
"""


# CSS для Gradio (вынесено из try, иначе парсер может споткнуться о многострочную строку рядом с except)
GRADIO_APP_CSS = """
        .gradio-container { max-width: 1280px !important; margin-left: auto !important; margin-right: auto !important; }
        main .block { max-width: 100% !important; }
        footer { display: none !important; }
        .gradio-container input[type="text"], .gradio-container textarea { min-width: 280px !important; }
        .gradio-container label + div input, .gradio-container .input-text { width: 100% !important; max-width: 100% !important; box-sizing: border-box !important; }
        .project-status { margin-top: 0.75rem !important; padding: 0.75rem 1rem !important; background: #f5f5f5 !important; border-radius: 8px !important; min-height: 2.5rem !important; }
        .run-categories-list { max-height: 280px !important; overflow-y: auto !important; }
        /* Вкладка «Запуск»: две колонки (промпт/категории | лог) — равная ширина, без узкого «ступеньки» */
        .run-tab-main-row { display: flex !important; flex-wrap: nowrap !important; align-items: flex-start !important; width: 100% !important; }
        .run-tab-main-row > div { flex: 1 1 0 !important; min-width: 0 !important; max-width: 50% !important; }
        .run-tab-filters-row { display: flex !important; flex-wrap: wrap !important; align-items: flex-start !important; width: 100% !important; gap: 0.5rem !important; }
        .run-tab-filters-row > div { flex: 1 1 320px !important; min-width: min(100%, 280px) !important; }
        .header-row { display: flex !important; align-items: center !important; width: 100% !important; }
        .header-row > div:nth-child(2) { margin-left: auto !important; }
        #header-project { margin-right: 1rem !important; }
        .header-project-badge { align-self: center !important; white-space: nowrap !important; }
        #header-model-badge, .header-model-badge { font-size: 12px !important; color: #666 !important; align-self: center !important; white-space: nowrap !important; padding: 4px 8px !important; background: #f0f0f0 !important; border-radius: 6px !important; }
        .result-card-row {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: flex-start !important;
            width: 100% !important;
            gap: 8px !important;
            position: relative !important;
        }
        /* Галочка: не занимает отдельную широкую колонку — «в углу» поверх карточки */
        .result-card-row > div:nth-child(1) {
            flex: 0 0 0 !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: visible !important;
            position: relative !important;
            z-index: 6 !important;
        }
        .result-card-row > div:nth-child(1) .block,
        .result-card-row > div:nth-child(1) .wrap {
            padding: 0 !important;
            margin: 0 !important;
            min-height: 0 !important;
        }
        .result-card-row > div:nth-child(1) .block {
            position: absolute !important;
            left: 10px !important;
            top: 10px !important;
            width: 22px !important;
            min-width: 22px !important;
            max-width: 28px !important;
        }
        .result-card-row > div:nth-child(1) input[type="checkbox"] {
            width: 18px !important;
            height: 18px !important;
            margin: 0 !important;
            cursor: pointer !important;
        }
        .result-card-row > div:nth-child(2) {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            align-self: flex-start !important;
        }
        .card-actions-col {
            flex: 0 0 auto !important;
            align-self: flex-start !important;
            padding-top: 2px !important;
        }
        .card-actions-inner {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            flex-wrap: nowrap !important;
            gap: 4px !important;
            justify-content: flex-end !important;
        }
        .card-actions-inner button, .card-edit-btn, .card-delete-btn {
            flex-shrink: 0 !important;
            min-width: 44px !important;
            height: 36px !important;
        }
        /* Полноэкранный просмотр картинки на «Запуск»: крупнее в модальном окне */
        .lightbox img, [class*="lightbox"] img, .modal-lg img, div[role="dialog"] img {
            max-width: min(96vw, 1400px) !important;
            max-height: 90vh !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
        }
        #run-current-image img, #run-current-image .image-frame, #run-current-image video {
            max-height: min(55vh, 480px) !important;
            object-fit: contain !important;
        }
        /* Лог обработки: фиксированная высота, прокрутка внутри (новые строки внизу) */
        #image-desc-run-log textarea,
        .run-log-textbox textarea {
            min-height: 220px !important;
            max-height: min(52vh, 420px) !important;
            overflow-y: auto !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
            font-size: 12px !important;
            line-height: 1.4 !important;
        }
"""


if __name__ == "__main__":
    app = build_app()
    app.queue()  # нужно для автообновления по таймеру на вкладке «Запуск»
    # 0.0.0.0 = локально + по Tailscale с другого компа
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    bind_host = _gradio_bind_host(server_name)
    explicit_port = int(os.environ.get("GRADIO_SERVER_PORT", 0)) or None
    if explicit_port:
        _ensure_gradio_port_free(explicit_port, bind_host)
        port = explicit_port
        if not _tcp_port_available(bind_host, port):
            print(
                f"[WARN] Порт {port} всё ещё занят. Ищем свободный в диапазоне 7860–7870..."
            )
            port = _find_free_port(server_name=server_name)
    else:
        _ensure_gradio_port_free(7860, bind_host)
        port = _find_free_port(server_name=server_name)
        _ensure_gradio_port_free(port, bind_host)
        if not _tcp_port_available(bind_host, port):
            port = _find_free_port(server_name=server_name)
    print("Starting Image analyzer ...")
    print(f"  Локально:        http://127.0.0.1:{port}")
    # Адрес для доступа с MacBook/другого ПК: из env или из сохранённых настроек
    public_host = os.environ.get("APP_PUBLIC_HOST", "").strip()
    if not public_host and Path(__file__).resolve().parent:
        try:
            settings_path = Path(__file__).resolve().parent / "app_settings.json"
            if settings_path.exists():
                with open(settings_path, encoding="utf-8") as f:
                    public_host = (json.load(f).get("app_public_host") or "").strip()
        except Exception:
            pass
    if public_host:
        print(f"  С другого ПК:    http://{public_host}:{port}")
    else:
        print(f"  По Tailscale:    http://<Tailscale-IP-этого-ПК>:{port}")
    # Лог: что сейчас в памяти Ollama (из настроек URL)
    try:
        settings_path = Path(__file__).resolve().parent / "app_settings.json"
        ollama_url = "http://127.0.0.1:11435"
        if settings_path.exists():
            with open(settings_path, encoding="utf-8") as f:
                ollama_url = json.load(f).get("ollama_url", ollama_url)
        models = ollama_loaded_models(ollama_url, timeout=3)
        if models:
            names = [m.get("name", "?") for m in models if isinstance(m, dict)]
            print(f"  Ollama в памяти: {', '.join(names)}")
        else:
            print("  Ollama в памяти: ни одной модели (при первом запросе подгрузится модель из настроек)")
    except Exception:
        print("  Ollama в памяти: не удалось получить список")
    try:
        app.launch(
            server_name=server_name,
            server_port=port,
            share=False,
            inbrowser=True,
            theme=gr.themes.Soft(),
            css=GRADIO_APP_CSS,
            head=RUN_LOG_SCROLL_HEAD,
        )
    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C.")
        sys.exit(0)
