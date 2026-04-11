#!/usr/bin/env python3
"""
Убрать из сохранённых результатов заглушки (неизвестно, неузнаваемый, известняк…)
в direction_attributes — та же логика, что strip_placeholder_attribute_values_inplace.

По умолчанию обрабатывается ТОЛЬКО проект zolla (остальные имена отклоняются).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import feed_cache as fc
import project_manager as pm

# Скрипт намеренно ограничен одним проектом (см. запрос пользователя).
ALLOWED_PROJECTS = frozenset({"zolla"})

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
    rows = con.execute("PRAGMA table_info(results)").fetchall()
    cols = {r[1] for r in rows}
    if "model_name" not in cols:
        con.execute("ALTER TABLE results ADD COLUMN model_name TEXT DEFAULT ''")


def _normalize_attrs_raw(raw_attrs: dict) -> dict:
    if raw_attrs and not ("clothing" in raw_attrs or "other" in raw_attrs):
        return {"clothing": raw_attrs}
    return raw_attrs


def _recompute_avg_confidence(direction_attributes: dict, text_detection: dict) -> int:
    scores: list[int] = []
    td = text_detection or {}
    if isinstance(td, dict) and not td.get("error") and td.get("confidence") is not None:
        scores.append(int(td["confidence"]))
    for dr in (direction_attributes or {}).values():
        if not isinstance(dr, dict) or dr.get("error"):
            continue
        for k, v in dr.items():
            if k == "error" or not isinstance(v, dict):
                continue
            if "confidence" in v:
                scores.append(int(v["confidence"]))
    return int(sum(scores) / len(scores)) if scores else 0


def migrate_project_results(project: str, dry_run: bool, verbose: bool) -> tuple[int, int]:
    """
    Возвращает (всего строк, изменённых строк).
    """
    db_path = pm.results_db_path(project)
    if not db_path.exists():
        print(f"Нет файла результатов: {db_path}", file=sys.stderr)
        return 0, 0

    con = fc.sqlite_connect(db_path)
    con.executescript(RESULTS_SCHEMA)
    _migrate_results_db(con)
    con.commit()
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT offer_id, text_json, attributes_json, avg_confidence FROM results").fetchall()
    total = len(rows)
    updated = 0

    for row in rows:
        offer_id = row["offer_id"]
        text_detection = json.loads(row["text_json"] or "{}")
        raw_attrs = json.loads(row["attributes_json"] or "{}")
        attrs = _normalize_attrs_raw(dict(raw_attrs))
        before = json.dumps(attrs, ensure_ascii=False, sort_keys=True)
        pm.strip_placeholder_attribute_values_inplace(attrs)
        after = json.dumps(attrs, ensure_ascii=False, sort_keys=True)
        if before == after:
            continue
        new_avg = _recompute_avg_confidence(attrs, text_detection)
        updated += 1
        if dry_run and verbose:
            print(f"[dry-run] {offer_id}: avg {row['avg_confidence']} -> {new_avg}")
            continue
        con.execute(
            "UPDATE results SET attributes_json = ?, avg_confidence = ? WHERE offer_id = ?",
            (json.dumps(attrs, ensure_ascii=False), new_avg, offer_id),
        )

    if not dry_run:
        con.commit()
    con.close()
    return total, updated


def main() -> int:
    p = argparse.ArgumentParser(description="Очистить заглушки в results.db (только проект zolla).")
    p.add_argument(
        "--project",
        default="zolla",
        help="Имя проекта (разрешено только zolla).",
    )
    p.add_argument("--dry-run", action="store_true", help="Только показать, что изменилось бы, без записи в БД.")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="В dry-run печатать каждую изменённую строку (иначе только итог).",
    )
    args = p.parse_args()

    proj = (args.project or "").strip()
    if proj not in ALLOWED_PROJECTS:
        print(
            f"Отказ: скрипт разрешён только для проектов {sorted(ALLOWED_PROJECTS)}. "
            f"Передано: {proj!r}.",
            file=sys.stderr,
        )
        return 2

    pdir = pm.project_dir(proj)
    if not pdir.is_dir():
        print(f"Отказ: папка проекта не найдена: {pdir}", file=sys.stderr)
        return 1

    total, changed = migrate_project_results(proj, dry_run=bool(args.dry_run), verbose=bool(args.verbose))
    mode = "dry-run" if args.dry_run else "запись"
    print(f"Проект: {proj} ({mode})")
    print(f"Строк в results: {total}, изменено: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
