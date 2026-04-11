#!/usr/bin/env python3
"""
Сравнительный прогон vision-моделей Ollama на одном наборе офферов (время + краткий результат).

  python scripts/bench_vision_models.py --project MyProj --models qwen3.5:9b qwen3.5:35b qwen3.5:35b-a3b --limit 10

Если в БД результатов мало строк — дополняется встроенным демо-набором URL.
"""
from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import project_manager as pm
from attribute_detector import analyze_offer, warmup_ollama_model

DEFAULT_MODELS = ["qwen3.5:4b", "qwen3.5:9b", "qwen3.5:27b", "qwen3.5:35b", "qwen3.5:35b-a3b"]

# Демо-офферы, если в проекте нет сохранённых результатов
_DEMO_OFFERS: list[dict] = [
    {
        "offer_id": "demo_zolla_card",
        "name": "Вязаный кардиган на пуговицах",
        "category": "Женская / Трикотаж / Кардиганы",
        "picture_urls": ["https://zolla.com/upload/iplm/images/goods/013436463033_80N0.jpg"],
    },
    {
        "offer_id": "demo_tsum_coat",
        "name": "Пальто",
        "category": "Женское / Пальто",
        "picture_urls": ["https://collect-static.tsum.com/sig/872f8cfa9b75bd10ebde712c5f8f07a9/height/1526/document-hub/01HA2J4G7QJK3KW2DA8WWNFP9M.jpg"],
    },
]


def fetch_offers_from_results_db(project_name: str, limit: int) -> list[dict]:
    rdb = pm.results_db_path(project_name)
    if not rdb.exists():
        return []
    con = sqlite3.connect(str(rdb))
    try:
        rows = con.execute(
            "SELECT offer_id, name, category, picture_url, error FROM results "
            "WHERE IFNULL(picture_url,'') != '' ORDER BY run_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        con.close()
    out: list[dict] = []
    for oid, name, cat, pic, err in rows:
        if err:
            continue
        out.append(
            {
                "offer_id": str(oid or ""),
                "name": str(name or ""),
                "category": str(cat or ""),
                "picture_urls": [str(pic).strip()] if pic else [],
            }
        )
    return [o for o in out if o["picture_urls"]]


def merge_offers_with_demo(db_offers: list[dict], limit: int) -> list[dict]:
    seen = {o["offer_id"] for o in db_offers}
    merged = list(db_offers)
    for d in _DEMO_OFFERS:
        if len(merged) >= limit:
            break
        if d["offer_id"] not in seen:
            merged.append(d)
            seen.add(d["offer_id"])
    return merged[:limit]


def _summarize_clothing(attrs: dict | None) -> str:
    if not attrs:
        return ""
    c = attrs.get("clothing") or {}
    if not isinstance(c, dict):
        return ""
    parts = []
    for key in ("fastener", "print_pattern", "collar", "material"):
        b = c.get(key)
        if isinstance(b, dict):
            v = (b.get("value") or "").strip()
            if v:
                parts.append(f"{key}={v[:40]}")
    return "; ".join(parts[:6])


def build_config_for_bench(project_cfg: dict, model: str, ollama_url: str) -> dict:
    dirs = copy.deepcopy([d for d in (project_cfg.get("directions") or pm.DEFAULT_DIRECTIONS) if d.get("id") == "clothing"])
    if not dirs:
        dirs = copy.deepcopy([d for d in pm.DEFAULT_DIRECTIONS if d.get("id") == "clothing"])
    for d in dirs:
        d["text_enabled"] = False
    gs = pm.get_global_settings()
    name = (project_cfg.get("name") or "").strip()
    cache = str(pm.image_cache_dir(name)) if name else ""
    return {
        **project_cfg,
        "model": model,
        "ollama_url": ollama_url,
        "image_max_size": int(gs.get("image_max_size") or 1024),
        "directions": dirs,
        "dynamic_clothing_attributes": bool(project_cfg.get("dynamic_clothing_attributes", False)),
        "extract_inscriptions": False,
        "inscription_mode": "separate_call",
        "attribute_value_llm_translate": bool(gs.get("attribute_value_llm_translate", False)),
        "attribute_value_translate_model": str(gs.get("attribute_value_translate_model") or ""),
        "max_parallel_vision": 1,
        "image_cache_dir": cache or None,
    }


def run_benchmark(
    project_name: str,
    models: list[str],
    limit: int,
    ollama_url: str | None = None,
    *,
    progress_cb: callable | None = None,
    skip_warmup: bool = False,
) -> dict:
    """
    Возвращает:
      summary_rows: [{model, wall_ms_total, ms_per_offer, n_ok, n_err, errors[]}]
      per_model: { model: [{offer_id, ms, error, summary}] }
    """
    if not project_name:
        raise ValueError("project_name required")
    gs = pm.get_global_settings()
    base_url = (ollama_url or gs.get("ollama_url") or "http://127.0.0.1:11435").strip()
    proj = pm.load_project(project_name)
    offers = fetch_offers_from_results_db(project_name, limit)
    offers = merge_offers_with_demo(offers, limit)
    if not offers:
        raise ValueError("Нет офферов с picture_url — сначала запустите анализ на вкладке «Запуск».")

    summary_rows: list[dict] = []
    per_model: dict[str, list[dict]] = {}

    for model in models:
        m = (model or "").strip()
        if not m:
            continue
        cfg = build_config_for_bench(proj, m, base_url)
        if not skip_warmup:
            if progress_cb:
                progress_cb(f"Прогрев {m}…")
            try:
                warmup_ollama_model({"model": m, "ollama_url": base_url}, timeout=300)
            except Exception as e:
                if progress_cb:
                    progress_cb(f"Прогрев {m}: {e}")
        rows_m: list[dict] = []
        t0 = time.perf_counter()
        n_ok = n_err = 0
        errs: list[str] = []
        for off in offers:
            oid = off.get("offer_id", "")
            if progress_cb:
                progress_cb(f"{m} · {oid}")
            t1 = time.perf_counter()
            try:
                r = analyze_offer(off, cfg, timeout=300)
                ms = (time.perf_counter() - t1) * 1000
                err = r.get("error")
                da = r.get("direction_attributes") or {}
                summ = _summarize_clothing(da)
                if err:
                    n_err += 1
                    errs.append(f"{oid}: {err}")
                    rows_m.append({"offer_id": oid, "ms": round(ms, 1), "error": err, "summary": ""})
                else:
                    n_ok += 1
                    rows_m.append({"offer_id": oid, "ms": round(ms, 1), "error": None, "summary": summ})
            except Exception as e:
                ms = (time.perf_counter() - t1) * 1000
                n_err += 1
                errs.append(f"{oid}: {e}")
                rows_m.append({"offer_id": oid, "ms": round(ms, 1), "error": str(e), "summary": ""})
        wall = (time.perf_counter() - t0) * 1000
        n = len(offers)
        per_model[m] = rows_m
        summary_rows.append(
            {
                "model": m,
                "wall_ms_total": round(wall, 1),
                "ms_per_offer": round(wall / max(n, 1), 1),
                "n_offers": n,
                "n_ok": n_ok,
                "n_err": n_err,
                "errors": errs[:12],
            }
        )

    return {
        "project": project_name,
        "ollama_url": base_url,
        "n_offers": len(offers),
        "offer_ids": [o.get("offer_id") for o in offers],
        "summary_rows": summary_rows,
        "per_model": per_model,
    }


def format_markdown_table(result: dict) -> str:
    lines = [
        f"**Проект:** `{result.get('project')}` · **офферов:** {result.get('n_offers')} · **Ollama:** `{result.get('ollama_url')}`",
        "",
        "| Модель | Всего (мс) | мс/оффер | OK | Ошибок |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in result.get("summary_rows") or []:
        lines.append(
            f"| `{r['model']}` | {r['wall_ms_total']:.0f} | {r['ms_per_offer']:.0f} | {r['n_ok']} | {r['n_err']} |"
        )
    lines.append("")
    lines.append("**Детали по первому офферу в прогоне (сводка атрибутов):**")
    oids = result.get("offer_ids") or []
    oid0 = oids[0] if oids else ""
    for m, rows in (result.get("per_model") or {}).items():
        row0 = next((x for x in rows if x.get("offer_id") == oid0), rows[0] if rows else None)
        if row0:
            lines.append(f"- `{m}`: {row0.get('summary') or row0.get('error') or '—'}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Сравнение vision-моделей на N офферах из БД результатов.")
    ap.add_argument("--project", "-p", required=True, help="Имя проекта (папка в projects/)")
    ap.add_argument("--models", "-m", nargs="+", default=DEFAULT_MODELS, help="Имена моделей Ollama")
    ap.add_argument("--limit", "-n", type=int, default=10, help="Сколько офферов взять из БД (свежие)")
    ap.add_argument("--ollama-url", default=None, help="URL Ollama (по умолчанию из app_settings)")
    ap.add_argument("--skip-warmup", action="store_true", help="Не вызывать warmup перед каждой моделью")
    ap.add_argument("--json-out", default="", help="Записать полный JSON в файл")
    args = ap.parse_args()
    try:
        r = run_benchmark(
            args.project,
            list(args.models),
            max(1, int(args.limit)),
            args.ollama_url,
            skip_warmup=args.skip_warmup,
            progress_cb=lambda s: print(s, flush=True),
        )
    except Exception as e:
        print("Ошибка:", e, file=sys.stderr)
        return 1
    print(format_markdown_table(r))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print("JSON:", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
