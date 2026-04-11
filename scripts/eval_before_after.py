#!/usr/bin/env python3
"""
Evaluate model outputs before and after fine-tuning on the same images.

Usage:
  1. python scripts/eval_before_after.py --images data/deepfashion/sample_50.jsonl --phase before --out results
  2. (Run fine-tuning and export to Ollama, switch model in app settings)
  3. python scripts/eval_before_after.py --images data/deepfashion/sample_50.jsonl --phase after --out results
  4. python scripts/eval_before_after.py --out results --phase compare

--images: path to JSONL with "image_path" (and optionally "attributes" for ground truth), or a directory of images.
--phase: before | after | compare
--out: directory for eval_before.json, eval_after.json and report.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _collect_image_paths(images_arg: str) -> list[dict]:
    """Return list of {image_path, attributes (optional)}."""
    p = Path(images_arg)
    if p.is_file() and p.suffix.lower() == ".jsonl":
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out.append({"image_path": row.get("image_path", ""), "attributes": row.get("attributes", {})})
        return out
    if p.is_dir():
        return [{"image_path": str(f), "attributes": {}} for f in sorted(p.iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    return []


def _run_inference(
    image_paths: list[dict],
    model: str,
    ollama_url: str,
    config: dict,
    progress_callback=None,
    stop_event=None,
) -> list[dict]:
    """Run attribute detection on each image; return list of {image_path, response_attributes, error}."""
    from attribute_detector import analyze_offer
    import project_manager as pm

    cache_dir = Path(config.get("image_cache_dir", ROOT / "data" / "eval_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Use default directions so we have attributes to detect
    run_config = {
        **config,
        "image_cache_dir": str(cache_dir),
        "directions": config.get("directions") or pm.DEFAULT_DIRECTIONS,
    }
    results = []
    total = len(image_paths)
    for i, item in enumerate(image_paths):
        if stop_event and stop_event.is_set():
            if progress_callback:
                progress_callback(f"Остановлено пользователем. Обработано {len(results)}/{total} картинок.")
            break
        if progress_callback:
            progress_callback(f"Обработано {i}/{total} картинок...")
        path = item.get("image_path", "")
        if not path:
            results.append({"image_path": path, "response_attributes": {}, "error": "empty path"})
            continue
        img_path = Path(path)
        if not img_path.exists():
            results.append({"image_path": path, "response_attributes": {}, "error": "file not found"})
            continue
        try:
            # Local path is supported by _url_to_base64 in attribute_detector
            offer = {
                "offer_id": path,
                "name": "",
                "picture_urls": [str(img_path.resolve())],
                "category": "",
            }
            result = analyze_offer(offer, run_config)
            attrs = {}
            for _dir_id, dir_attrs in (result.get("direction_attributes") or {}).items():
                if isinstance(dir_attrs, dict):
                    for k, v in dir_attrs.items():
                        if k != "error" and isinstance(v, dict):
                            attrs[k] = v.get("value", "")
            results.append({"image_path": path, "response_attributes": attrs, "error": result.get("error")})
        except Exception as e:
            results.append({"image_path": path, "response_attributes": {}, "error": str(e)})
    if progress_callback and total and not (stop_event and stop_event.is_set()):
        progress_callback(f"Обработано {total}/{total} картинок.")
    return results


def main():
    ap = argparse.ArgumentParser(description="Eval before/after fine-tuning")
    ap.add_argument("--images", default="", help="JSONL with image_path or directory of images")
    ap.add_argument("--out", default="results", help="Output directory for eval_before.json, eval_after.json")
    ap.add_argument("--phase", choices=("before", "after", "compare"), required=True)
    ap.add_argument("--model", default="", help="Ollama model (default: from app_settings)")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "compare":
        before_path = out_dir / "eval_before.json"
        after_path = out_dir / "eval_after.json"
        if not before_path.exists() or not after_path.exists():
            print("Run --phase before and --phase after first.", file=sys.stderr)
            sys.exit(1)
        with open(before_path, encoding="utf-8") as f:
            before = json.load(f)
        with open(after_path, encoding="utf-8") as f:
            after = json.load(f)
        by_path = {r["image_path"]: r for r in before}
        matches = 0
        both_empty = 0
        total = 0
        for r in after:
            path = r["image_path"]
            if path not in by_path:
                continue
            total += 1
            b_attrs = by_path[path].get("response_attributes", {})
            a_attrs = r.get("response_attributes", {})
            if not b_attrs and not a_attrs:
                both_empty += 1
                print(f"Empty both: {path}")
            elif b_attrs == a_attrs:
                matches += 1
            else:
                print(f"Diff {path}: before {b_attrs} -> after {a_attrs}")
        print(f"Same (non-empty): {matches}/{total}, both empty: {both_empty}")
        report_path = out_dir / "eval_report.txt"
        report_path.write_text(f"Same: {matches}/{total}, both empty: {both_empty}\n", encoding="utf-8")
        print(f"Report: {report_path}")
        return

    if not args.images or not Path(args.images).exists():
        print("Provide --images (JSONL or directory)", file=sys.stderr)
        sys.exit(1)
    image_list = _collect_image_paths(args.images)
    if not image_list:
        print("No images found.", file=sys.stderr)
        sys.exit(1)
    import project_manager as pm
    config = pm.get_global_settings()
    if args.model:
        config["model"] = args.model
    config["ollama_url"] = args.ollama_url or config.get("ollama_url", "http://localhost:11434")
    results = _run_inference(image_list, config.get("model", "qwen3.5:35b"), config.get("ollama_url"), config)
    out_file = out_dir / ("eval_before.json" if args.phase == "before" else "eval_after.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {out_file}")


def run_eval_ui(
    images_path: str,
    out_dir: str,
    phase: str,
    config_override: dict | None = None,
    progress_callback=None,
    stop_event=None,
    max_examples: int = 0,
) -> dict:
    """
    Run eval phase from UI. Returns {success: bool, message: str, report_text: str | None}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if phase == "compare":
        before_path = out_dir / "eval_before.json"
        after_path = out_dir / "eval_after.json"
        if not before_path.exists() or not after_path.exists():
            return {"success": False, "message": "Сначала запустите «Оценка: до» и «Оценка: после».", "report_text": None}
        with open(before_path, encoding="utf-8") as f:
            before = json.load(f)
        with open(after_path, encoding="utf-8") as f:
            after = json.load(f)
        by_path = {r["image_path"]: r for r in before}
        matches = 0
        both_empty = 0
        total = 0
        lines = []
        for r in after:
            path = r["image_path"]
            if path not in by_path:
                continue
            total += 1
            b_attrs = by_path[path].get("response_attributes", {})
            a_attrs = r.get("response_attributes", {})
            if not b_attrs and not a_attrs:
                both_empty += 1
                lines.append(f"Пусто: {path}\n  до и после — оба без атрибутов (модель «после» могла не вернуть JSON).")
            elif b_attrs == a_attrs:
                matches += 1
            else:
                lines.append(f"Различие: {path}\n  до:   {b_attrs}\n  после: {a_attrs}")
        report_path = out_dir / "eval_report.txt"
        report_path.write_text(f"Совпадений (непустых): {matches}/{total}\nОба пусто: {both_empty}\n" + "\n".join(lines), encoding="utf-8")
        if total == 0:
            report_text = "Нет общих картинок для сравнения (запустите сравнение после того, как отработают «до» и «после»)."
        else:
            report_text = (
                f"По {matches} из {total} картинок ответы «до» и «после» совпали (одинаковые непустые атрибуты). "
                f"Чем больше — тем ближе поведение двух моделей.\n\n"
            )
            if both_empty > 0:
                report_text += f"Внимание: по {both_empty} картинкам обе модели вернули пустой ответ (например, модель «после» не вернула JSON) — это не считается совпадением.\n\n"
            if lines:
                report_text += "Отличия / пустые по картинкам:\n" + "\n".join(lines[:20])
                if len(lines) > 20:
                    report_text += f"\n... и ещё {len(lines) - 20}."
            else:
                report_text += "Отличий нет — по всем картинкам ответы совпали."
        return {"success": True, "message": f"Отчёт сохранён: {report_path}", "report_text": report_text}
    if not images_path or not Path(images_path).exists():
        return {"success": False, "message": "Укажите путь к JSONL или папке с картинками.", "report_text": None}
    image_list = _collect_image_paths(images_path)
    if not image_list:
        return {"success": False, "message": "В указанном пути нет картинок или записей JSONL.", "report_text": None}
    if max_examples > 0:
        image_list = image_list[:max_examples]
    import project_manager as pm
    config = {**(pm.get_global_settings()), **(config_override or {})}
    if progress_callback:
        progress_callback(f"Старт оценки: {len(image_list)} картинок, фаза «{phase}».")
    results = _run_inference(
        image_list,
        config.get("model", ""),
        config.get("ollama_url", "http://localhost:11434"),
        config,
        progress_callback=progress_callback,
        stop_event=stop_event,
    )
    stopped = stop_event and stop_event.is_set()
    if progress_callback and not stopped:
        progress_callback(f"Готово: {len(results)} результатов.")
    out_file = out_dir / ("eval_before.json" if phase == "before" else "eval_after.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    msg = f"Сохранено {len(results)} результатов в {out_file}"
    if stopped:
        msg = f"Остановлено пользователем. {msg} (обработано {len(results)} из {len(image_list)})."
    return {"success": True, "message": msg, "report_text": None}


if __name__ == "__main__":
    main()
