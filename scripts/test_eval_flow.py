#!/usr/bin/env python3
"""
Проверка пайплайна «Оценка до/после» без реального Ollama.
Создаёт временный JSONL с одним путём (файла нет), запускает run_eval_ui(phase=before),
проверяет что скрипт не падает и в результате есть запись с ошибкой "file not found".
Запуск: python scripts/test_eval_flow.py
"""
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_before_after import run_eval_ui


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jsonl = tmp / "test.jsonl"
        jsonl.write_text(json.dumps({"image_path": str(tmp / "nonexistent.jpg"), "attributes": {}}) + "\n", encoding="utf-8")
        out = tmp / "out"
        out.mkdir()

        r = run_eval_ui(
            str(jsonl),
            str(out),
            "before",
            progress_callback=lambda m: print(m),
            max_examples=1,
        )

        if not r["success"]:
            print("FAIL: expected success=True (one result with error). Got:", r)
            sys.exit(1)

        before_file = out / "eval_before.json"
        if not before_file.exists():
            print("FAIL: eval_before.json not created")
            sys.exit(1)

        import json as j
        data = j.loads(before_file.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) != 1:
            print("FAIL: expected 1 result, got", len(data) if isinstance(data, list) else type(data))
            sys.exit(1)

        if "error" not in data[0] and "file not found" not in str(data[0].get("error", "")).lower():
            print("OK (no image file):", data[0].get("error", "no error key"))
        print("OK: eval before flow runs, eval_before.json written.")

        # Фаза «после» — те же пути
        r2 = run_eval_ui(str(jsonl), str(out), "after", max_examples=1)
        if not r2["success"]:
            print("FAIL: phase after:", r2)
            sys.exit(1)
        print("OK: eval after flow runs.")

        # Сравнить
        r3 = run_eval_ui("", str(out), "compare")
        if not r3["success"]:
            print("FAIL: compare:", r3)
            sys.exit(1)
        if "report_text" not in r3 or "картинок" not in r3.get("report_text", ""):
            print("FAIL: compare report unexpected:", r3.get("report_text", "")[:200])
            sys.exit(1)
        print("OK: compare runs.", r3["report_text"].strip().split("\n")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
