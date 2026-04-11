#!/usr/bin/env python3
"""
Собрать минимальный датасет с картинками для проверки VL-обучения.

Читает внешний JSONL (например data/deepfashion/train.jsonl), берёт первые
N примеров, подставляет картинки по image_path в ShareGPT-формат с base64,
пишет в один JSONL. Этим файлом можно запускать обучение и убедиться, что
в логе есть "VL (images): True" и "Starting VL training (with images)".

Пример:
  python scripts/build_mini_vl_dataset.py --input data/deepfashion/train.jsonl --output fine_tune/dataset/train_mini_vl.jsonl --max 5
  python fine_tune/train.py --dataset fine_tune/dataset/train_mini_vl.jsonl --max-train-examples 5 --max-steps 2
"""

import argparse
import base64
import json
import sys
from pathlib import Path

# чтобы вызывать _external_row_to_sharegpt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fine_tune.dataset_builder import _external_row_to_sharegpt  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Build minimal VL dataset (ShareGPT + base64 images)")
    p.add_argument("--input", default="data/deepfashion/train.jsonl", help="Input JSONL (image_path + attributes)")
    p.add_argument("--output", default="fine_tune/dataset/train_mini_vl.jsonl", help="Output JSONL path")
    p.add_argument("--max", type=int, default=5, help="Max number of examples")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        print(f"Error: input file not found: {inp}", file=sys.stderr)
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    images_dir = inp.parent

    examples = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            if len(examples) >= args.max:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ex = _external_row_to_sharegpt(row, images_dir, include_image=True)
            if ex and any(
                isinstance(c.get("value"), list)
                and any(v.get("type") == "image" for v in c.get("value", []) if isinstance(v, dict))
                for c in ex.get("conversations", [])
            ):
                examples.append(ex)

    if not examples:
        print("Error: no examples with images found. Check --input and that image_path exists.", file=sys.stderr)
        sys.exit(1)

    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} VL examples to {out}")
    print("Run training with: python fine_tune/train.py --dataset", str(out), "--max-train-examples", len(examples), "--max-steps 2")


if __name__ == "__main__":
    main()
