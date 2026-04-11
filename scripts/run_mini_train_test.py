#!/usr/bin/env python3
"""
Быстрый тест дообучения (mini train): 2 шага (по умолчанию) на выбранной модели.
Запуск из корня проекта:
  python scripts/run_mini_train_test.py --model Qwen/Qwen3.5-4B-Instruct
Проверяет пайплайн: загрузка VL-модели + LoRA + небольшой train.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Минимальная картинка 1x1 red PNG (base64)
TINY_IMG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B-Instruct", help="Base model HF id")
    p.add_argument("--steps", type=int, default=2, help="Max steps")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    p.add_argument("--max-train-examples", type=int, default=2, help="How many examples from mini dataset")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    run_tag = f"{int(time.time())}"
    out_dir = root / f"test_lora_mini_{run_tag}"
    dataset_path = root / f"test_mini_dataset_{run_tag}.jsonl"
    # 2 VL-примера в формате датасета приложения
    examples = [
        {
            "conversations": [
                {"role": "user", "value": [{"type": "image", "image": TINY_IMG_B64}, {"type": "text", "text": "What is this?"}]},
                {"role": "assistant", "value": "A test image."},
            ]
        },
        {
            "conversations": [
                {"role": "user", "value": [{"type": "image", "image": TINY_IMG_B64}, {"type": "text", "text": "Describe."}]},
                {"role": "assistant", "value": "Red pixel."},
            ]
        },
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("Dataset:", dataset_path, "| 2 examples")
    from fine_tune.train import train
    result = train(
        dataset_path=dataset_path,
        base_model=args.model,
        output_dir=out_dir,
        max_steps=int(args.steps),
        batch_size=int(args.batch_size),
        lora_rank=int(args.lora_rank),
        progress_callback=lambda s, t, l: print(f"  Step {s}/{t}  loss={l:.4f}"),
        max_train_examples=int(args.max_train_examples),
    )
    if result.get("success"):
        print("OK. Adapter:", result.get("output_dir"))
    else:
        print("FAIL:", result.get("error"))
        sys.exit(1)
    # Удаляем тестовые артефакты
    try:
        import shutil
        if dataset_path.exists():
            dataset_path.unlink()
        if out_dir.exists():
            shutil.rmtree(out_dir)
    except Exception:
        pass

if __name__ == "__main__":
    main()
