#!/usr/bin/env python3
"""
Build direction chunks for DeepFashion attribute keys.

Input:
 - keys JSON (array of en keys) from extract_attribute_keys_from_sharegpt_jsonl.py
 - labels cache JSON mapping en_key -> ru_label

Output:
 - deepfashion_directions.json (list of directions) usable in projects/<name>/config.json
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Build deepfashion direction objects for app config.")
    p.add_argument("--keys", required=True, help="JSON array file with EN attribute keys.")
    p.add_argument("--labels", required=True, help="JSON mapping en_key -> ru_label.")
    p.add_argument("--out", default="generated/deepfashion/deepfashion_directions.json", help="Output json file.")
    p.add_argument("--keys-per-direction", type=int, default=80, help="How many attribute keys per direction chunk.")
    p.add_argument("--direction-prefix", default="deepfashion_chunk", help="Direction id prefix.")
    args = p.parse_args()

    keys_path = Path(args.keys)
    labels_path = Path(args.labels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys_obj = _load_json(keys_path)
    labels_obj = _load_json(labels_path)
    if not isinstance(keys_obj, list):
        raise ValueError("--keys must point to JSON array file.")
    if not isinstance(labels_obj, dict):
        raise ValueError("--labels must be a JSON object mapping.")

    keys = [str(x) for x in keys_obj]
    labels = {str(k): str(v) for k, v in labels_obj.items()}

    # Sort keys for deterministic chunking.
    keys = sorted(set(keys))
    n = max(1, int(args.keys_per_direction))

    directions: list[dict[str, Any]] = []
    for i in range(0, len(keys), n):
        chunk = keys[i : i + n]
        chunk_index = i // n
        dir_id = f"{args.direction_prefix}_{chunk_index:03d}"
        attributes = []
        for k in chunk:
            label = labels.get(k) or labels.get(k.strip().lower()) or k
            attributes.append(
                {
                    "key": k,
                    "label": label,
                    # In DeepFashion JSONL the values are yes/no. Keep the same options for the detector prompt.
                    "options": ["yes", "no", "unknown"],
                }
            )
        directions.append(
            {
                "id": dir_id,
                "name": f"DeepFashion атрибуты {chunk_index + 1}",
                "text_enabled": False,
                "attributes": attributes,
                "custom_prompt": "",
            }
        )

    out_path.write_text(json.dumps(directions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(directions)} direction chunks -> {out_path}")


if __name__ == "__main__":
    main()

