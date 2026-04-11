#!/usr/bin/env python3
"""
Convert DeepFashion Attribute Prediction annotations to our JSONL format.

Expects:
  - root/Anno/list_attr_cloth.txt  (first line = count, then one attribute name per line)
  - root/Anno/list_attr_img.txt    (first line = count, then "image_path v1 v2 ... v1000")
  - root/img/ or root/ with images

Output JSONL: one line per image, {"image_path": "...", "attributes": {"attr_name": "present", ...}}.
We only output attributes where value == 1 (present); value -1/0 are skipped or mapped to "no"/"unknown" if needed.

Usage:
  python scripts/convert_deepfashion_to_jsonl.py --root data/deepfashion --out data/deepfashion/train_attr.jsonl [--max-examples 500]
"""

import argparse
import json
from pathlib import Path


def parse_list_attr_cloth(path: Path) -> list[str]:
    """Parse list_attr_cloth.txt: first line is count, then lines with 'attribute_name  attribute_type' or just names."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if not lines:
        return []
    try:
        n = int(lines[0].strip())
        block = lines[1 : 1 + n]
    except ValueError:
        block = [ln for ln in lines if ln.strip()]
    # Each line can be "name  type" or just "name"; take first token, skip header
    names = []
    for ln in block:
        parts = ln.strip().split()
        if not parts or parts[0].lower() == "attribute_name":
            continue
        names.append(parts[0])
    return names


def parse_list_attr_img(path: Path, attr_names: list[str], root: Path, max_examples: int = 0) -> list[dict]:
    """Parse list_attr_img.txt; return list of {image_path, attributes}."""
    if not path.exists() or not attr_names:
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if not lines:
        return []
    try:
        n_imgs = int(lines[0].strip())
    except ValueError:
        n_imgs = len(lines) - 1
    # Line 1 can be header "image_name  attribute_labels"
    start = 1
    if len(lines) > 1 and lines[1].strip().lower().startswith("image_name"):
        start = 2
    out = []
    for line in lines[start : start + n_imgs]:
        if max_examples and len(out) >= max_examples:
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        img_name = parts[0]
        # Resolve image path: first column can be "img/xxx.jpg" or "xxx.jpg"
        if "/" in img_name or "\\" in img_name:
            img_path = root / img_name
        else:
            img_path = root / "img" / img_name
        if not img_path.exists():
            img_path = root / img_name
        vals = []
        for p in parts[1:]:
            try:
                vals.append(int(p))
            except ValueError:
                vals.append(0)
        attrs = {}
        for j, name in enumerate(attr_names):
            if j >= len(vals):
                break
            v = vals[j]
            if v == 1:
                attrs[name] = "yes"
            elif v == -1:
                attrs[name] = "no"
            # 0 = unknown: skip or add "unknown"
        if attrs:
            out.append({"image_path": str(img_path.resolve()), "attributes": attrs})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/deepfashion", help="DeepFashion root (with Anno/, img/)")
    ap.add_argument("--out", default="data/deepfashion/train_attr.jsonl", help="Output JSONL path")
    ap.add_argument("--max-examples", type=int, default=0, help="Max examples (0 = all)")
    args = ap.parse_args()
    root = Path(args.root)
    anno = root / "Anno"
    cloth_file = anno / "list_attr_cloth.txt"
    img_file = anno / "list_attr_img.txt"

    attr_names = parse_list_attr_cloth(cloth_file)
    if not attr_names:
        # Try alternative: list_attr_cloth might be single line with 1000 names
        if cloth_file.exists():
            text = cloth_file.read_text(encoding="utf-8", errors="replace").strip()
            attr_names = [s.strip() for s in text.replace("\n", " ").split() if s.strip()]
    print(f"Found {len(attr_names)} attribute names")

    examples = parse_list_attr_img(img_file, attr_names, root, max_examples=args.max_examples)
    print(f"Writing {len(examples)} examples to {args.out}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
