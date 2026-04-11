#!/usr/bin/env python3
"""
Extract unique attribute keys from ShareGPT-style JSONL.

Expected shape per line:
{
  "conversations": [
     {"role":"system","value":...},
     {"role":"user","value":...},
     {"role":"assistant","value":"{\"key1\": {\"value\":...}, \"key2\": {...}}"}
  ]
}

Output:
 - JSON array with unique keys (strings), sorted.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _extract_assistant_json_value(obj: dict[str, Any]) -> str | None:
    conv = obj.get("conversations") or []
    if not isinstance(conv, list):
        return None
    for c in reversed(conv):
        if isinstance(c, dict) and c.get("role") == "assistant":
            v = c.get("value")
            if isinstance(v, str):
                return v
            # some datasets might store assistant answer as structured value already
            if isinstance(v, dict):
                # caller can handle dicts, but keep interface as string for simplicity
                return json.dumps(v, ensure_ascii=False)
    return None


def extract_keys_from_sharegpt_jsonl(jsonl_path: Path, limit_lines: int = 0) -> set[str]:
    keys: set[str] = set()
    if not jsonl_path.exists():
        raise FileNotFoundError(str(jsonl_path))
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit_lines and i >= limit_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            assistant_value = _extract_assistant_json_value(row)
            if not assistant_value:
                continue

            try:
                ans = json.loads(assistant_value)
            except json.JSONDecodeError:
                continue

            if isinstance(ans, dict):
                keys.update([str(k) for k in ans.keys()])
    return keys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", action="append", required=True, help="Path to JSONL. Can be passed multiple times.")
    p.add_argument("--out", default="generated/deepfashion/attribute_keys.json", help="Output JSON path.")
    p.add_argument("--limit-lines", type=int, default=0, help="Debug limit (0 = no limit).")
    args = p.parse_args()

    jsonl_paths = [Path(x) for x in args.jsonl]
    all_keys: set[str] = set()
    for jp in jsonl_paths:
        found = extract_keys_from_sharegpt_jsonl(jp, limit_lines=args.limit_lines)
        all_keys.update(found)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sorted(all_keys), f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(all_keys)} unique attribute keys -> {out_path}")


if __name__ == "__main__":
    main()

