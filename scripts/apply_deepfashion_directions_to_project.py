#!/usr/bin/env python3
"""
Apply generated deepfashion directions to projects/<project>/config.json.

- Appends directions that have new ids.
- Skips direction ids that already exist.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Apply deepfashion direction chunks to project config.json.")
    p.add_argument("--project", default="Befree", help="Project name (projects/<name>/config.json).")
    p.add_argument("--directions", required=True, help="Path to deepfashion_directions.json (list of directions).")
    p.add_argument("--repo-root", default=None, help="Repo root path. Auto-detect if not provided.")
    p.add_argument("--update-existing", action="store_true", help="Update labels for existing direction ids (by id).")
    p.add_argument("--dry-run", action="store_true", help="Don't write changes.")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = Path(args.repo_root).resolve() if args.repo_root else script_dir.parent
    cfg_path = repo_root / "projects" / args.project / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))

    directions_path = Path(args.directions)
    if not directions_path.exists():
        raise FileNotFoundError(str(directions_path))

    cfg = _load_json(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("config.json must be a JSON object.")
    dirs = _load_json(directions_path)
    if not isinstance(dirs, list):
        raise ValueError("--directions must be a JSON array.")

    cfg_dirs = cfg.get("directions") or []
    if not isinstance(cfg_dirs, list):
        cfg_dirs = []

    existing_by_id = {}
    for d in cfg_dirs:
        if isinstance(d, dict) and d.get("id"):
            existing_by_id[str(d["id"])] = d

    to_add = []
    to_update = []
    for d in dirs:
        if not isinstance(d, dict) or not d.get("id"):
            continue
        did = str(d["id"])
        if did in existing_by_id:
            if args.update_existing:
                to_update.append(d)
        else:
            to_add.append(d)

    print(f"Project: {args.project}")
    print(f"Current directions: {len(cfg_dirs)} | New to add: {len(to_add)} | To update: {len(to_update)}")

    if args.dry_run:
        print("Dry-run: no file write.")
        return

    # Update existing directions labels when requested.
    if args.update_existing and to_update:
        for nd in to_update:
            did = str(nd.get("id"))
            ed = existing_by_id.get(did)
            if not ed:
                continue
            # Update name/custom_prompt/text_enabled if provided.
            for field in ("name", "custom_prompt", "text_enabled"):
                if field in nd:
                    ed[field] = nd[field]
            # Build key->attr for existing and new.
            existing_attrs = ed.get("attributes") or []
            if not isinstance(existing_attrs, list):
                continue
            existing_by_key = {}
            for a in existing_attrs:
                if isinstance(a, dict) and a.get("key"):
                    existing_by_key[str(a["key"])] = a
            new_attrs = nd.get("attributes") or []
            if not isinstance(new_attrs, list):
                continue
            for na in new_attrs:
                if not isinstance(na, dict) or not na.get("key"):
                    continue
                k = str(na["key"])
                if k in existing_by_key:
                    # Only update label/options if they exist in generated data.
                    if "label" in na:
                        existing_by_key[k]["label"] = na["label"]
                    if "options" in na:
                        existing_by_key[k]["options"] = na["options"]
                else:
                    # Key not present: append it to attributes.
                    existing_attrs.append(na)
            ed["attributes"] = existing_attrs

    cfg_dirs = cfg_dirs + to_add
    cfg["directions"] = cfg_dirs
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated: {cfg_path}")


if __name__ == "__main__":
    main()

