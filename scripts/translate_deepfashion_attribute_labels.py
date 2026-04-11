#!/usr/bin/env python3
"""
Translate EN attribute keys to RU short labels (for config.json "label").

Seeds translations from existing attribute_glossary.json (EN->RU) when present.
For missing keys, uses Ollama /api/chat and strict JSON responses.

Outputs a cache JSON mapping: { "<en_key>": "<ru_label>" }.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable when running as `python scripts/xxx.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """
    Extract first balanced {...} and parse as JSON.
    Returns {} on failure.
    """
    s = (text or "").strip()
    if not s:
        return {}
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = s[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return {}
                return {}
    return {}


def _safe_ollama_chat(prompt: str, model: str, ollama_url: str, timeout: int) -> str:
    # Reuse the existing Ollama wrapper from attribute_detector to keep url normalization consistent.
    from attribute_detector import _ollama_chat  # type: ignore

    return _ollama_chat(
        prompt=prompt,
        image_b64=None,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        system=None,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Translate EN attribute keys to RU labels via Ollama (with cache).")
    p.add_argument("--keys", required=True, help="Path to JSON file with keys (array or {key:...}).")
    p.add_argument("--out-cache", required=True, help="Path to output cache JSON mapping en_key -> ru_label.")
    p.add_argument("--glossary", default="attribute_glossary.json", help="Path to attribute_glossary.json (EN->RU).")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base url.")
    p.add_argument("--translate-model", default="qwen3.5:0.5b", help="Ollama model to use for translation.")
    p.add_argument("--batch-size", type=int, default=20, help="How many keys per Ollama call.")
    p.add_argument("--timeout", type=int, default=120, help="Ollama request timeout seconds.")
    p.add_argument("--max-translate", type=int, default=0, help="For debug: translate only first N missing keys (0=all).")
    p.add_argument("--dry-run", action="store_true", help="Do not call Ollama; only seed from glossary.")
    args = p.parse_args()

    keys_path = Path(args.keys)
    out_cache_path = Path(args.out_cache)
    out_cache_path.parent.mkdir(parents=True, exist_ok=True)

    keys_obj = _load_json(keys_path)
    if keys_obj is None:
        raise FileNotFoundError(str(keys_path))

    if isinstance(keys_obj, list):
        keys = [str(x) for x in keys_obj]
    elif isinstance(keys_obj, dict):
        keys = [str(k) for k in keys_obj.keys()]
    else:
        raise ValueError("Unsupported keys JSON format (expected list or object).")

    glossary_path = Path(args.glossary)
    glossary_obj = _load_json(glossary_path) or {}
    glossary = {str(k).strip().lower(): str(v).strip() for k, v in (glossary_obj or {}).items()}

    cache_obj = _load_json(out_cache_path) or {}
    if not isinstance(cache_obj, dict):
        cache_obj = {}
    raw_cache: dict[str, str] = {str(k): str(v) for k, v in cache_obj.items()}

    # Heuristic: if label equals the key itself (common fallback), treat it as "untranslated"
    # so we can resume translation on the same file.
    cache: dict[str, str] = {}
    for k, v in raw_cache.items():
        v_clean = (v or "").strip()
        if not v_clean:
            continue
        if v_clean.lower() == str(k).strip().lower():
            continue
        cache[k] = v_clean

    # Seed from glossary first (these terms already have RU translations).
    for k in keys:
        kl = k.strip().lower()
        if kl in glossary and k not in cache:
            cache[k] = glossary[kl]

    missing = [k for k in sorted(set(keys)) if k not in cache]
    if args.max_translate and args.max_translate > 0:
        missing = missing[: args.max_translate]

    print(f"Keys: {len(set(keys))} | Cached/seeded: {len(cache)} | Missing: {len(missing)}")
    if not missing:
        out_cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"All labels are available -> {out_cache_path}")
        return

    if args.dry_run:
        out_cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Dry-run: skipped Ollama translation.")
        return

    # Translate in batches (retry + dynamic batch shrinking on timeouts).
    # We also save progress after each successful batch so you can resume safely.
    pending = list(missing)
    total_keys = len(set(keys))
    batch_size = max(1, int(args.batch_size))
    success_batches = 0

    while pending:
        batch = pending[:batch_size]
        pending = pending[batch_size:]

        bullet_keys = "\n".join([f"- {k}" for k in batch])
        prompt = (
            "Translate English fashion attribute keys into Russian short labels for UI.\n"
            "Requirements:\n"
            "1) Return STRICT JSON only, no markdown.\n"
            "2) JSON keys MUST be EXACTLY the same as in input (including spaces/hyphens).\n"
            "3) Labels should be short Russian phrases (usually 1-3 words), nominative.\n"
            "4) Do NOT add extra explanations.\n\n"
            f"Input keys:\n{bullet_keys}\n"
        )

        try:
            resp = _safe_ollama_chat(
                prompt,
                model=args.translate_model,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
            )
            obj = _extract_first_json_object(resp)
            if not obj:
                raise RuntimeError(f"Failed to parse JSON. Response preview: {resp[:300]!r}")

            translated_now = 0
            for k in batch:
                if k in obj and isinstance(obj[k], str):
                    label = obj[k].strip()
                    if label:
                        cache[k] = label
                        translated_now += 1

            # Save progress (translated-only mapping).
            # This keeps the file resumable and allows monitoring.
            out_progress = {}
            # Uniqueness is handled later, but we can save raw translations now.
            for k, v in cache.items():
                if isinstance(v, str) and v.strip():
                    out_progress[k] = v.strip()
            out_cache_path.write_text(json.dumps(out_progress, ensure_ascii=False, indent=2), encoding="utf-8")

            success_batches += 1
            print(
                f"Translated batch={len(batch)} (ok_keys={translated_now}) | cache now {len(cache)} / {total_keys} | saved progress (batch {success_batches})",
                flush=True,
            )
        except Exception as e:
            # On timeout/network errors, retry with smaller batches.
            print(f"Batch failed (batch_size={batch_size}) error={str(e)[:120]}...", flush=True)
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            # Put failed batch back to pending to retry with smaller size.
            pending = batch + pending
            print(f"Reduced batch_size -> {batch_size}. Retrying...", flush=True)

    # Enforce uniqueness for labels (normalize_correction_attrs uses label.lower() as key).
    # Save only translated labels (missing keys stay absent from cache for resumability).
    used: dict[str, str] = {}  # label_lower -> en_key (first owner)
    translated_keys_sorted = sorted(cache.keys())
    final_cache: dict[str, str] = {}
    for k in translated_keys_sorted:
        label = (cache.get(k) or "").strip()
        if not label:
            continue
        lk = label.lower()
        if lk in used:
            # Disambiguate by adding en key in parentheses.
            label2 = f"{label} ({k})"
            lk2 = label2.lower()
            if lk2 in used:
                label2 = f"{label} ({k}) #{len(used)+1}"
            label = label2
            lk = label.lower()
        used[lk] = k
        final_cache[k] = label

    out_cache_path.write_text(json.dumps(final_cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved RU labels cache (translated only) -> {out_cache_path} (entries={len(final_cache)})")


if __name__ == "__main__":
    main()

