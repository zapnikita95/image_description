#!/usr/bin/env python3
"""
Image analyzer (CLI): parse YML feed, send product images to Ollama vision model,
output attributes / text on product and descriptions to JSON/CSV.
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import requests

from feed_parser import parse_feed
from ollama_vision import describe_clothing_from_image

DEFAULT_MODEL = "qwen3.5:2b"
# Пул по умолчанию :11435 → Ollama :11434 (Desktop/ollama-queue-proxy).
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11435"


def download_image(url: str, timeout: int = 30) -> Path | None:
    """Download image to temp file; return path or None on failure."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        suf = Path(url.split("?")[0]).suffix or ".jpg"
        if suf.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            suf = ".jpg"
        f = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
        f.write(r.content)
        f.close()
        return Path(f.name)
    except Exception as e:
        print(f"  [WARN] Failed to download {url}: {e}", file=sys.stderr)
        return None


def main():
    p = argparse.ArgumentParser(
        description="Extract image attributes (text on clothing, adjectives) from YML feed using Ollama vision."
    )
    p.add_argument("--feed", "-f", type=str, help="Path to YML feed file")
    p.add_argument("--output", "-o", type=str, default="feed_image_attributes_results.json", help="Output JSON or CSV path")
    p.add_argument("--limit", "-n", type=int, default=0, help="Max offers to process (0 = all)")
    p.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL, help="Ollama API base URL")
    p.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Ollama model name (e.g. qwen3.5:2b)")
    p.add_argument("--format", choices=("json", "csv"), default="json", help="Output format")
    p.add_argument("--no-interactive", action="store_true", help="Do not prompt for feed/output if missing")
    args = p.parse_args()

    feed_path = args.feed
    output_path = args.output
    if not args.no_interactive and not feed_path:
        try:
            feed_path = input("Path to YML feed: ").strip()
        except EOFError:
            pass
    if not feed_path:
        print("Usage: python run.py --feed /path/to/feed.yml [--output results.json] [--limit 10]", file=sys.stderr)
        sys.exit(1)
    if not Path(feed_path).exists():
        print(f"Feed file not found: {feed_path}", file=sys.stderr)
        sys.exit(1)
    if not args.no_interactive and not output_path:
        output_path = input("Output file (default: feed_image_attributes_results.json): ").strip() or "feed_image_attributes_results.json"

    offers = parse_feed(feed_path)
    if args.limit > 0:
        offers = offers[: args.limit]
    print(f"Processing {len(offers)} offers (model={args.model})...")

    results = []
    for i, off in enumerate(offers):
        offer_id = off["offer_id"]
        name = off["name"]
        pic_url = off["picture_urls"][0]
        print(f"  [{i+1}/{len(offers)}] {offer_id} ... ", end="", flush=True)
        tmp = download_image(pic_url)
        if not tmp:
            results.append({
                "offer_id": offer_id,
                "name": name,
                "picture_url": pic_url,
                "text_on_clothing": "",
                "description_adjectives": "",
                "error": "download_failed",
            })
            print("skip (download failed)")
            continue
        try:
            text_on, desc = describe_clothing_from_image(
                image_path=tmp,
                model=args.model,
                ollama_url=args.ollama_url,
            )
            results.append({
                "offer_id": offer_id,
                "name": name,
                "picture_url": pic_url,
                "text_on_clothing": text_on,
                "description_adjectives": desc,
            })
            print("ok")
        except Exception as e:
            print(f"error: {e}")
            results.append({
                "offer_id": offer_id,
                "name": name,
                "picture_url": pic_url,
                "text_on_clothing": "",
                "description_adjectives": "",
                "error": str(e),
            })
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv" or str(out).lower().endswith(".csv"):
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["offer_id", "name", "picture_url", "text_on_clothing", "description_adjectives"])
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
    else:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Output: {out}")


if __name__ == "__main__":
    main()
