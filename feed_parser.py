#!/usr/bin/env python3
"""Parse YML feed and extract offers with picture URLs."""

import xml.etree.ElementTree as ET
from pathlib import Path


def _tag_local(elem):
    """Strip namespace from tag name."""
    if elem.tag and "}" in elem.tag:
        return elem.tag.split("}", 1)[1]
    return elem.tag or ""


def parse_feed(feed_path: str) -> list[dict]:
    """
    Extract from YML a list of offers with id, name, and picture URLs.

    Supports <picture>, <image>, and <picture_url> tags (with or without namespace).
    Returns list of dicts: {"offer_id": str, "name": str, "picture_urls": [str]}.
    """
    feed_path = Path(feed_path).resolve()
    if not feed_path.exists():
        raise FileNotFoundError(f"Feed file not found: {feed_path}")

    tree = ET.parse(feed_path)
    root = tree.getroot()

    offers = []
    for offer in root.findall(".//offer") or root.findall(".//{http://www.shopcreator.ru/shop/yml}offer"):
        offer_id = offer.get("id") or ""
        name_el = offer.find("name") or offer.find("{http://www.shopcreator.ru/shop/yml}name")
        name = (name_el.text or "").strip() if name_el is not None else ""

        picture_urls = []
        for tag in ("picture", "image", "picture_url"):
            for el in offer.findall(tag) or offer.findall(f"{{http://www.shopcreator.ru/shop/yml}}{tag}"):
                url = (el.text or "").strip()
                if url and url.startswith(("http://", "https://")):
                    picture_urls.append(url)

        if not picture_urls:
            continue
        offers.append({
            "offer_id": offer_id,
            "name": name,
            "picture_urls": picture_urls,
        })

    return offers
