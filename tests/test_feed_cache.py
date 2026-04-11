"""Tests for feed_cache."""
import sqlite3
from pathlib import Path

import pytest

import feed_cache as fc


def test_parse_feed_to_cache(sample_feed_xml, temp_dir):
    db_path = temp_dir / "cache.db"
    summary = fc.parse_feed_to_cache(sample_feed_xml, db_path)
    assert summary["total"] == 2
    assert "categories" in summary
    assert db_path.exists()
    con = sqlite3.connect(str(db_path))
    rows = con.execute("SELECT offer_id, name FROM offers ORDER BY offer_id").fetchall()
    con.close()
    assert len(rows) == 2
    ids = [r[0] for r in rows]
    assert "101" in ids and "102" in ids


def test_get_categories(sample_feed_xml, temp_dir):
    db_path = temp_dir / "cache.db"
    fc.parse_feed_to_cache(sample_feed_xml, db_path)
    cats = fc.get_categories(db_path)
    assert isinstance(cats, dict)
    assert len(cats) >= 1


def test_get_cache_meta(sample_feed_xml, temp_dir):
    db_path = temp_dir / "cache.db"
    fc.parse_feed_to_cache(sample_feed_xml, db_path)
    meta = fc.get_cache_meta(db_path)
    assert int(meta.get("offer_count", 0)) == 2


def test_get_offers(sample_feed_xml, temp_dir):
    db_path = temp_dir / "cache.db"
    fc.parse_feed_to_cache(sample_feed_xml, db_path)
    offers = fc.get_offers(db_path, categories=None, limit=10)
    assert len(offers) == 2
    for o in offers:
        assert "offer_id" in o and "name" in o
        assert isinstance(o.get("picture_urls"), list)


def test_count_offers(sample_feed_xml, temp_dir):
    db_path = temp_dir / "cache.db"
    fc.parse_feed_to_cache(sample_feed_xml, db_path)
    assert fc.count_offers(db_path, None) == 2
    assert fc.count_offers(db_path, []) == 2


def test_is_cache_valid(sample_feed_xml, temp_dir):
    db_path = temp_dir / "cache.db"
    fc.parse_feed_to_cache(sample_feed_xml, db_path)
    assert fc.is_cache_valid(db_path, sample_feed_xml) is True


def test_get_offers_limit(sample_feed_xml, temp_dir):
    db_path = temp_dir / "cache.db"
    fc.parse_feed_to_cache(sample_feed_xml, db_path)
    offers = fc.get_offers(db_path, categories=None, limit=1)
    assert len(offers) == 1


def test_one_line_feed_utf8_slashslash_picture(temp_dir):
    """Как SUNLIGHT: одна строка, encoding=UTF8, картинка //cdn..."""
    xml = (
        "<?xml version='1.0' encoding='UTF8'?><yml_catalog><shop><categories>"
        '<category id="1">Jewelry</category></categories><offers>'
        '<offer id="a1"><name>Ring</name><categoryId>1</categoryId>'
        '<picture>//cdn.sunlight.net/x/y/photo.jpg</picture>'
        "</offer></offers></shop></yml_catalog>"
    )
    path = temp_dir / "sunlight_like.xml"
    path.write_text(xml, encoding="utf-8")
    db = temp_dir / "c1.db"
    summary = fc.parse_feed_to_cache(path, db)
    assert summary["total"] == 1


def test_param_named_picture(temp_dir):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<yml_catalog><shop><categories><category id="1">C</category></categories><offers>
<offer id="9"><name>N</name><categoryId>1</categoryId>
<param name="picture">https://example.com/z.webp</param>
</offer></offers></shop></yml_catalog>"""
    path = temp_dir / "param.xml"
    path.write_text(xml, encoding="utf-8")
    db = temp_dir / "c2.db"
    assert fc.parse_feed_to_cache(path, db)["total"] == 1


def test_parse_picture_url_in_attribute(temp_dir):
    """Yandex-style: picture URL only in attribute, default namespace."""
    path = temp_dir / "feed_ns.xml"
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<yml_catalog xmlns="http://market.yandex.ru/schemas/2008.09">
  <shop>
    <categories>
      <category id="1">Jewelry</category>
    </categories>
    <offers>
      <offer id="501">
        <name>Ring</name>
        <categoryId>1</categoryId>
        <picture url="https://example.com/ring.jpg"/>
      </offer>
    </offers>
  </shop>
</yml_catalog>""",
        encoding="utf-8",
    )
    db_path = temp_dir / "cache_ns.db"
    summary = fc.parse_feed_to_cache(path, db_path)
    assert summary["total"] == 1


def test_get_offers_with_categories(sample_feed_xml, temp_dir):
    db_path = temp_dir / "cache.db"
    fc.parse_feed_to_cache(sample_feed_xml, db_path)
    cats = fc.get_categories(db_path)
    cat_names = list(cats.keys())
    if cat_names:
        offers = fc.get_offers(db_path, categories=[cat_names[0]], limit=10)
        assert isinstance(offers, list)
    n = fc.count_offers(db_path, cat_names[:1] if cat_names else None)
    assert n >= 0
