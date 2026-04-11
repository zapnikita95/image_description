"""
Tests for multi-image / picture-attr-filter functionality.

Covers:
  - feed_cache: tagged URL tracking, get_feed_image_attr_names, picture_attr_filter
  - attribute_detector: _select_best_image_b64 logic
  - project_manager: new config keys preserved on save/load
"""
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import feed_cache as fc
import project_manager as pm


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def feed_with_param_pictures(temp_dir):
    """Feed where pictures are in <param name='Picture'> tags (common in Yandex feeds)."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<yml_catalog>
  <shop>
    <categories>
      <category id="1">Clothing</category>
    </categories>
    <offers>
      <offer id="201">
        <name>Shirt red</name>
        <categoryId>1</categoryId>
        <param name="Picture">https://cdn.example.com/shirt_front.jpg</param>
        <param name="Picture">https://cdn.example.com/shirt_back.jpg</param>
        <param name="Photo">https://cdn.example.com/shirt_detail.jpg</param>
      </offer>
      <offer id="202">
        <name>Pants blue</name>
        <categoryId>1</categoryId>
        <picture>https://cdn.example.com/pants1.jpg</picture>
        <picture>https://cdn.example.com/pants2.jpg</picture>
      </offer>
    </offers>
  </shop>
</yml_catalog>"""
    path = temp_dir / "param_feed.xml"
    path.write_text(xml, encoding="utf-8")
    return path


@pytest.fixture
def parsed_db(feed_with_param_pictures, temp_dir):
    db = temp_dir / "cache.db"
    fc.parse_feed_to_cache(feed_with_param_pictures, db)
    return db


# ── feed_cache: tagged URL tracking ──────────────────────────────────────────

class TestTaggedPictureUrls:
    def test_collect_tagged_urls_param(self):
        """_collect_tagged_picture_urls groups param images under 'param:picture'."""
        import xml.etree.ElementTree as ET
        xml = """<offer id="1">
            <param name="Picture">https://a.com/1.jpg</param>
            <param name="Picture">https://a.com/2.jpg</param>
            <param name="Photo">https://a.com/3.jpg</param>
        </offer>"""
        offer = ET.fromstring(xml)
        tagged = fc._collect_tagged_picture_urls(offer)
        assert "param:picture" in tagged
        assert len(tagged["param:picture"]) == 2
        assert "param:photo" in tagged
        assert len(tagged["param:photo"]) == 1

    def test_collect_tagged_urls_picture_tag(self):
        """Standard <picture> tag goes under key 'picture'."""
        import xml.etree.ElementTree as ET
        xml = """<offer id="2">
            <picture>https://b.com/img1.jpg</picture>
            <picture>https://b.com/img2.jpg</picture>
        </offer>"""
        offer = ET.fromstring(xml)
        tagged = fc._collect_tagged_picture_urls(offer)
        assert "picture" in tagged
        assert len(tagged["picture"]) == 2

    def test_collect_tagged_urls_dedup(self):
        """Duplicate URLs within a tag are deduplicated."""
        import xml.etree.ElementTree as ET
        xml = """<offer id="3">
            <picture>https://c.com/same.jpg</picture>
            <picture>https://c.com/same.jpg</picture>
        </offer>"""
        offer = ET.fromstring(xml)
        tagged = fc._collect_tagged_picture_urls(offer)
        assert len(tagged.get("picture", [])) == 1

    def test_flat_picture_urls_still_work(self):
        """get_offers still returns flat picture_urls list."""
        import xml.etree.ElementTree as ET
        xml = """<offer id="4">
            <picture>https://d.com/p1.jpg</picture>
            <picture>https://d.com/p2.jpg</picture>
        </offer>"""
        offer = ET.fromstring(xml)
        urls = fc._collect_picture_urls(offer)
        assert len(urls) == 2
        assert all(u.startswith("https://") for u in urls)


class TestGetFeedImageAttrNames:
    def test_empty_db(self, temp_dir):
        """Returns [] for non-existent DB."""
        names = fc.get_feed_image_attr_names(temp_dir / "nonexistent.db")
        assert names == []

    def test_param_picture_feed(self, parsed_db):
        """After parsing param-picture feed, attr names include 'param:picture'."""
        names = fc.get_feed_image_attr_names(parsed_db)
        assert isinstance(names, list)
        assert "param:picture" in names
        assert "param:photo" in names

    def test_picture_tag_feed(self, temp_dir):
        """After parsing standard picture-tag feed, attr names include 'picture'."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<yml_catalog><shop>
  <categories><category id="1">Cat</category></categories>
  <offers>
    <offer id="10"><name>X</name><categoryId>1</categoryId>
      <picture>https://x.com/a.jpg</picture>
    </offer>
  </offers>
</shop></yml_catalog>"""
        path = temp_dir / "std.xml"
        path.write_text(xml, encoding="utf-8")
        db = temp_dir / "std.db"
        fc.parse_feed_to_cache(path, db)
        names = fc.get_feed_image_attr_names(db)
        assert "picture" in names

    def test_attr_names_sorted(self, parsed_db):
        """Returned list is sorted."""
        names = fc.get_feed_image_attr_names(parsed_db)
        assert names == sorted(names)


class TestPictureAttrFilter:
    def test_filter_param_picture(self, parsed_db):
        """With filter=['param:picture'], offer 201 only gets param:picture URLs."""
        offers = fc.get_offers(parsed_db, picture_attr_filter=["param:picture"])
        offer_201 = next(o for o in offers if o["offer_id"] == "201")
        # Should only contain the 2 param:picture URLs, not param:photo
        assert len(offer_201["picture_urls"]) == 2
        assert all("shirt_front" in u or "shirt_back" in u for u in offer_201["picture_urls"])

    def test_filter_empty_means_all(self, parsed_db):
        """Empty filter returns all picture URLs."""
        offers_all = fc.get_offers(parsed_db, picture_attr_filter=None)
        offers_filt = fc.get_offers(parsed_db, picture_attr_filter=[])
        # Both should return same counts (no filtering)
        for o_all, o_filt in zip(
            sorted(offers_all, key=lambda x: x["offer_id"]),
            sorted(offers_filt, key=lambda x: x["offer_id"]),
        ):
            assert o_all["offer_id"] == o_filt["offer_id"]
            assert o_all["picture_urls"] == o_filt["picture_urls"]

    def test_filter_nonexistent_tag_keeps_original(self, parsed_db):
        """Filter with tag that doesn't exist → picture_urls unchanged (fallback to all)."""
        offers_all = fc.get_offers(parsed_db, picture_attr_filter=None)
        offers_filt = fc.get_offers(parsed_db, picture_attr_filter=["nonexistent_tag_xyz"])
        # When no matching URLs found, original URLs preserved
        for o_all, o_filt in zip(
            sorted(offers_all, key=lambda x: x["offer_id"]),
            sorted(offers_filt, key=lambda x: x["offer_id"]),
        ):
            assert o_all["picture_urls"] == o_filt["picture_urls"]

    def test_picture_tagged_urls_preserved(self, parsed_db):
        """picture_tagged_urls field is always available as dict."""
        offers = fc.get_offers(parsed_db)
        for o in offers:
            assert isinstance(o.get("picture_tagged_urls"), dict)

    def test_get_offer_by_id_with_filter(self, parsed_db):
        """get_offer_by_id respects picture_attr_filter."""
        o = fc.get_offer_by_id(parsed_db, "201", picture_attr_filter=["param:picture"])
        assert o is not None
        assert len(o["picture_urls"]) == 2

    def test_meta_picture_attr_names_stored(self, parsed_db):
        """picture_attr_names is stored in meta table after parsing."""
        con = sqlite3.connect(str(parsed_db))
        row = con.execute("SELECT value FROM meta WHERE key='picture_attr_names'").fetchone()
        con.close()
        assert row is not None
        names = json.loads(row[0])
        assert "param:picture" in names


# ── project_manager: new config keys ─────────────────────────────────────────

class TestProjectConfigNewKeys:
    def test_defaults_present(self):
        """DEFAULT_CONFIG contains picture_attr_filter and multi_image_mode."""
        assert "picture_attr_filter" in pm.DEFAULT_CONFIG
        assert "multi_image_mode" in pm.DEFAULT_CONFIG
        assert pm.DEFAULT_CONFIG["picture_attr_filter"] == []
        assert pm.DEFAULT_CONFIG["multi_image_mode"] == "first_only"

    def test_project_keys_include_new_fields(self):
        """PROJECT_KEYS includes the new config keys."""
        assert "picture_attr_filter" in pm.PROJECT_KEYS
        assert "multi_image_mode" in pm.PROJECT_KEYS

    def test_save_and_load_new_keys(self, tmp_path, monkeypatch):
        """New keys survive save/load round-trip."""
        monkeypatch.setattr(pm, "PROJECTS_DIR", tmp_path / "projects")
        (tmp_path / "projects").mkdir(parents=True, exist_ok=True)

        cfg = pm.create_project("testproj", vertical="Одежда")
        cfg["picture_attr_filter"] = ["param:picture", "photo"]
        cfg["multi_image_mode"] = "best_select"
        pm.save_project(cfg)

        loaded = pm.load_project("testproj")
        assert loaded["picture_attr_filter"] == ["param:picture", "photo"]
        assert loaded["multi_image_mode"] == "best_select"

    def test_multi_image_mode_values(self):
        """multi_image_mode accepts only valid values conceptually."""
        valid = {"first_only", "best_select", "all_images"}
        assert pm.DEFAULT_CONFIG["multi_image_mode"] in valid


# ── attribute_detector: _select_best_image_b64 ───────────────────────────────

class TestSelectBestImageB64:
    def test_single_image_returns_itself(self):
        """With one image, no model call needed — returns that image."""
        from attribute_detector import _select_best_image_b64
        result = _select_best_image_b64(
            ["base64data"],
            product_name=None,
            task_hint=None,
            model="qwen3.5:4b",
            ollama_url="http://127.0.0.1:11434",
        )
        assert result == "base64data"

    def test_empty_returns_none(self):
        """Empty input returns None."""
        from attribute_detector import _select_best_image_b64
        result = _select_best_image_b64(
            [],
            product_name=None,
            task_hint=None,
            model="qwen3.5:4b",
            ollama_url="http://127.0.0.1:11434",
        )
        assert result is None

    def test_valid_model_response(self):
        """Model returns '2' → second image is selected."""
        from attribute_detector import _select_best_image_b64

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "2"}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = _select_best_image_b64(
                ["img_a", "img_b", "img_c"],
                product_name="Test shirt",
                task_hint="determine color",
                model="qwen3.5:4b",
                ollama_url="http://127.0.0.1:11434",
            )
        assert result == "img_b"

    def test_out_of_range_falls_back_to_first(self):
        """Model returns out-of-range index → fall back to first image."""
        from attribute_detector import _select_best_image_b64

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "99"}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = _select_best_image_b64(
                ["img_a", "img_b"],
                product_name=None,
                task_hint=None,
                model="qwen3.5:4b",
                ollama_url="http://127.0.0.1:11434",
            )
        assert result == "img_a"

    def test_model_error_falls_back_to_first(self):
        """Network error → first image returned."""
        from attribute_detector import _select_best_image_b64
        import requests as req

        with patch("requests.post", side_effect=req.RequestException("timeout")):
            result = _select_best_image_b64(
                ["img_a", "img_b"],
                product_name=None,
                task_hint=None,
                model="qwen3.5:4b",
                ollama_url="http://127.0.0.1:11434",
            )
        assert result == "img_a"

    def test_model_returns_noisy_text(self):
        """Model returns text with digit embedded — still extracts correctly."""
        from attribute_detector import _select_best_image_b64

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "The best image is number 3."}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = _select_best_image_b64(
                ["a", "b", "c", "d"],
                product_name=None,
                task_hint=None,
                model="qwen3.5:4b",
                ollama_url="http://127.0.0.1:11434",
            )
        assert result == "c"


# ── feed_cache: backward compat — old DB without picture_tagged_urls ──────────

class TestBackwardCompat:
    def test_old_db_without_tagged_column(self, temp_dir):
        """DB created without picture_tagged_urls column is handled gracefully."""
        # Create minimal DB without the new column
        db = temp_dir / "old.db"
        con = sqlite3.connect(str(db))
        con.execute("""CREATE TABLE offers (
            offer_id TEXT PRIMARY KEY,
            name TEXT, category_id TEXT, category TEXT,
            picture_urls TEXT, vendor TEXT, url TEXT
        )""")
        con.execute("""CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)""")
        con.execute("INSERT INTO offers VALUES ('x1','Item','1','Cat','[\"https://a.com/img.jpg\"]','','https://a.com')")
        con.execute("INSERT INTO meta VALUES ('offer_count','1')")
        con.commit()
        con.close()

        # Should not raise despite missing column — graceful fallback
        offers = fc.get_offers(db)
        assert len(offers) == 1
        assert offers[0]["picture_urls"] == ["https://a.com/img.jpg"]
        # picture_tagged_urls should be empty dict or {}
        assert isinstance(offers[0].get("picture_tagged_urls", {}), dict)
