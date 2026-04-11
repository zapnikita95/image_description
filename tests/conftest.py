"""Pytest fixtures."""
import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def projects_dir(temp_dir):
    (temp_dir / "projects").mkdir()
    return temp_dir / "projects"


@pytest.fixture
def app_settings_path(temp_dir):
    return temp_dir / "app_settings.json"


@pytest.fixture
def sample_feed_xml(temp_dir):
    path = temp_dir / "feed.xml"
    path.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<yml_catalog>
  <shop>
    <categories>
      <category id="1">Blouses</category>
      <category id="2" parentId="1">Tops</category>
    </categories>
    <offers>
      <offer id="101">
        <name>Top viscose</name>
        <categoryId>2</categoryId>
        <url>https://example.com/101</url>
        <picture>https://example.com/pic1.jpg</picture>
      </offer>
      <offer id="102">
        <name>Blouse</name>
        <categoryId>1</categoryId>
        <url>https://example.com/102</url>
        <picture>https://example.com/pic2.jpg</picture>
      </offer>
      <offer id="101">
        <name>Duplicate id</name>
        <categoryId>1</categoryId>
        <url>https://example.com/dup</url>
        <picture>https://example.com/pic3.jpg</picture>
      </offer>
    </offers>
  </shop>
</yml_catalog>""", encoding="utf-8")
    return path
