"""Тесты лёгкого dHash-дедупа (picture_dedupe)."""

import threading
from pathlib import Path

from PIL import Image

import picture_dedupe as pd


def test_dhash_same_content_same_hash(tmp_path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.new("RGB", (40, 40), color=(200, 100, 50)).save(p1)
    Image.new("RGB", (40, 40), color=(200, 100, 50)).save(p2)
    m: dict[str, str] = {}
    assert pd.dhash64_file(p1, m) == pd.dhash64_file(p2, m)
    assert len(m) == 2


def test_dhash_memo(tmp_path):
    p = tmp_path / "x.jpg"
    Image.new("L", (32, 32), color=128).save(p)
    m: dict[str, str] = {}
    a = pd.dhash64_file(p, m)
    b = pd.dhash64_file(p, m)
    assert a == b
    assert len(m) == 1


def test_dedupe_mode_legacy_bool():
    assert pd.dedupe_mode_from_config({"process_unique_pictures_only": True}) == "url"
    assert pd.dedupe_mode_from_config({"process_unique_pictures_mode": "phash"}) == "phash"


def test_dedupe_mode_legacy_russian_label_from_broken_ui():
    """Раньше в JSON попадала подпись Radio (рус.), а не ключ off|url|phash."""
    assert (
        pd.dedupe_mode_from_config(
            {"process_unique_pictures_mode": "По URL первой картинки (быстро)"}
        )
        == "url"
    )
    assert (
        pd.dedupe_mode_from_config(
            {
                "process_unique_pictures_mode": "По содержимому файла в кэше (dHash; разные URL, тот же файл — один вызов; чуть дольше на группировку)"
            }
        )
        == "phash"
    )


def test_group_phash_merges_identical_files(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    p = cache / "img.png"
    Image.new("RGB", (20, 20), color=(10, 20, 30)).save(p)

    def fake_ensure(url: str, cdir: Path, max_size: int):
        assert cdir == cache
        return p

    o1 = {"offer_id": "1", "picture_urls": ["https://a/x.jpg"]}
    o2 = {"offer_id": "2", "picture_urls": ["https://b/y.jpg"]}
    groups = pd.group_offers_by_picture_dedupe([o1, o2], "phash", cache, 512, fake_ensure)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_normalize_url_strips_query():
    a = "https://cdn.example.com/img/1.jpg?w=400&token=x"
    b = "https://cdn.example.com/img/1.jpg"
    assert pd.normalize_picture_url(a) == pd.normalize_picture_url(b)


def test_dhash256_same_content(tmp_path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.new("RGB", (60, 60), color=(200, 100, 50)).save(p1)
    Image.new("RGB", (60, 60), color=(200, 100, 50)).save(p2)
    m: dict[str, str] = {}
    assert pd.dhash256_file(p1, m) == pd.dhash256_file(p2, m)


def test_group_url_separate_urls(tmp_path):
    cache = tmp_path / "c"
    cache.mkdir()

    def no_cache(*_a, **_k):
        return None

    o1 = {"offer_id": "1", "picture_urls": ["https://a/1"]}
    o2 = {"offer_id": "2", "picture_urls": ["https://b/2"]}
    groups = pd.group_offers_by_picture_dedupe([o1, o2], "url", cache, 512, no_cache)
    assert len(groups) == 2


def test_group_url_same_path_different_query(tmp_path):
    cache = tmp_path / "c"
    cache.mkdir()

    def no_cache(*_a, **_k):
        return None

    o1 = {"offer_id": "1", "picture_urls": ["https://a/x.jpg?size=1"]}
    o2 = {"offer_id": "2", "picture_urls": ["https://a/x.jpg?size=2"]}
    groups = pd.group_offers_by_picture_dedupe([o1, o2], "url", cache, 512, no_cache)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_group_url_stop_event_tail_singletons():
    ev = threading.Event()
    ev.set()
    o1 = {"offer_id": "1", "picture_urls": ["https://a/1.jpg"]}
    o2 = {"offer_id": "2", "picture_urls": ["https://a/2.jpg"]}
    o3 = {"offer_id": "3", "picture_urls": ["https://a/3.jpg"]}
    groups = pd.group_offers_by_picture_dedupe([o1, o2, o3], "url", Path("/tmp"), 512, lambda *_a, **_k: None, stop_event=ev)
    assert len(groups) == 3
    assert all(len(g) == 1 for g in groups)
