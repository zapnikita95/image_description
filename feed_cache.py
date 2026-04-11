#!/usr/bin/env python3
"""
Feed cache: parse YML/XML feed into SQLite DB for fast repeated access.
Устойчивый парсинг: кодировки (BOM, cp1251, utf-8), lxml recover при «битом» XML,
потоковый разбор (iterparse) для больших файлов (сотни МБ — ГБ) без загрузки всего дерева в RAM.
"""

import json
import re
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from lxml import etree as LXML_ET

    _HAVE_LXML = True
except ImportError:
    LXML_ET = None
    _HAVE_LXML = False

# Долгие записи кэша + параллельные чтения из UI (Gradio) → без этого на Windows часто «database is locked»
SQLITE_TIMEOUT_SEC = 60.0


def sqlite_connect(db_path: str | Path) -> sqlite3.Connection:
    """Открыть SQLite с ожиданием блокировки и WAL (читатели не мешают писателю так сильно)."""
    con = sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT_SEC)
    try:
        con.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass
    try:
        con.execute("PRAGMA busy_timeout=60000")
    except sqlite3.Error:
        pass
    return con


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS offers (
    offer_id             TEXT PRIMARY KEY,
    name                 TEXT,
    category_id          TEXT,
    category             TEXT,
    picture_urls         TEXT,   -- JSON array of strings (flat, all sources)
    picture_tagged_urls  TEXT,   -- JSON object: {tag_name: [url, ...]}
    vendor               TEXT,
    url                  TEXT
);
CREATE TABLE IF NOT EXISTS categories (
    category_id  TEXT PRIMARY KEY,
    name         TEXT,
    parent_id    TEXT
);
"""

# Файлы больше этого порога — только потоковый разбор (и не пробуем DOM целиком).
STREAM_SIZE_THRESHOLD = 32 * 1024 * 1024
BATCH_INSERT = 3000

# YML — offer; Google Merchant / RSS — item; встречаются варианты product
OFFER_LIKE_TAGS = frozenset({"offer", "item", "product"})


# ── XML helpers ──────────────────────────────────────────────────────────────

def _local(tag: str) -> str:
    if not tag:
        return ""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _find(elem, *tags):
    for tag in tags:
        el = elem.find(tag)
        if el is not None:
            return el
        for child in elem:
            if _local(child.tag) == tag:
                return child
    return None


def _text(elem, *tags) -> str:
    el = _find(elem, *tags)
    return (el.text or "").strip() if el is not None else ""


def _text_any(elem, *tags) -> str:
    """Первое непустое текстовое поле по списку тегов (разные фиды: categoryId vs category_id)."""
    for tag in tags:
        v = _text(elem, tag)
        if v:
            return v
    return ""


def _normalize_media_url(raw: str | None) -> str | None:
    """http(s), протокол-относительные //cdn..., обрезка хвостовых символов из XML."""
    if not raw:
        return None
    u = raw.strip().strip('"').strip("'")
    if not u:
        return None
    u = u.rstrip(".,;)]}>")
    if u.startswith("//"):
        u = "https:" + u
    if not u.startswith(("http://", "https://")):
        return None
    return u


def _urls_from_media_element(el) -> list[str]:
    """URL картинки: текст элемента, атрибуты url/src/href, вложенный <url>."""
    out: list[str] = []
    nu = _normalize_media_url((el.text or "").strip())
    if nu:
        out.append(nu)
    for attr in ("url", "src", "href", "link", "value"):
        nu = _normalize_media_url((el.get(attr) or "").strip())
        if nu:
            out.append(nu)
    for child in el:
        loc = _local(child.tag).lower()
        if loc in ("url", "link", "src", "href"):
            nu = _normalize_media_url((child.text or "").strip())
            if nu:
                out.append(nu)
    return out


# Резерв: фиды в одну строку / картинка только в тексте или нестандартные вставки
_IMG_URL_RE = re.compile(
    r"(?:https?:)?//[^\s<>'\"]+\.(?:jpe?g|png|webp|gif)(?:\?[^\s<>'\"]*)?"
    r"|https?://[^\s<>'\"]+\.(?:jpe?g|png|webp|gif)(?:\?[^\s<>'\"]*)?",
    re.IGNORECASE,
)


def _fallback_image_urls_from_offer(offer) -> list[str]:
    """Если теги picture/param не сработали — вытащить URL картинок из любого текста оффера."""
    try:
        blob = "".join(offer.itertext() or [])
    except Exception:
        blob = ""
    if not blob:
        return []
    found: list[str] = []
    for m in _IMG_URL_RE.finditer(blob):
        nu = _normalize_media_url(m.group(0))
        if nu:
            found.append(nu)
    return list(dict.fromkeys(found))


def _param_name_suggests_image(name: str) -> bool:
    n = (name or "").lower()
    if not n:
        return False
    keys = (
        "picture",
        "image",
        "photo",
        "img",
        "thumb",
        "картин",
        "изображ",
        "фото",
    )
    return any(k in n for k in keys)


_MEDIA_TAG_NAMES = frozenset({
    "picture",
    "image",
    "picture_url",
    "img",
    "image_link",
    "imageurl",
    "largeimage",
    "picture_link",
    "thumbnail",
    "photo",
    "mobile_picture",
    "enclosure",
})


def _collect_tagged_picture_urls(offer) -> dict[str, list[str]]:
    """
    Собирает ссылки на изображения с оффера с привязкой к XML-тегу/param-имени.
    Возвращает dict: {tag_name: [url, ...]}
    Для <param name="Picture"> ключ будет "param:picture".
    Плоский список = list(dict.fromkeys(...all urls...)).
    """
    tagged: dict[str, list[str]] = {}

    def _add(key: str, urls: list[str]) -> None:
        k = key.lower()
        for u in urls:
            nu = _normalize_media_url(u)
            if nu:
                if k not in tagged:
                    tagged[k] = []
                if nu not in tagged[k]:
                    tagged[k].append(nu)

    for child in offer:
        loc = _local(child.tag)
        loc_l = loc.lower()
        if loc_l == "param":
            pname = (child.get("name") or child.get("unit") or "").strip()
            txt = (child.text or "").strip()
            if _param_name_suggests_image(pname) or _normalize_media_url(txt):
                key = f"param:{pname.lower()}" if pname else "param"
                urls = _urls_from_media_element(child)
                nu_txt = _normalize_media_url(txt)
                if nu_txt and nu_txt not in urls:
                    urls.append(nu_txt)
                _add(key, urls)
            continue
        if loc in _MEDIA_TAG_NAMES or loc_l in _MEDIA_TAG_NAMES:
            _add(loc_l, _urls_from_media_element(child))
            continue
        if "picture" in loc_l or loc_l.endswith("image_link") or (
            "image" in loc_l and "url" in loc_l
        ):
            _add(loc_l, _urls_from_media_element(child))

    return tagged


def _collect_picture_urls(offer) -> list[str]:
    """
    Собирает ссылки на изображения с оффера (плоский список, без тегов).
    Учитывает namespace (теги вида {ns}picture), картинки в атрибутах,
    вложенные url, теги вроде g:image_link, Yandex <param name="...">.
    """
    tagged = _collect_tagged_picture_urls(offer)
    out: list[str] = []
    for urls in tagged.values():
        for u in urls:
            if u not in out:
                out.append(u)
    return out


def _category_name_from_elem(cat) -> str:
    t = (cat.text or "").strip()
    if t:
        return t
    for child in cat:
        if _local(child.tag) in ("name", "title"):
            return (child.text or "").strip()
    return ""


def _full_category_name(cats: dict, cid: str, depth: int = 0) -> str:
    """Хлебные крошки: 'Одежда / Верхняя одежда / Куртки'."""
    if not cid or cid not in cats or depth > 8:
        return cats.get(cid, {}).get("name", cid) if cid else ""
    entry = cats[cid]
    parent = entry.get("parent_id", "")
    if parent and parent in cats:
        return _full_category_name(cats, parent, depth + 1) + " / " + entry["name"]
    return entry["name"]


def _normalize_xml_declared_encoding(enc: str) -> str:
    """
    В фидах часто пишут encoding='UTF8' без дефиса — приводим к имени кодека Python.
    """
    e = (enc or "").strip()
    if not e:
        return "utf-8"
    el = e.lower().replace("-", "").replace("_", "")
    aliases = {
        "utf8": "utf-8",
        "utf16": "utf-16",
        "utf16le": "utf-16-le",
        "utf16be": "utf-16-be",
        "utf32": "utf-32",
        "cp1251": "cp1251",
        "windows1251": "cp1251",
    }
    if el in aliases:
        return aliases[el]
    if el in ("windows1251",):
        return "cp1251"
    return e


def _detect_xml_encoding(feed_path: Path, sample_bytes: int = 65536) -> str:
    """BOM + объявление XML + эвристика."""
    raw = feed_path.read_bytes()[:sample_bytes]
    if raw.startswith(b"\xff\xfe"):
        return "utf-16"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    head = raw.decode("utf-8", errors="ignore")
    m = re.search(r'encoding\s*=\s*["\']([^"\']+)["\']', head[:2000], re.I)
    if m:
        enc = m.group(1).strip().lower()
        if enc in ("windows-1251", "cp1251"):
            return "cp1251"
        return _normalize_xml_declared_encoding(m.group(1).strip())
    # Частый случай: кириллица в cp1251 без декларации
    try:
        raw.decode("utf-8")
        return "utf-8"
    except Exception:
        pass
    return "cp1251"


def _strip_xml_illegal_chars(text: str) -> str:
    """Удаляет символы, запрещённые в XML 1.0."""
    out = []
    for ch in text:
        o = ord(ch)
        if o in (0x9, 0xA, 0xD) or 0x20 <= o <= 0xD7FF or 0xE000 <= o <= 0xFFFD or 0x10000 <= o <= 0x10FFFF:
            out.append(ch)
    return "".join(out)


def _prepare_text_for_stdlib_parse(feed_path: Path, encoding: str) -> Path:
    """
    Читает файл кусками, чистит управляющие символы — для ElementTree без recover.
    Возвращает путь к временному UTF-8 файлу.
    """
    fd, tmp = tempfile.mkstemp(suffix=".xml", prefix="feed_clean_")
    import os

    os.close(fd)
    out_path = Path(tmp)
    try:
        with open(feed_path, "rb") as fin, open(out_path, "w", encoding="utf-8", newline="\n") as fout:
            dec = encoding if encoding != "utf-16" else "utf-16"
            while True:
                chunk = fin.read(8 * 1024 * 1024)
                if not chunk:
                    break
                s = chunk.decode(dec, errors="replace")
                s = _strip_xml_illegal_chars(s)
                fout.write(s)
        return out_path
    except Exception:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        raise


def _offer_to_row(offer, cats: dict) -> dict | None:
    oid = str(
        offer.get("id") or offer.get("offerId") or offer.get("gid") or ""
    ).strip()
    name = _text_any(offer, "name", "title", "model")
    cat_id = _text_any(offer, "categoryId", "category_id", "categoryid", "cid")
    vendor = _text(offer, "vendor")
    url = _text_any(offer, "url", "link")

    tagged = _collect_tagged_picture_urls(offer)
    pic_urls: list[str] = []
    for urls in tagged.values():
        for u in urls:
            if u not in pic_urls:
                pic_urls.append(u)

    if not pic_urls:
        fallback = _fallback_image_urls_from_offer(offer)
        if fallback:
            tagged = {"_fallback": fallback}
            pic_urls = fallback
    if not pic_urls:
        return None

    return {
        "offer_id": oid,
        "name": name,
        "category_id": cat_id,
        "category": _full_category_name(cats, cat_id),
        "picture_urls": json.dumps(pic_urls, ensure_ascii=False),
        "picture_tagged_urls": json.dumps(tagged, ensure_ascii=False),
        "vendor": vendor,
        "url": url,
    }


def _collect_all_attr_names(offers_data: list[dict]) -> list[str]:
    """Собирает список уникальных тегов-источников картинок из всех офферов (из picture_tagged_urls)."""
    seen: set[str] = set()
    for row in offers_data:
        tagged_raw = row.get("picture_tagged_urls") or "{}"
        try:
            tagged = json.loads(tagged_raw) if isinstance(tagged_raw, str) else tagged_raw
        except Exception:
            tagged = {}
        for k in (tagged or {}).keys():
            seen.add(k)
    return sorted(seen)


def _flush_offers(con: sqlite3.Connection, rows: list[dict]) -> None:
    if not rows:
        return
    # Ensure picture_tagged_urls is present (backward compat with old rows)
    for row in rows:
        if "picture_tagged_urls" not in row:
            row["picture_tagged_urls"] = "{}"
    con.executemany(
        "INSERT OR REPLACE INTO offers "
        "VALUES (:offer_id,:name,:category_id,:category,:picture_urls,:picture_tagged_urls,:vendor,:url)",
        rows,
    )


def _parse_streaming_lxml(feed_path: Path, db_path: Path, feed_mtime: str) -> dict:
    """Потоковый разбор lxml — большие файлы, recover=True."""
    cats: dict[str, dict] = {}
    batch: list[dict] = []
    offers_data: list[dict] = []

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite_connect(db_path)
    con.executescript(SCHEMA)
    con.execute("DELETE FROM offers")
    con.execute("DELETE FROM categories")
    con.execute("DELETE FROM meta")

    # iterparse() не принимает keyword parser= в актуальном lxml — recover/huge_tree передаём напрямую.
    # ВАЖНО: elem.clear() + del prev делаем ТОЛЬКО после обработки offer/category,
    # иначе дочерние теги (<picture>, <param>...) удаляются ДО прихода события end для <offer>.
    context = LXML_ET.iterparse(
        str(feed_path),
        events=("end",),
        huge_tree=True,
        recover=True,
        strip_cdata=False,
    )
    for _event, elem in context:
        tag = _local(elem.tag)
        if tag == "category":
            cid = elem.get("id") or ""
            parent = elem.get("parentId") or elem.get("parent_id") or ""
            name = _category_name_from_elem(elem)
            if cid:
                cats[cid] = {"name": name, "parent_id": parent}
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        elif tag in OFFER_LIKE_TAGS:
            row = _offer_to_row(elem, cats)
            if row:
                batch.append(row)
                offers_data.append(row)
                if len(batch) >= BATCH_INSERT:
                    _flush_offers(con, batch)
                    con.commit()
                    batch.clear()
            # Освобождаем память только здесь — после чтения всех дочерних тегов оффера
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    _flush_offers(con, batch)
    con.commit()

    for cid, info in cats.items():
        con.execute(
            "INSERT OR REPLACE INTO categories VALUES (?,?,?)",
            (cid, info["name"], info["parent_id"]),
        )
    attr_names = _collect_all_attr_names(offers_data)
    con.execute("INSERT INTO meta VALUES ('feed_path', ?)", (str(feed_path.resolve()),))
    con.execute("INSERT INTO meta VALUES ('feed_mtime', ?)", (feed_mtime,))
    con.execute("INSERT INTO meta VALUES ('offer_count', ?)", (str(len(offers_data)),))
    con.execute("INSERT INTO meta VALUES ('picture_attr_names', ?)", (json.dumps(attr_names, ensure_ascii=False),))
    con.commit()
    con.close()

    category_counts: dict[str, int] = {}
    for row in offers_data:
        cat = row["category"] or "Без категории"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "total": len(offers_data),
        "categories": category_counts,
        "feed_mtime": feed_mtime,
        "feed_path": str(feed_path.resolve()),
    }


def _parse_streaming_etree(feed_path: Path, encoding: str, db_path: Path, feed_mtime: str) -> dict:
    """Потоковый разбор через ElementTree (без lxml)."""
    cats: dict[str, dict] = {}
    batch: list[dict] = []
    offers_data: list[dict] = []

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite_connect(db_path)
    con.executescript(SCHEMA)
    con.execute("DELETE FROM offers")
    con.execute("DELETE FROM categories")
    con.execute("DELETE FROM meta")

    clean_path = _prepare_text_for_stdlib_parse(feed_path, encoding)
    try:
        context = ET.iterparse(str(clean_path), events=("end",))
        for _event, elem in context:
            tag = _local(elem.tag)
            if tag == "category":
                cid = elem.get("id") or ""
                parent = elem.get("parentId") or elem.get("parent_id") or ""
                name = _category_name_from_elem(elem)
                if cid:
                    cats[cid] = {"name": name, "parent_id": parent}
            elif tag in OFFER_LIKE_TAGS:
                row = _offer_to_row(elem, cats)
                if row:
                    batch.append(row)
                    offers_data.append(row)
                    if len(batch) >= BATCH_INSERT:
                        _flush_offers(con, batch)
                        con.commit()
                        batch.clear()
            elem.clear()
    finally:
        clean_path.unlink(missing_ok=True)

    _flush_offers(con, batch)
    con.commit()

    for cid, info in cats.items():
        con.execute(
            "INSERT OR REPLACE INTO categories VALUES (?,?,?)",
            (cid, info["name"], info["parent_id"]),
        )
    attr_names = _collect_all_attr_names(offers_data)
    con.execute("INSERT INTO meta VALUES ('feed_path', ?)", (str(feed_path.resolve()),))
    con.execute("INSERT INTO meta VALUES ('feed_mtime', ?)", (feed_mtime,))
    con.execute("INSERT INTO meta VALUES ('offer_count', ?)", (str(len(offers_data)),))
    con.execute("INSERT INTO meta VALUES ('picture_attr_names', ?)", (json.dumps(attr_names, ensure_ascii=False),))
    con.commit()
    con.close()

    category_counts: dict[str, int] = {}
    for row in offers_data:
        cat = row["category"] or "Без категории"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "total": len(offers_data),
        "categories": category_counts,
        "feed_mtime": feed_mtime,
        "feed_path": str(feed_path.resolve()),
    }


def _parse_dom_lxml(feed_path: Path, db_path: Path, feed_mtime: str) -> dict:
    """Небольшие файлы: DOM + recover (чинит неэкранированные &, кривой XML)."""
    parser = LXML_ET.XMLParser(recover=True, huge_tree=True)
    tree = LXML_ET.parse(str(feed_path), parser=parser)
    root = tree.getroot()

    cats: dict = {}
    cats_root = root.find(".//categories")
    if cats_root is None:
        # фиды без обёртки categories — ищем category по дереву
        for cat in root.iter():
            if _local(cat.tag) != "category":
                continue
            cid = cat.get("id") or ""
            parent = cat.get("parentId") or cat.get("parent_id") or ""
            name = _category_name_from_elem(cat)
            if cid:
                cats[cid] = {"name": name, "parent_id": parent}
    else:
        for cat in cats_root:
            if _local(cat.tag) != "category":
                continue
            cid = cat.get("id") or ""
            parent = cat.get("parentId") or cat.get("parent_id") or ""
            name = _category_name_from_elem(cat)
            if cid:
                cats[cid] = {"name": name, "parent_id": parent}

    by_offer_id: dict[str, dict] = {}
    for offer in root.iter():
        if _local(offer.tag) not in OFFER_LIKE_TAGS:
            continue
        row = _offer_to_row(offer, cats)
        if row:
            by_offer_id[str(row["offer_id"])] = row
    offers_data = list(by_offer_id.values())

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite_connect(db_path)
    con.executescript(SCHEMA)
    con.execute("DELETE FROM offers")
    con.execute("DELETE FROM categories")
    con.execute("DELETE FROM meta")
    for row in offers_data:
        if "picture_tagged_urls" not in row:
            row["picture_tagged_urls"] = "{}"
    con.executemany(
        "INSERT OR REPLACE INTO offers "
        "VALUES (:offer_id,:name,:category_id,:category,:picture_urls,:picture_tagged_urls,:vendor,:url)",
        offers_data,
    )
    for cid, info in cats.items():
        con.execute(
            "INSERT OR REPLACE INTO categories VALUES (?,?,?)",
            (cid, info["name"], info["parent_id"]),
        )
    attr_names = _collect_all_attr_names(offers_data)
    con.execute("INSERT INTO meta VALUES ('feed_path', ?)", (str(feed_path.resolve()),))
    con.execute("INSERT INTO meta VALUES ('feed_mtime', ?)", (feed_mtime,))
    con.execute("INSERT INTO meta VALUES ('offer_count', ?)", (str(len(offers_data)),))
    con.execute("INSERT INTO meta VALUES ('picture_attr_names', ?)", (json.dumps(attr_names, ensure_ascii=False),))
    con.commit()
    con.close()

    category_counts: dict[str, int] = {}
    for row in offers_data:
        cat = row["category"] or "Без категории"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "total": len(offers_data),
        "categories": category_counts,
        "feed_mtime": feed_mtime,
        "feed_path": str(feed_path.resolve()),
    }


# ── Parse feed ───────────────────────────────────────────────────────────────

def parse_feed_to_cache(feed_path: str | Path, db_path: str | Path) -> dict:
    """
    Parse YML feed and populate SQLite cache.
    Returns summary dict: {total, categories, feed_mtime, feed_path}.
    """
    feed_path = Path(feed_path).resolve()
    db_path = Path(db_path)
    feed_mtime = str(feed_path.stat().st_mtime)
    size = feed_path.stat().st_size

    if not feed_path.exists():
        raise FileNotFoundError(str(feed_path))

    encoding = _detect_xml_encoding(feed_path)

    # 1) Большие файлы — только поток; без lxml полная чистка в UTF-8 на диск нецелесообразна
    if size > STREAM_SIZE_THRESHOLD:
        if not _HAVE_LXML:
            raise RuntimeError(
                "Для фидов больше ~32 МБ установите пакет lxml: pip install lxml "
                "(потоковый разбор с recover и без раздувания памяти)."
            )
        return _parse_streaming_lxml(feed_path, db_path, feed_mtime)

    # 2) Малые файлы: сначала lxml recover DOM (лучше переживает кривой XML из фидов)
    if _HAVE_LXML:
        try:
            return _parse_dom_lxml(feed_path, db_path, feed_mtime)
        except Exception:
            pass
        try:
            return _parse_streaming_lxml(feed_path, db_path, feed_mtime)
        except Exception:
            pass

    # 3) Без lxml — чистим и поток
    return _parse_streaming_etree(feed_path, encoding, db_path, feed_mtime)


# ── Query cache (unchanged API) ──────────────────────────────────────────────

def is_cache_valid(db_path: str | Path, feed_path: str | Path) -> bool:
    """Return True if cache exists and matches current feed mtime."""
    db_path = Path(db_path)
    feed_path = Path(feed_path)
    if not db_path.exists():
        return False
    try:
        current_mtime = str(feed_path.stat().st_mtime)
        con = sqlite_connect(db_path)
        row = con.execute("SELECT value FROM meta WHERE key='feed_mtime'").fetchone()
        con.close()
        return row is not None and row[0] == current_mtime
    except Exception:
        return False


def get_categories(db_path: str | Path) -> dict[str, int]:
    """Return {category_name: count} dict from cache."""
    con = sqlite_connect(db_path)
    rows = con.execute(
        "SELECT category, COUNT(*) FROM offers GROUP BY category ORDER BY category"
    ).fetchall()
    con.close()
    return {r[0] or "Без категории": r[1] for r in rows}


def get_cache_meta(db_path: str | Path) -> dict:
    con = sqlite_connect(db_path)
    rows = con.execute("SELECT key, value FROM meta").fetchall()
    con.close()
    return dict(rows)


def get_feed_image_attr_names(db_path: str | Path) -> list[str]:
    """
    Возвращает список XML-тегов/param-имён, в которых были найдены URL картинок при парсинге фида.
    Например: ["picture", "param:picture", "photo"].
    Пустой список если фид ещё не распознан или данных нет.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    try:
        con = sqlite_connect(db_path)
        row = con.execute("SELECT value FROM meta WHERE key='picture_attr_names'").fetchone()
        con.close()
        if row and row[0]:
            return json.loads(row[0])
    except Exception:
        pass
    return []


def _parse_offer_row(d: dict) -> dict:
    """Десериализует JSON-поля оффера из SQLite-строки."""
    d["picture_urls"] = json.loads(d.get("picture_urls") or "[]")
    try:
        d["picture_tagged_urls"] = json.loads(d.get("picture_tagged_urls") or "{}")
    except Exception:
        d["picture_tagged_urls"] = {}
    return d


def _filter_offer_picture_urls(d: dict, attr_filter: list[str] | None) -> dict:
    """
    Если attr_filter задан — оставляет в picture_urls только URL из указанных тегов.
    picture_tagged_urls не меняется.
    """
    if not attr_filter:
        return d
    tagged = d.get("picture_tagged_urls") or {}
    filtered: list[str] = []
    for key in attr_filter:
        for u in tagged.get(key, []):
            if u not in filtered:
                filtered.append(u)
    if filtered:
        d = dict(d)
        d["picture_urls"] = filtered
    return d


def get_offers(
    db_path: str | Path,
    categories: list[str] | None = None,
    limit: int = 0,
    offset: int = 0,
    picture_attr_filter: list[str] | None = None,
) -> list[dict]:
    """
    Fetch offers from cache.
    If categories is None or empty — return all.
    limit=0 means no limit.
    picture_attr_filter: если задан — picture_urls будет содержать только URL из этих тегов.
    """
    con = sqlite_connect(db_path)
    con.row_factory = sqlite3.Row

    if categories:
        placeholders = ",".join("?" * len(categories))
        query = f"SELECT * FROM offers WHERE category IN ({placeholders})"
        params: list = list(categories)
    else:
        query = "SELECT * FROM offers"
        params = []

    if limit > 0:
        query += " LIMIT ? OFFSET ?"
        params += [limit, offset]

    rows = con.execute(query, params).fetchall()
    con.close()

    result = []
    for row in rows:
        d = _parse_offer_row(dict(row))
        d = _filter_offer_picture_urls(d, picture_attr_filter)
        result.append(d)
    return result


def get_offer_by_id(
    db_path: str | Path,
    offer_id: str,
    picture_attr_filter: list[str] | None = None,
) -> dict | None:
    """Один оффер из кэша по offer_id (для повторной обработки с вкладки «Результаты»)."""
    if not offer_id or not db_path:
        return None
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    con = sqlite_connect(db_path)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM offers WHERE offer_id = ?", (str(offer_id),)).fetchone()
    con.close()
    if not row:
        return None
    d = _parse_offer_row(dict(row))
    d = _filter_offer_picture_urls(d, picture_attr_filter)
    return d


def count_offers(db_path: str | Path, categories: list[str] | None = None) -> int:
    con = sqlite_connect(db_path)
    if categories:
        placeholders = ",".join("?" * len(categories))
        n = con.execute(
            f"SELECT COUNT(*) FROM offers WHERE category IN ({placeholders})",
            list(categories),
        ).fetchone()[0]
    else:
        n = con.execute("SELECT COUNT(*) FROM offers").fetchone()[0]
    con.close()
    return n
