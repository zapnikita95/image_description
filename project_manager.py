#!/usr/bin/env python3
"""Project management: create, load, save project configs to projects/<name>/config.json."""

import copy
import json
import re
from pathlib import Path

PROJECTS_DIR = Path(__file__).parent / "projects"
APP_SETTINGS_PATH = Path(__file__).parent / "app_settings.json"
LAST_PROJECT_PATH = Path(__file__).parent / "last_project.txt"
ATTRIBUTE_GLOSSARY_PATH = Path(__file__).parent / "attribute_glossary.json"

# Глобальные настройки (модель, Ollama, адрес для Tailscale) — общие для всех проектов.
# ollama_url по умолчанию :11435 — локальный пул (ollama-queue-proxy), который ограничивает параллель
# запросов к настоящему Ollama (:11434) и хранит отложенные задания. Прямой :11434 обходит очередь.
GLOBAL_DEFAULTS = {
    "model": "qwen3.5:35b",
    "ollama_url": "http://127.0.0.1:11435",
    "image_max_size": 1024,
    "app_public_host": "",  # IP или хост для доступа с другого ПК (Tailscale), напр. 100.115.68.2
    # 0 = без лимита (как раньше: все vision-задачи оффера параллельно). >0 — верхняя граница потоков к Ollama.
    "max_parallel_vision": 0,
    # Сколько групп дедупа (уникальных картинок) обрабатывать одновременно в батч-запуске. 1 — по очереди.
    "batch_offer_workers": 1,
    # Параллельных HTTP к upstream через ollama-queue-proxy (:11435). Сохраняется в UI и шлётся в пул API.
    "ollama_pool_http_concurrency": 3,
    # После глоссария: текстовый запрос к Ollama для value, где осталась латиница (≥3 подряд).
    "attribute_value_llm_translate": False,
    "attribute_value_translate_model": "",  # пусто = та же модель, что основная vision
}


def parse_model_size_billions(model_name: str) -> float | None:
    """Из имени вроде qwen3.5:9b или qwen2.5-vl:7b извлечь размер в «B» (берётся максимум из всех :Nb)."""
    s = (model_name or "").lower()
    matches = re.findall(r"(?<![\w.])(\d+)b\b", s)
    if not matches:
        return None
    return max(float(x) for x in matches)


def recommended_batch_offer_workers(
    model_name: str,
    *,
    vram_gb: float = 24.0,
    ram_gb: float = 64.0,
) -> int:
    """
    Сколько групп (уникальных картинок) разумно гонять параллельно на одной GPU.
    Ориентир: ~24 ГБ VRAM; RAM учитывается слабо (узкое место — видеопамять под вес модели).
    """
    _ = ram_gb  # резерв: при CPU-инференсе можно поднять лимиты отдельно
    b = parse_model_size_billions(model_name)
    if b is None:
        return 2
    if b <= 4:
        return 4
    if b <= 8:
        return 3
    if b <= 14:
        return 2
    if b <= 40:
        return 1 if vram_gb < 48 else 2
    return 1


def get_global_settings() -> dict:
    """Настройки модели и Ollama, общие для всех проектов."""
    if not APP_SETTINGS_PATH.exists():
        return dict(GLOBAL_DEFAULTS)
    try:
        with open(APP_SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {**GLOBAL_DEFAULTS, **data}
    except Exception:
        return dict(GLOBAL_DEFAULTS)


def save_global_settings(settings: dict) -> None:
    """Сохранить глобальные настройки (модель, ollama_url, image_max_size, app_public_host)."""
    allowed = set(GLOBAL_DEFAULTS)
    with open(APP_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in settings.items() if k in allowed}, f, ensure_ascii=False, indent=2)


def get_last_project() -> str | None:
    """Имя последнего выбранного проекта (для восстановления после обновления страницы)."""
    if not LAST_PROJECT_PATH.exists():
        return None
    try:
        name = LAST_PROJECT_PATH.read_text(encoding="utf-8").strip()
        return name if name else None
    except Exception:
        return None


def set_last_project(name: str) -> None:
    """Сохранить имя последнего выбранного проекта."""
    try:
        LAST_PROJECT_PATH.write_text(name.strip(), encoding="utf-8")
    except Exception:
        pass


def load_attribute_glossary() -> dict[str, str]:
    """Глоссарий EN -> RU для отображения значений атрибутов. Ключи в нижнем регистре."""
    if not ATTRIBUTE_GLOSSARY_PATH.exists():
        return {}
    try:
        with open(ATTRIBUTE_GLOSSARY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {str(k).strip().lower(): str(v).strip() for k, v in (data or {}).items()}
    except Exception:
        return {}


def _glossary_match_at(lowered: str, start: int, k: str) -> bool:
    """Совпадение ключа на позиции без «склейки» с соседними латинскими буквами (не режем shorts → short)."""
    L = len(k)
    n = len(lowered)
    if L == 0 or start + L > n:
        return False
    if lowered[start : start + L] != k:
        return False
    if start > 0 and lowered[start - 1].isascii() and lowered[start - 1].isalnum():
        return False
    end = start + L
    if end < n and lowered[end].isascii() and lowered[end].isalnum():
        return False
    return True


def _translate_glossary_longest_scan(segment: str, g: dict[str, str]) -> str:
    """
    Идём слева направо: на каждой позиции берём самое длинное совпадение ключа глоссария (нижний регистр).
    Покрывает фразы вроде «cotton mesh upper», если в JSON есть «cotton mesh», «mesh», «upper».
    """
    if not segment or not g:
        return segment
    keys_sorted = sorted((k for k in g if k), key=len, reverse=True)
    lowered = segment.lower()
    i = 0
    n = len(segment)
    out: list[str] = []
    while i < n:
        matched_len = 0
        repl = ""
        for k in keys_sorted:
            L = len(k)
            if _glossary_match_at(lowered, i, k):
                matched_len = L
                repl = g[k]
                break
        if matched_len:
            out.append(repl)
            i += matched_len
        else:
            out.append(segment[i])
            i += 1
    return "".join(out)


def _translate_ascii_tokens(segment: str, g: dict[str, str]) -> str:
    """Оставшиеся цельные англ. слова (латиница + дефис) подменяем по глоссарию."""

    def repl(m: re.Match) -> str:
        w = m.group(0)
        lw = w.lower()
        return g.get(lw, w)

    return re.sub(r"\b[A-Za-z][A-Za-z\-]*\b", repl, segment)


_SPATIAL_LONG_ATTR_KEYS = frozenset({"sleeve_length", "length"})


def fix_ru_spatial_length_words(attr_key: str, value: str) -> str:
    """
    Модель путает «долгий» (о времени) с «длинный» (о физической длине рукава/изделия).
    """
    if attr_key not in _SPATIAL_LONG_ATTR_KEYS or not value or not isinstance(value, str):
        return value
    s = value
    for wrong, right in (
        ("долгие", "длинные"),
        ("Долгие", "Длинные"),
        ("долгая", "длинная"),
        ("Долгая", "Длинная"),
        ("долгое", "длинное"),
        ("Долгое", "Длинное"),
        ("долгий", "длинный"),
        ("Долгий", "Длинный"),
    ):
        s = s.replace(wrong, right)
    return s


def translate_attribute_value(value: str, glossary: dict[str, str] | None = None) -> str:
    """
    EN → RU для отображения и БД (после analyze_offer).
    1) полное совпадение строки; 2) части через запятую / «;» / «/»; 3) вхождения фраз из глоссария (длинные первыми);
    4) отдельные латинские слова. Полный машинный перевод всех языков не делаем — только словарь + то, что модель уже дала по-русски.
    """
    if not value:
        return value
    g = glossary if glossary is not None else load_attribute_glossary()
    key = value.strip().lower()
    if key in g:
        return g[key]

    splitters = (",", ";", "/")
    if any(sp in value for sp in splitters):
        # Сначала запятая (частые списки), потом ; и /
        if "," in value:
            parts = [translate_attribute_value(p.strip(), g) for p in value.split(",")]
            return ", ".join(parts)
        if ";" in value:
            parts = [translate_attribute_value(p.strip(), g) for p in value.split(";")]
            return "; ".join(parts)
        if "/" in value:
            parts = [translate_attribute_value(p.strip(), g) for p in value.split("/")]
            return " / ".join(parts)

    v = value.strip()
    scanned = _translate_glossary_longest_scan(v, g)
    return _translate_ascii_tokens(scanned, g)


# Токены «однотонность» несовместимы с явным узором — модель часто склеивает «однотонный, клетка».
_PRINT_PATTERN_PLAIN_PHRASES = frozenset(
    {
        "однотонный",
        "однотонная",
        "однотонное",
        "однотонные",
        "plain",
        "solid",
        "monochrome",
        "монохром",
        "монохромный",
        "монохромная",
    }
)
# Подстроки в нижнем регистре: если есть узор/фактура рисунка — убираем plain-токены.
_PRINT_PATTERN_STRUCTURED_MARKERS = (
    "клетка",
    "клетчат",
    "шотланд",
    "гусиная лапка",
    "ёлочк",
    "елочк",
    "houndstooth",
    "dogstooth",
    "pied-de-poule",
    "полоск",
    "striped",
    "stripe",
    "горошек",
    "горох",
    "polka",
    "принт",
    "узор",
    "паисл",
    "пейсли",
    "цветоч",
    "floral",
    "геометр",
    "geometric",
    "абстракт",
    "abstract",
    "графич",
    "graphic",
    "леопард",
    "зебр",
    "змеин",
    "камуфляж",
    "camo",
    "пайетк",
    "sequin",
    "люрекс",
    "lurex",
    "меланж",
    "melange",
    "marl",
    "heather",
    "мелирован",
    "пёстрый",
    "пестрый",
    "speckled",
    "вязк",
    "трикотажн",
    "рибан",
    "ribbed",
    "косич",
    "cable",
    "жаккард",
    "jacquard",
    "аран",
    "двутон",
    "двухцвет",
    "two-tone",
    "duotone",
    "мультицвет",
    "multicolor",
    "мульти",
    "контрастн",
    "animal print",
    "lettering",
    "slogan",
    "logo",
    "tie-dye",
    "тай-дай",
    "батик",
    "ethnic",
    "трайбл",
    "checked",
    "plaid",
    "tartan",
    "herringbone",
    "ёлочка",
    "елочка",
    "в клетку",
    "клетку",
    "букле",
    "boucle",
    "bouclé",
    "nubby",
    "барашек",
    "плюш",
    "петельн",
    "ёжик",
    "ежик",
    "тедди",
    "экомех",
    "овечк",
)


def sanitize_print_pattern_value(value: str) -> str:
    """
    Убирает «однотонный» рядом с реальным узором (после глоссария, RU/EN).
    «однотонный, клетка» → «клетка»; «однотонный, двутонный» → «двутонный».
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return s
    blob = s.lower()
    has_structured = any(m in blob for m in _PRINT_PATTERN_STRUCTURED_MARKERS)
    if not has_structured:
        return s
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return s
    kept: list[str] = []
    for p in parts:
        pl = p.strip().lower()
        if pl in _PRINT_PATTERN_PLAIN_PHRASES:
            continue
        kept.append(p.strip())
    if not kept:
        return s
    return ", ".join(kept)


def sanitize_print_pattern_in_direction_inplace(direction_attributes: dict | None) -> None:
    """Постобработка print_pattern после перевода / LLM-дожима."""
    if not direction_attributes or not isinstance(direction_attributes, dict):
        return
    for _did, attrs in direction_attributes.items():
        if not isinstance(attrs, dict) or attrs.get("error"):
            continue
        block = attrs.get("print_pattern")
        if not isinstance(block, dict):
            continue
        raw = block.get("value")
        if isinstance(raw, str) and raw.strip():
            block["value"] = sanitize_print_pattern_value(raw)


_FORBIDDEN_OUTPUT_ATTR_KEYS = frozenset({"original_name"})

# yes/no-подобные поля: модель любит «присутствуют», «отсутствие» — приводим к коротким да/нет.
_PRESENCE_LIKE_ATTR_KEYS = frozenset({"pockets", "fastener", "hood"})


def _normalize_presence_like_value(attr_key: str, value: str) -> str:
    if attr_key not in _PRESENCE_LIKE_ATTR_KEYS:
        return value
    s = (value or "").strip()
    if not s:
        return s
    low = re.sub(r"\s+", " ", s.lower())
    if low in ("наличие", "присутствует", "присутствуют", "имеется"):
        return "да"
    if low in ("отсутствие", "отсутствует"):
        return "нет"
    # «застёжка - отсутствие», «карманы — наличие»
    m = re.match(
        r"^(.+?)\s*[-–—]\s*(отсутствие|отсутствует|наличие|присутствует|присутствуют)\s*$",
        low,
    )
    if m:
        tail = m.group(2)
        if tail.startswith("отсутств"):
            return "нет"
        return "да"
    if re.search(r"\bприсутствуют\b$", low) and len(low) < 35:
        return re.sub(r"\s*присутствуют\s*$", "", s, flags=re.I).strip() or "да"
    if re.search(r"\bприсутствует\b$", low) and len(low) < 35:
        return re.sub(r"\s*присутствует\s*$", "", s, flags=re.I).strip() or "да"
    return s


def normalize_presence_like_attribute_values_inplace(direction_attributes: dict | None) -> None:
    if not direction_attributes or not isinstance(direction_attributes, dict):
        return
    for _did, attrs in direction_attributes.items():
        if not isinstance(attrs, dict) or attrs.get("error"):
            continue
        for k, v in attrs.items():
            if k == "error" or not isinstance(v, dict):
                continue
            raw = v.get("value")
            if isinstance(raw, str) and raw.strip():
                v["value"] = _normalize_presence_like_value(str(k), raw)


def strip_forbidden_attribute_keys_inplace(direction_attributes: dict | None) -> None:
    """Для clothing: original_name не сохраняем (дублирует название оффера). Другие направления не трогаем."""
    if not direction_attributes or not isinstance(direction_attributes, dict):
        return
    for did, attrs in direction_attributes.items():
        if str(did).strip() != "clothing" or not isinstance(attrs, dict):
            continue
        for k in list(attrs.keys()):
            if k == "error":
                continue
            if str(k).strip().lower() in _FORBIDDEN_OUTPUT_ATTR_KEYS:
                attrs.pop(k, None)


def translate_direction_attribute_values_inplace(
    direction_attributes: dict | None,
    glossary: dict[str, str] | None = None,
) -> None:
    """
    Переводит поля value в direction_attributes (после ответа модели на EN).
    Надписи на фото (text_detection) сюда не передаются — не трогаем.
    """
    if not direction_attributes or not isinstance(direction_attributes, dict):
        return
    g = glossary if glossary is not None else load_attribute_glossary()
    for _did, attrs in direction_attributes.items():
        if not isinstance(attrs, dict):
            continue
        for k, v in attrs.items():
            if k == "error" or not isinstance(v, dict):
                continue
            raw = v.get("value")
            if isinstance(raw, str) and raw.strip():
                raw = fix_ru_spatial_length_words(k, raw)
                v["value"] = translate_attribute_value(raw, g)
                if (k or "").strip() == "print_pattern":
                    v["value"] = sanitize_print_pattern_value(v["value"])
                v["value"] = _normalize_presence_like_value(str(k), v["value"])


# Заглушки: модель часто выбирает их с высокой уверенностью — в БД/карточках не нужны (очищаем после перевода).
_PLACEHOLDER_VALUE_TOKENS = frozenset(
    {
        "unknown",
        "known",
        "undefined",
        "n/a",
        "na",
        "tbd",
        "null",
        "неизвестно",
        "неизвестный",
        "неизвестная",
        "неизвестен",
        "неизвестны",
        "известный",
        "известная",
        "известно",
        "неузнаваемый",
        "неузнаваемая",
        "неузнаваемое",
        "неузнаваемые",
        "известняк",
        "известняковый",
        "?",
        "—",
        "-",
        "…",
        "...",
    }
)


def attribute_value_is_placeholder_noise(value: str) -> bool:
    """
    True, если value — отмазка модели (неизвестно / неузнаваемый / известняк / known…), а не цвет или признак.
    Используется при разборе JSON и на постобработке после глоссария.
    """
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s:
        return True
    sl = s.lower()
    if len(sl) <= 2 and sl in ("?", "-", "—", "…"):
        return True
    if sl in _PLACEHOLDER_VALUE_TOKENS:
        return True
    # Любые формы «неизвест…», «неузнава…», «не определ…»
    if sl.startswith("неизвест"):
        return True
    if sl.startswith("неузнава"):
        return True
    if sl.startswith("не определ") or sl.startswith("неопредел"):
        return True
    if "невозможно определить" in sl or "нельзя определить" in sl or "не могу определить" in sl:
        return True
    return False


def _token_is_placeholder(tok: str) -> bool:
    return attribute_value_is_placeholder_noise(tok)


def _strip_placeholders_from_attribute_value(s: str) -> str | None:
    """
    Убрать заглушки из значения. Возвращает None, если нечего сохранить (ключ атрибута лучше удалить).
    Списки через запятую: оставляем только осмысленные части.
    """
    s = (s or "").strip()
    if not s:
        return None
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        kept = [p for p in parts if p and not _token_is_placeholder(p)]
        if not kept:
            return None
        return ", ".join(kept)
    if _token_is_placeholder(s):
        return None
    return s


def strip_placeholder_attribute_values_inplace(direction_attributes: dict | None) -> None:
    """
    После translate: убрать unknown/неизвестно/known и т.п. — пустое значение = удалить ключ атрибута из направления.
    """
    if not direction_attributes or not isinstance(direction_attributes, dict):
        return
    for _did, attrs in direction_attributes.items():
        if not isinstance(attrs, dict) or attrs.get("error"):
            continue
        for k in list(attrs.keys()):
            if k == "error":
                continue
            v = attrs.get(k)
            if not isinstance(v, dict):
                continue
            raw = v.get("value")
            if not isinstance(raw, str):
                continue
            cleaned = _strip_placeholders_from_attribute_value(raw)
            if cleaned is None:
                attrs.pop(k, None)
            else:
                v["value"] = cleaned


def reverse_glossary(glossary: dict[str, str] | None = None) -> dict[str, str]:
    """RU -> EN для подстановки при сохранении правок (пользователь пишет по-русски)."""
    g = glossary if glossary is not None else load_attribute_glossary()
    return {str(v).strip().lower(): k for k, v in g.items() if k and v}


def _label_to_key_and_known_keys(config: dict) -> tuple[dict[str, str], set[str]]:
    """По конфигу направлений: {русский label: key}, set(все key)."""
    label_to_key = {}
    known_keys = set()
    for d in config.get("directions", []):
        for a in d.get("attributes", []):
            key = a.get("key", "")
            label = (a.get("label") or key).strip()
            if key:
                known_keys.add(key)
                if label:
                    label_to_key[label] = key
                    label_to_key[label.lower()] = key
    return label_to_key, known_keys


def normalize_correction_attrs(
    user_attrs: dict, config: dict, glossary: dict[str, str] | None = None
) -> dict[str, str]:
    """
    Приводит атрибуты из формы правки к виду для сохранения: ключи EN, значения EN.
    user_attrs может содержать ключи на русском (label) и значения на русском.
    """
    if not user_attrs:
        return {}
    g = glossary if glossary is not None else load_attribute_glossary()
    rev = reverse_glossary(g)
    label_to_key, known_keys = _label_to_key_and_known_keys(config)
    out = {}
    for k, v in (user_attrs or {}).items():
        if not isinstance(v, (str, int, float)):
            continue
        v_str = str(v).strip()
        key = k if k in known_keys else label_to_key.get(k) or label_to_key.get(k.strip().lower()) or k
        if v_str:
            out[key] = rev.get(v_str.lower(), v_str)
    return out


def attrs_to_russian_json(attrs: dict[str, str], config: dict, glossary: dict[str, str] | None = None) -> str:
    """Собирает JSON для отображения в форме правки: ключи и значения по-русски."""
    if not attrs:
        return "{}"
    g = glossary if glossary is not None else load_attribute_glossary()
    _, known_keys = _label_to_key_and_known_keys(config)
    key_to_label = {}
    for d in config.get("directions", []):
        for a in d.get("attributes", []):
            if a.get("key"):
                key_to_label[a["key"]] = a.get("label") or a["key"]
    out = {}
    for k, v in (attrs or {}).items():
        label = key_to_label.get(k) or k
        out[label] = translate_attribute_value(v, g)
    return json.dumps(out, ensure_ascii=False, indent=2)


# Палитра только для одежды/аксессуаров из ткани (мода). Другие вертикали — свои атрибуты/промпты, не сюда.
# Базовая гамма (~30): общие названия. Нюансы — ключ color_shade.
_CLOTHING_COLOR_BASE_OPTIONS = [
    "black",
    "white",
    "grey",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "beige",
    "navy",
    "burgundy",
    "maroon",
    "gold",
    "silver",
    "cream",
    "ivory",
    "khaki",
    "olive",
    "tan",
    "charcoal",
    "bronze",
    "copper",
    "turquoise",
    "multicolor",
    "metallic",
    "rose",
]

# Нюансы и модные оттенки одежды (EN в options → RU через attribute_glossary.json).
_CLOTHING_COLOR_SHADE_OPTIONS = [
    "dusty rose",
    "blush",
    "blush pink",
    "hot pink",
    "baby pink",
    "pale pink",
    "dark pink",
    "nude",
    "fuchsia",
    "magenta",
    "lavender",
    "lilac",
    "mauve",
    "plum",
    "orchid",
    "eggplant",
    "aubergine",
    "violet",
    "periwinkle",
    "amethyst",
    "indigo",
    "cobalt blue",
    "royal blue",
    "electric blue",
    "sky blue",
    "powder blue",
    "baby blue",
    "ice blue",
    "midnight blue",
    "denim blue",
    "steel blue",
    "slate blue",
    "cornflower blue",
    "cerulean",
    "azure",
    "teal",
    "aqua",
    "aquamarine",
    "cyan",
    "mint",
    "mint green",
    "sage",
    "sage green",
    "lime",
    "lime green",
    "emerald",
    "emerald green",
    "jade",
    "forest green",
    "army green",
    "hunter green",
    "bottle green",
    "pine green",
    "sea green",
    "olive drab",
    "moss green",
    "chartreuse",
    "neon green",
    "neon pink",
    "neon yellow",
    "neon orange",
    "neon blue",
    "mustard",
    "mustard yellow",
    "lemon",
    "lemon yellow",
    "coral",
    "salmon",
    "peach",
    "apricot",
    "melon",
    "terracotta",
    "rust",
    "burnt orange",
    "amber",
    "tangerine",
    "pumpkin",
    "scarlet",
    "crimson",
    "cherry",
    "raspberry",
    "tomato",
    "brick red",
    "wine",
    "sangria",
    "rose gold",
    "white gold",
    "champagne",
    "pearl",
    "taupe",
    "ecru",
    "greige",
    "mocha",
    "espresso",
    "cappuccino",
    "latte",
    "chocolate",
    "cinnamon",
    "caramel",
    "toffee",
    "walnut",
    "chestnut",
    "mahogany",
    "sand",
    "stone",
    "smoke",
    "dove grey",
    "pewter",
    "gunmetal",
    "silver grey",
    "heathered",
    "washed",
    "acid wash",
    "distressed",
    "ombre",
    "iridescent",
    "holographic",
    "colorblock",
    "two-tone",
    "rainbow",
    # Нейтрали, варка денима, «модные» зелёные/красные — часто в фидах и fashion-атрибутах (в т.ч. DeepFashion-подобные таксономии).
    "aegean blue",
    "antique white",
    "biscuit",
    "bleached",
    "bubblegum pink",
    "butter yellow",
    "camel",
    "celery green",
    "clay",
    "cocoa",
    "coffee",
    "dark wash",
    "duck green",
    "dusty pink",
    "eucalyptus",
    "evergreen",
    "fog grey",
    "ginger",
    "hazelnut",
    "heather grey",
    "ice grey",
    "iron grey",
    "kelly green",
    "light wash",
    "lilac grey",
    "marigold",
    "marl",
    "mist grey",
    "oatmeal",
    "oxblood",
    "oyster",
    "petrol",
    "petrol blue",
    "powder pink",
    "putty",
    "raw indigo",
    "seafoam",
    "speckled",
    "spruce green",
    "stonewash",
    "sunflower yellow",
    "tobacco",
    "ultraviolet",
    "undyed",
]


# Стартовый набор атрибутов только как дефолт для направления «Одежда» — всё редактируется в настройках.
# options исторически на EN — в промпте просим русские value; глоссарий дожимает остатки EN (фразы и слова).
# Паттерны, материалы, принты, детали — богатый словарь для 3.5 35B и дообученных моделей.
DEFAULT_DIRECTIONS = [
    {
        "id": "clothing",
        "name": "Одежда",
        "text_enabled": True,
        "attributes": [
            {"key": "sleeve_length", "label": "Длина рукава", "options": ["short", "long", "3/4", "sleeveless", "cap", "batwing"]},
            {
                "key": "fastener",
                "label": "Застёжка",
                "options": [
                    "buttons",
                    "zipper",
                    "snaps",
                    "buckle",
                    "belt",
                    "tie",
                    "sash",
                    "drawstring",
                    "hook_and_eye",
                    "toggle",
                    "velcro",
                    "laces",
                    "hook",
                    "magnetic",
                    "wrap",
                    "none",
                ],
            },
            {"key": "hood", "label": "Капюшон", "options": ["yes", "no"]},
            {"key": "collar", "label": "Воротник", "options": ["crew", "v-neck", "polo", "turtleneck", "mandarin", "shawl", "notch", "none"]},
            {"key": "pockets", "label": "Карманы", "options": ["yes", "no", "patch", "inset"]},
            {"key": "length", "label": "Длина (юбка/платье и т.д.)", "options": ["mini", "midi", "maxi", "short", "knee", "ankle", "cropped"]},
            {"key": "print_pattern", "label": "Паттерн / принт", "options": [
                "plain", "melange", "marl", "heather", "knit texture", "ribbed knit", "boucle knit", "nubby boucle",
                "teddy texture", "satin sheen", "lurex shimmer",
                "sequin pattern", "striped", "checked", "plaid", "houndstooth", "herringbone",
                "floral", "paisley", "geometric", "abstract", "graphic", "animal print", "leopard", "zebra", "snake",
                "python skin", "snakeskin",
                "polka dot", "camo", "tie-dye", "ethnic", "tribal", "lettering", "slogan", "logo",
                "other"
            ]},
            {"key": "color", "label": "Цвет (базовый)", "options": list(_CLOTHING_COLOR_BASE_OPTIONS)},
            {"key": "color_shade", "label": "Оттенок", "options": list(_CLOTHING_COLOR_SHADE_OPTIONS)},
            {"key": "material", "label": "Материал (основной)", "options": [
                "cotton", "linen", "silk", "satin", "lurex", "velvet", "denim", "jersey", "knit", "wool", "fleece",
                "lace", "chiffon", "organza", "tulle", "leather", "suede", "synthetic", "mixed",
                "other"
            ]},
            {"key": "details_decor", "label": "Детали и декор", "options": [
                "none", "ruffles", "bows", "sequins", "embroidery", "applique", "beads", "studs",
                "lace trim", "ribbon", "buttons decorative", "patches", "pleats", "gathering",
                "cutouts", "slits", "fringes", "other"
            ]},
            {"key": "gender_target", "label": "Целевой пол", "options": ["men", "women", "unisex", "kids"]},
        ],
        "custom_prompt": "",
    },
    {
        "id": "other",
        "name": "Другое",
        "text_enabled": False,
        "attributes": [],
        "custom_prompt": "",
    },
]


def default_clothing_attribute_catalog() -> list[tuple[str, str]]:
    """(подпись для UI, key) — шаблонные атрибуты направления clothing из DEFAULT_DIRECTIONS."""
    for d in DEFAULT_DIRECTIONS:
        if d.get("id") != "clothing":
            continue
        out: list[tuple[str, str]] = []
        for a in d.get("attributes") or []:
            k = (a.get("key") or "").strip()
            if not k:
                continue
            lab = (a.get("label") or k).strip()
            out.append((f"{lab} ({k})", k))
        return out
    return []


def default_clothing_standard_keys() -> list[str]:
    """Ключи шаблонных атрибутов одежды (для фильтрации промпта и сортировки карточек)."""
    return [k for _, k in default_clothing_attribute_catalog()]


# Только настройки проекта. Модель и Ollama — в глобальных настройках (get_global_settings).
DEFAULT_CONFIG = {
    "name": "",
    "feed_path": "",
    "output_dir": "",
    "confidence_threshold": 50,
    "directions": DEFAULT_DIRECTIONS,
    "vertical": "",
    # Для стандартного списка атрибутов одежды: в JSON только те ключи, которые реально применимы к типу товара (без рукава у трусов и т.п.)
    "dynamic_clothing_attributes": True,
    # Искать на фото надписи/принты (см. inscription_mode). Если false — текст не извлекается.
    "extract_inscriptions": True,
    # Дедуп перед vision: off | url (первый URL) | phash (dHash файла в кэше; разные URL, тот же файл).
    "process_unique_pictures_mode": "off",
    # separate_call — отдельный vision-запрос (параллельно атрибутам); same_prompt — надписи в том же JSON, что атрибуты.
    "inscription_mode": "separate_call",
    # Для separate_call: пусто = та же модель, что основная; иначе — отдельная (прогревается перед запуском).
    "inscription_model": "",
    # None — извлекать все шаблонные ключи одежды; [] — ни одного шаблонного; иначе список ключей.
    "clothing_standard_keys_enabled": None,
    # True — не передавать название товара из фида в vision-промпт (аптеки, этикетки: меньше копирования title в атрибуты).
    "omit_offer_title_in_prompt": False,
    # Список XML-тегов/param-имён (из фида), из которых брать URL картинок.
    # Пустой список [] = использовать все найденные URL (поведение по умолчанию).
    # Примеры: ["picture"], ["param:picture", "photo"].
    "picture_attr_filter": [],
    # Режим работы с несколькими картинками одного оффера:
    #   "first_only"  — только первая картинка (классический режим)
    #   "best_select" — небольшая модель выбирает лучшую картинку, затем анализ по ней
    #   "all_images"  — все картинки передаются модели одним запросом
    "multi_image_mode": "first_only",
    # Порядковые номера картинок (1-based) которые брать из тега. Пустой список [] = все.
    # Например [1, 3] — только первая и третья картинка в рамках выбранного тега.
    "picture_index_filter": [],
}
PROJECT_KEYS = {
    "name",
    "feed_path",
    "output_dir",
    "confidence_threshold",
    "directions",
    "vertical",
    "dynamic_clothing_attributes",
    "extract_inscriptions",
    "process_unique_pictures_mode",
    "inscription_mode",
    "inscription_model",
    "clothing_standard_keys_enabled",
    "omit_offer_title_in_prompt",
    "picture_attr_filter",
    "multi_image_mode",
    "picture_index_filter",
}

# Устаревшие ключи из старых config.json — не подмешиваются в инференс (промпты только из Запуска / пресетов).
_LEGACY_PROMPT_KEYS = frozenset(
    {"task_instruction", "task_constraints", "task_examples", "task_target_attribute"}
)


def strip_legacy_prompt_from_config(config: dict) -> dict:
    """Убрать из конфига устаревшие поля промпта (они задаются только на вкладке «Запуск»)."""
    c = dict(config)
    for k in _LEGACY_PROMPT_KEYS:
        c.pop(k, None)
    return c

# Вертикаль "Одежда" — стандартное извлечение атрибутов без задания; для остальных задание подставляется в промпт.
VERTICAL_CLOTHING = "Одежда"
VERTICAL_OTHER = "Другое"
VERTICAL_DEFAULT_CHOICES = ["Одежда", "Ювелирные изделия", "Авто", "Другое"]

# Пользовательские вертикали (при выборе «Другое» и вводе названия сохраняются сюда).
CUSTOM_VERTICALS_PATH = Path(__file__).parent / "custom_verticals.json"


def load_custom_verticals() -> list[str]:
    """Список добавленных пользователем вертикалей (для выпадающего списка)."""
    if not CUSTOM_VERTICALS_PATH.exists():
        return []
    try:
        with open(CUSTOM_VERTICALS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("verticals") or [])
    except Exception:
        return []


def get_vertical_choices() -> list[str]:
    """Варианты для выпадающего списка: базовые + сохранённые пользователем."""
    custom = [v for v in load_custom_verticals() if v and v.strip() and v not in VERTICAL_DEFAULT_CHOICES]
    return list(VERTICAL_DEFAULT_CHOICES) + sorted(set(custom))


def add_custom_vertical(name: str) -> None:
    """Добавить вертикаль в список и сохранить в custom_verticals.json."""
    n = (name or "").strip()
    if not n or n in VERTICAL_DEFAULT_CHOICES:
        return
    current = load_custom_verticals()
    if n in current:
        return
    current.append(n)
    CUSTOM_VERTICALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_VERTICALS_PATH, "w", encoding="utf-8") as f:
        json.dump({"verticals": current}, f, ensure_ascii=False, indent=2)

# Типовые задания (шаблоны) — общий файл, не привязан к проекту.
TASK_TEMPLATES_PATH = Path(__file__).parent / "task_templates.json"
# Последнее состояние полей на вкладке «Запуск» (не в проекте).
RUN_PROMPT_LAST_PATH = Path(__file__).parent / "run_prompt_last.json"


def pending_run_path(project_name: str) -> Path:
    """Очередь офферов для продолжения после «Паузы» (тот же лимит/категории)."""
    return project_dir(project_name) / "pending_run.json"


def run_batch_fingerprint(
    project: str,
    cats: list[str] | None,
    limit_n: int,
    all_feed: bool,
    force_reprocess: bool,
    unique_pictures_mode: str = "",
    inscription_mode: str = "",
) -> dict:
    return {
        "project": (project or "").strip(),
        "cats": sorted([c for c in (cats or []) if c]) or None,
        "limit_n": int(limit_n) if limit_n else 0,
        "all_feed": bool(all_feed),
        "force_reprocess": bool(force_reprocess),
        "unique_pictures_mode": (unique_pictures_mode or "").strip(),
        "inscription_mode": (inscription_mode or "").strip(),
    }


def save_pending_run(project_name: str, fingerprint: dict, offer_ids: list[str]) -> None:
    n = (project_name or "").strip()
    if not n:
        return
    path = pending_run_path(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "offer_ids": [str(x).strip() for x in (offer_ids or []) if str(x).strip()],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_pending_run(project_name: str) -> dict | None:
    n = (project_name or "").strip()
    if not n:
        return None
    path = pending_run_path(n)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def clear_pending_run(project_name: str) -> None:
    n = (project_name or "").strip()
    if not n:
        return
    path = pending_run_path(n)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def list_projects() -> list[str]:
    """Return sorted list of existing project names."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        p.name for p in PROJECTS_DIR.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )


def project_dir(name: str) -> Path:
    return PROJECTS_DIR / name


LAST_LORA_FILENAME = "last_lora_adapter.txt"


def get_last_lora_path(project_name: str) -> str | None:
    """Путь к последнему сохранённому LoRA-адаптеру по проекту (для экспорта/продолжения)."""
    if not project_name:
        return None
    p = project_dir(project_name) / LAST_LORA_FILENAME
    if not p.exists():
        return None
    try:
        path = p.read_text(encoding="utf-8").strip()
        return path if path else None
    except Exception:
        return None


def save_last_lora_path(project_name: str, adapter_path: str) -> None:
    """Сохранить путь к адаптеру после успешного обучения (переживает перезапуск)."""
    if not project_name or not adapter_path:
        return
    try:
        p = project_dir(project_name) / LAST_LORA_FILENAME
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(adapter_path.strip(), encoding="utf-8")
    except Exception:
        pass


def _adapter_base_slug(base_model_name_or_path: str) -> str:
    """Нормализованный слаг базы для сравнения (последняя часть пути, нижний регистр)."""
    if not base_model_name_or_path or not isinstance(base_model_name_or_path, str):
        return ""
    s = base_model_name_or_path.strip().split("/")[-1].lower()
    return s


def get_last_lora_path_for_base(project_name: str, base_model: str) -> str | None:
    """Путь к последнему адаптеру, обученному под выбранную базовую модель (по adapter_config.json).
    Если под эту модель адаптеров нет — возвращает None (нужно обучать с нуля).
    """
    if not project_name or not base_model:
        return get_last_lora_path(project_name)
    root = project_dir(project_name)
    if not root.exists():
        return None
    target_slug = _adapter_base_slug(base_model)
    # 35B в форме часто означает 35B-A3B на HF
    if "35b" in target_slug and "a3b" not in target_slug:
        target_slug = target_slug.replace("35b-instruct", "35b-a3b").replace("35b", "35b-a3b")
    candidates = []
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith("lora_out"):
            continue
        adapter_dir = d / "lora_adapter"
        if not adapter_dir.is_dir():
            continue
        cfg_path = adapter_dir / "adapter_config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            base_in_cfg = cfg.get("base_model_name_or_path") or cfg.get("base_model") or ""
            slug = _adapter_base_slug(base_in_cfg)
            if slug and (slug == target_slug or target_slug in slug or slug in target_slug):
                mtime = (adapter_dir / "adapter_config.json").stat().st_mtime
                candidates.append((mtime, str(adapter_dir)))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _default_clothing_attributes_template() -> list:
    """Атрибуты направления clothing из эталона (для подмешивания новых ключей в старые проекты)."""
    for d in DEFAULT_DIRECTIONS:
        if d.get("id") == "clothing":
            return copy.deepcopy(d.get("attributes") or [])
    return []


def _merge_clothing_attributes(saved_attrs: list, default_attrs: list) -> list:
    """
    Сохранить порядок и правки пользователя по известным ключам; добавить новые шаблонные ключи из default
    (например color / color_shade), кастомные атрибуты в конце.
    """
    default_attrs = default_attrs or []
    saved_attrs = saved_attrs or []
    default_by_key = {a.get("key"): a for a in default_attrs if a.get("key")}
    default_order = [a.get("key") for a in default_attrs if a.get("key")]
    saved_by_key = {a.get("key"): a for a in saved_attrs if a.get("key")}
    out: list = []
    for k in default_order:
        out.append(saved_by_key.get(k) or default_by_key[k])
    for a in saved_attrs:
        k = a.get("key")
        if k and k not in default_by_key:
            out.append(a)
    return out


def _merge_directions(saved: list) -> list:
    """Merge saved directions with defaults (by id); ensure at least default two."""
    by_id = {d["id"]: copy.deepcopy(d) for d in DEFAULT_DIRECTIONS}
    clothing_template = _default_clothing_attributes_template()
    for d in saved or []:
        did = d.get("id") or d.get("name", "").lower().replace(" ", "_")
        base = by_id.get(
            did,
            {"id": did, "name": d.get("name", did), "text_enabled": False, "attributes": [], "custom_prompt": ""},
        )
        merged = {**base, **d}
        if did == "clothing" and "attributes" in d:
            merged["attributes"] = _merge_clothing_attributes(d.get("attributes") or [], clothing_template)
        by_id[did] = merged
    return list(by_id.values())


def load_project(name: str) -> dict:
    """Load project config (без модели/Ollama — они глобальные). Raise ValueError if not found."""
    cfg_path = project_dir(name) / "config.json"
    if not cfg_path.exists():
        raise ValueError(f"Project '{name}' not found")
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)
    merged = {**DEFAULT_CONFIG, "name": name}
    for key in PROJECT_KEYS:
        if key == "name":
            continue
        if key in data:
            merged[key] = data[key]
    if "directions" in data:
        merged["directions"] = _merge_directions(data["directions"])
    for lk in _LEGACY_PROMPT_KEYS:
        merged.pop(lk, None)
    if "process_unique_pictures_mode" not in data and data.get("process_unique_pictures_only"):
        merged["process_unique_pictures_mode"] = "url"
    # Универсально для аптечных и др. вертикалей: в старых config не было ключа — не тащить title в vision.
    if isinstance(data, dict) and "omit_offer_title_in_prompt" not in data:
        v = (merged.get("vertical") or "").strip()
        if v and v != VERTICAL_CLOTHING:
            merged["omit_offer_title_in_prompt"] = True
    return merged


def save_project(config: dict) -> None:
    """Save project config to disk. Сохраняются только поля проекта (не модель/Ollama)."""
    name = config.get("name", "").strip()
    if not name:
        raise ValueError("Project name cannot be empty")
    pdir = project_dir(name)
    pdir.mkdir(parents=True, exist_ok=True)
    cfg_path = pdir / "config.json"
    to_save = {k: config[k] for k in PROJECT_KEYS if k in config}
    to_save["name"] = name
    for lk in _LEGACY_PROMPT_KEYS:
        to_save.pop(lk, None)
    if not (to_save.get("directions")):
        to_save["directions"] = DEFAULT_DIRECTIONS
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)


def create_project(name: str, **kwargs) -> dict:
    """Create a new project with default config; raise if already exists."""
    name = name.strip()
    if not name:
        raise ValueError("Project name cannot be empty")
    if (project_dir(name) / "config.json").exists():
        raise ValueError(f"Project '{name}' already exists")
    config = {**DEFAULT_CONFIG, "name": name, **kwargs}
    save_project(config)
    return config


def get_or_create_project(name: str) -> dict:
    """Load project if exists, create with defaults if not."""
    try:
        return load_project(name)
    except ValueError:
        return create_project(name)


def results_db_path(name: str) -> Path:
    return project_dir(name) / "results.db"


def run_state_path(name: str) -> Path:
    """Файл состояния запуска (лог, прогресс). Обновление страницы не прерывает процесс."""
    return project_dir(name) / "run_state.json"


def cache_db_path(name: str) -> Path:
    return project_dir(name) / "cache.db"


def image_cache_dir(name: str) -> Path:
    """Папка для кэша картинок по URL (projects/<name>/image_cache/)."""
    return project_dir(name) / "image_cache"


def corrections_path(name: str) -> Path:
    return project_dir(name) / "corrections.json"


def finetune_queue_path(name: str) -> Path:
    """Очередь offer_id для добавления в датасет дообучения из вкладки «Результаты»."""
    return project_dir(name) / "fine_tune_dataset" / "queued_offer_ids.json"


def load_finetune_queue_offer_ids(name: str) -> list[str]:
    p = finetune_queue_path(name)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        raw = data.get("ids") or data.get("offer_ids") or []
        return [str(x).strip() for x in raw if str(x).strip()]
    except Exception:
        return []


def append_finetune_queue_offer_ids(name: str, new_ids: list[str]) -> tuple[int, int]:
    """(сколько новых добавлено, всего в очереди)."""
    existing = list(dict.fromkeys(load_finetune_queue_offer_ids(name)))
    seen = set(existing)
    added = 0
    for oid in new_ids:
        s = str(oid).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        existing.append(s)
        added += 1
    p = finetune_queue_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"ids": existing}, f, ensure_ascii=False, indent=2)
    return added, len(existing)


def clear_finetune_queue_offer_ids(name: str) -> None:
    p = finetune_queue_path(name)
    if p.exists():
        p.unlink()


def prompt_presets_path(name: str) -> Path:
    """Пресеты промпта вкладки «Запуск» — только для этого проекта."""
    return project_dir(name) / "prompt_presets.json"


def load_project_prompt_presets(name: str) -> list[dict]:
    if not (name or "").strip():
        return []
    path = prompt_presets_path(name.strip())
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    out: list[dict] = []
    for t in data.get("presets") or []:
        if not isinstance(t, dict):
            continue
        out.append(
            {
                "id": (t.get("id") or "").strip(),
                "name": (t.get("name") or "").strip() or "Без названия",
                "instruction": (t.get("instruction") or "").strip(),
                "task_constraints": (t.get("task_constraints") or "").strip(),
                "task_examples": (t.get("task_examples") or "").strip(),
                "task_target_attribute": (t.get("task_target_attribute") or "").strip(),
                "full_prompt_text": (t.get("full_prompt_text") or ""),
                "use_full_prompt_edit": bool(t.get("use_full_prompt_edit")),
                "dynamic_clothing_attributes": bool(t.get("dynamic_clothing_attributes", True)),
            }
        )
    return out


def save_project_prompt_presets(name: str, presets: list[dict]) -> None:
    n = (name or "").strip()
    if not n:
        raise ValueError("Project name is empty")
    pdir = project_dir(n)
    pdir.mkdir(parents=True, exist_ok=True)
    path = prompt_presets_path(n)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"presets": presets}, f, ensure_ascii=False, indent=2)


def add_project_prompt_preset(
    name: str,
    preset_name: str,
    instruction: str = "",
    task_constraints: str = "",
    task_examples: str = "",
    task_target_attribute: str = "",
    full_prompt_text: str = "",
    use_full_prompt_edit: bool = False,
    dynamic_clothing_attributes: bool = True,
) -> dict:
    """Добавить пресет в проект (имя уникально в рамках списка — при совпадении перезаписываем по имени)."""
    import uuid

    n = (name or "").strip()
    if not n:
        raise ValueError("Project name is empty")
    pname = (preset_name or "").strip() or "Без названия"
    presets = load_project_prompt_presets(n)
    new_entry = {
        "id": str(uuid.uuid4()),
        "name": pname,
        "instruction": (instruction or "").strip(),
        "task_constraints": (task_constraints or "").strip(),
        "task_examples": (task_examples or "").strip(),
        "task_target_attribute": (task_target_attribute or "").strip(),
        "full_prompt_text": full_prompt_text or "",
        "use_full_prompt_edit": bool(use_full_prompt_edit),
        "dynamic_clothing_attributes": bool(dynamic_clothing_attributes),
    }
    replaced = False
    for i, p in enumerate(presets):
        if (p.get("name") or "").strip() == pname:
            new_entry["id"] = p.get("id") or new_entry["id"]
            presets[i] = new_entry
            replaced = True
            break
    if not replaced:
        presets.append(new_entry)
    save_project_prompt_presets(n, presets)
    return new_entry


def delete_project_prompt_preset(project_name: str, template_id: str) -> bool:
    tid = (template_id or "").strip()
    if not tid:
        return False
    presets = load_project_prompt_presets(project_name)
    new = [x for x in presets if x.get("id") != tid]
    if len(new) == len(presets):
        return False
    save_project_prompt_presets(project_name, new)
    return True


def correction_offer_ids(name: str) -> set[str]:
    """Offer ID, для которых есть правки (чтобы не дублировать «авто-хорошие» примерами из results)."""
    out: set[str] = set()
    for c in load_corrections(name):
        oid = str(c.get("offer_id") or "").strip()
        if oid:
            out.add(oid)
    return out


def load_corrections(name: str) -> list[dict]:
    path = corrections_path(name)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_correction(name: str, correction: dict) -> None:
    """Append or update one correction entry in the project's corrections.json."""
    corrections = load_corrections(name)
    oid = correction.get("offer_id")
    existing_ids = {c.get("offer_id") for c in corrections}
    if oid in existing_ids:
        corrections = [correction if c.get("offer_id") == oid else c for c in corrections]
    else:
        corrections.append(correction)
    path = corrections_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)


def get_all_attribute_definitions(config: dict) -> list[dict]:
    """Return flat list of all attribute definitions from all directions (key, label, options)."""
    out = []
    seen = set()
    for d in config.get("directions", []):
        for a in d.get("attributes", []):
            key = a.get("key", "")
            if key and key not in seen:
                seen.add(key)
                out.append(a)
    return out


# ── Типовые задания (шаблоны) ─────────────────────────────────────────────────

def load_task_templates() -> list[dict]:
    """Список шаблонов заданий (глобально, не в проекте): id, name, instruction, task_constraints, task_examples, task_target_attribute."""
    if not TASK_TEMPLATES_PATH.exists():
        return []
    try:
        with open(TASK_TEMPLATES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for t in data.get("templates") or []:
            if not isinstance(t, dict):
                continue
            out.append(
                {
                    "id": t.get("id", ""),
                    "name": (t.get("name") or "").strip() or "Без названия",
                    "instruction": (t.get("instruction") or "").strip(),
                    "task_constraints": (t.get("task_constraints") or "").strip(),
                    "task_examples": (t.get("task_examples") or "").strip(),
                    "task_target_attribute": (t.get("task_target_attribute") or "").strip(),
                }
            )
        return out
    except Exception:
        return []


def save_task_templates(templates: list[dict]) -> None:
    """Сохранить список шаблонов в task_templates.json."""
    TASK_TEMPLATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TASK_TEMPLATES_PATH, "w", encoding="utf-8") as f:
        json.dump({"templates": templates}, f, ensure_ascii=False, indent=2)


def add_task_template(
    name: str,
    instruction: str,
    task_constraints: str = "",
    task_examples: str = "",
    task_target_attribute: str = "",
) -> dict:
    """Добавить шаблон; вернуть созданный объект с id."""
    import uuid
    templates = load_task_templates()
    t = {
        "id": str(uuid.uuid4()),
        "name": (name or "").strip() or "Без названия",
        "instruction": (instruction or "").strip(),
        "task_constraints": (task_constraints or "").strip(),
        "task_examples": (task_examples or "").strip(),
        "task_target_attribute": (task_target_attribute or "").strip(),
    }
    templates.append(t)
    save_task_templates(templates)
    return t


def upsert_task_template_full(template: dict) -> None:
    """Сохранить или обновить шаблон по id (внутренний формат как add_task_template)."""
    tid = (template or {}).get("id")
    if not tid:
        return
    templates = [x for x in load_task_templates() if x.get("id") != tid]
    templates.append(
        {
            "id": tid,
            "name": (template.get("name") or "").strip() or "Без названия",
            "instruction": (template.get("instruction") or "").strip(),
            "task_constraints": (template.get("task_constraints") or "").strip(),
            "task_examples": (template.get("task_examples") or "").strip(),
            "task_target_attribute": (template.get("task_target_attribute") or "").strip(),
        }
    )
    save_task_templates(templates)


def delete_task_template(template_id: str) -> bool:
    """Удалить шаблон по id. Вернуть True если удалён."""
    before = load_task_templates()
    templates = [x for x in before if x.get("id") != template_id]
    if len(templates) == len(before):
        return False
    save_task_templates(templates)
    return True


def get_task_template_instruction(template_id: str) -> str:
    """Текст инструкции шаблона по id."""
    for t in load_task_templates():
        if t.get("id") == template_id:
            return (t.get("instruction") or "").strip()
    return ""


def get_task_template_by_id(template_id: str) -> dict | None:
    for t in load_task_templates():
        if t.get("id") == template_id:
            return dict(t)
    return None


def load_run_prompt_last() -> dict:
    """Последние значения полей промпта с вкладки «Запуск»."""
    if not RUN_PROMPT_LAST_PATH.exists():
        return {}
    try:
        with open(RUN_PROMPT_LAST_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_run_prompt_last(
    task_instruction: str = "",
    task_constraints: str = "",
    task_examples: str = "",
    task_target_attribute: str = "",
    use_full_prompt_edit: bool = False,
    full_prompt_text: str = "",
    task_target_attributes: list[str] | None = None,
) -> None:
    RUN_PROMPT_LAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    attrs: list[str] = []
    if task_target_attributes is not None:
        attrs = [x.strip() for x in task_target_attributes if (x or "").strip()]
    if not attrs and (task_target_attribute or "").strip():
        attrs = [x.strip() for x in (task_target_attribute or "").split("\n") if x.strip()]
    legacy_line = (task_target_attribute or "").strip() or ("\n".join(attrs) if attrs else "")
    payload = {
        "task_instruction": (task_instruction or "").strip(),
        "task_constraints": (task_constraints or "").strip(),
        "task_examples": (task_examples or "").strip(),
        "task_target_attribute": legacy_line,
        "task_target_attributes": attrs,
        "use_full_prompt_edit": bool(use_full_prompt_edit),
        "full_prompt_text": (full_prompt_text or ""),
    }
    with open(RUN_PROMPT_LAST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
