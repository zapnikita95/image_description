#!/usr/bin/env python3
"""
Multi-direction image analysis via Ollama vision.

- Text detection: один общий запрос «надписи на изображении» (если включён в любом направлении).
- Атрибуты по направлениям: для каждого направления — свой промпт по списку атрибутов из конфига;
  модель может возвращать и дополнительные поля (например length, print_pattern).
- Текст и все направления атрибутов запускаются параллельно.
"""

import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
import time as _time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

import project_manager as pm
from ollama_pool_trace import pool_trace_headers

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

DEFAULT_MODEL = "qwen3.5:35b"
# По умолчанию пул на 11435 (см. project_manager.GLOBAL_DEFAULTS и Desktop/ollama-queue-proxy).
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11435"
MAX_IMAGE_SIZE = 1024  # px, longest side

_PROFILE_LOCK = threading.Lock()


def image_analysis_profiling_enabled() -> bool:
    """Включить замеры: env IMAGE_DESC_PROFILE=1 (или true/yes/on)."""
    v = (os.environ.get("IMAGE_DESC_PROFILE") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


_VISION_SKIP_ATTR_KEYS_CLOTHING = frozenset({"original_name"})


def _filter_vision_attributes(attrs: list | None, direction_id: str | None = None) -> list:
    """Для направления clothing не запрашиваем original_name — дублирует название из фида."""
    if not attrs:
        return []
    if (direction_id or "").strip() != "clothing":
        return list(attrs)
    return [a for a in attrs if (a.get("key") or "").strip().lower() not in _VISION_SKIP_ATTR_KEYS_CLOTHING]


def _vision_profile_append(calls: list | None, entry: dict) -> None:
    # Только None отключает запись; пустой list[] — валидный буфер (иначе append никогда не срабатывал).
    if calls is None:
        return
    with _PROFILE_LOCK:
        calls.append(entry)


# Кэш загруженного локального адаптера (путь -> (model, processor))
_local_adapter_cache: dict[str, tuple] = {}
_local_adapter_lock = threading.Lock()


def _is_adapter_path(model_or_path: str) -> bool:
    """True если значение — путь к папке с LoRA-адаптером (для инференса через Unsloth без Ollama)."""
    if not model_or_path or not isinstance(model_or_path, str):
        return False
    s = model_or_path.strip()
    if s.startswith("adapter:"):
        s = s[8:].strip()
    p = Path(s)
    if not p.is_dir():
        return False
    return (p / "config.json").exists() or (p / "adapter_config.json").exists() or (p / "processor_config.json").exists()


def _resolve_adapter_path(model_or_path: str) -> Path | None:
    """Возвращает Path к адаптеру или None."""
    if not _is_adapter_path(model_or_path):
        return None
    s = model_or_path.strip()
    if s.startswith("adapter:"):
        s = s[8:].strip()
    return Path(s).resolve()


def _find_merged_dir(adapter_path: Path) -> Path | None:
    """Ищет merged-модель рядом с адаптером (../gguf_out/merged или ../merged)."""
    candidates = [
        adapter_path.parent.parent / "gguf_out" / "merged",
        adapter_path.parent / "merged",
        adapter_path.parent.parent / "merged",
    ]
    for c in candidates:
        if c.is_dir() and (c / "config.json").exists():
            return c.resolve()
    return None


def _load_local_adapter(adapter_path: Path):
    """Загружает модель по пути. Кэширует. Сначала проверяет merged-версию (нет скачивания),
    потом пробует загрузить адаптер через Unsloth."""
    key = str(adapter_path.resolve())
    with _local_adapter_lock:
        if key in _local_adapter_cache:
            return _local_adapter_cache[key]
    # Если есть слитая модель рядом — грузим через transformers напрямую, без скачивания
    merged_dir = _find_merged_dir(adapter_path)
    if merged_dir:
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            processor = AutoProcessor.from_pretrained(str(merged_dir), trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                str(merged_dir),
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            if not torch.cuda.is_available():
                model = model.to(device)
            model.eval()
            with _local_adapter_lock:
                _local_adapter_cache[key] = (model, processor)
            return (model, processor)
        except Exception:
            pass  # fallback to unsloth below
    # Fallback: грузим адаптер через Unsloth (требует CUDA и базовую модель в кэше)
    try:
        try:
            import unsloth  # noqa: F401
        except Exception:
            pass
        try:
            from unsloth import FastVisionModel
            model, processor = FastVisionModel.from_pretrained(
                model_name=key,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,
            )
            try:
                FastVisionModel.for_inference(model)
            except Exception:
                pass
            tok = processor
        except Exception:
            from unsloth import FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(
                model_name=key,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,
            )
            try:
                FastLanguageModel.for_inference(model)
            except Exception:
                pass
            processor = None
        with _local_adapter_lock:
            _local_adapter_cache[key] = (model, tok)
        return (model, tok)
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить модель по пути {adapter_path}: {e}") from e


def _local_adapter_chat(
    prompt: str,
    image_b64: str | None,
    adapter_path: Path,
    system: str | None = None,
    max_new_tokens: int = 512,
    vision_profile_calls: list | None = None,
    vision_profile_task: str = "local_adapter",
) -> str:
    """Один запрос к локальному адаптеру (Unsloth). Возвращает сырой текст ответа."""
    t_wall = _time.perf_counter()
    model, proc_or_tok = _load_local_adapter(adapter_path)
    t_after_load = _time.perf_counter()
    tokenizer = getattr(proc_or_tok, "tokenizer", proc_or_tok)
    if not _PIL_AVAILABLE and image_b64:
        _vision_profile_append(
            vision_profile_calls,
            {
                "task": vision_profile_task,
                "backend": "local",
                "ms": round((_time.perf_counter() - t_wall) * 1000, 2),
                "note": "no_pil",
            },
        )
        return '{"error": "PIL required for local adapter with image"}'
    image = None
    if image_b64:
        try:
            raw = base64.b64decode(image_b64)
            image = PILImage.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            _vision_profile_append(
                vision_profile_calls,
                {
                    "task": vision_profile_task,
                    "backend": "local",
                    "ms": round((_time.perf_counter() - t_wall) * 1000, 2),
                    "note": "bad_image",
                },
            )
            return f'{{"error": "Bad image: {e}"}}'
    # Чат: image + text только для VL (у processor есть image_processor), иначе только text
    has_vision = getattr(proc_or_tok, "image_processor", None) is not None
    content = []
    if image is not None and has_vision:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt or ""})
    messages = [{"role": "user", "content": content}]
    if system and system.strip():
        messages.insert(0, {"role": "system", "content": system.strip()})
    try:
        text = proc_or_tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        text = prompt
        if system and system.strip():
            text = f"{system.strip()}\n\n{text}"
    # Токенизация: VL — processor(images=..., text=...), text-only — tokenizer(text=...)
    try:
        if image is not None and getattr(proc_or_tok, "image_processor", None) is not None:
            inputs = proc_or_tok(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = tokenizer(text, return_tensors="pt", padding=True)
    except (TypeError, Exception):
        try:
            inputs = proc_or_tok(images=[image], text=text, return_tensors="pt")
        except Exception:
            inputs = tokenizer(text, return_tensors="pt", padding=True)
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: (v.to(model.device).to(model_dtype) if v.is_floating_point() else v.to(model.device))
        if hasattr(v, "to") and hasattr(v, "is_floating_point")
        else (v.to(model.device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }
    t_inputs_ready = _time.perf_counter()
    _vision_profile_append(
        vision_profile_calls,
        {
            "task": f"{vision_profile_task}:inputs",
            "backend": "local",
            "ms": round((t_inputs_ready - t_after_load) * 1000, 2),
        },
    )
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
    import torch
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=pad_id,
        )
    t_gen_done = _time.perf_counter()
    _vision_profile_append(
        vision_profile_calls,
        {
            "task": f"{vision_profile_task}:generate",
            "backend": "local",
            "ms": round((t_gen_done - t_inputs_ready) * 1000, 2),
        },
    )
    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return decoded.strip()


# ── Image utilities ───────────────────────────────────────────────────────────

def _resize_image_bytes(data: bytes, max_size: int = MAX_IMAGE_SIZE) -> bytes:
    if not _PIL_AVAILABLE:
        return data
    try:
        img = PILImage.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if max(w, h) <= max_size:
            return data
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return data


def _image_cache_path(cache_dir: Path | None, url: str) -> Path | None:
    if not cache_dir:
        return None
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    return Path(cache_dir) / f"{key}.jpg"


def _url_to_base64(
    url: str,
    max_size: int = MAX_IMAGE_SIZE,
    timeout: int = 30,
    cache_dir: Path | str | None = None,
) -> str | None:
    if not url:
        return None
    # Local file: file:// or path that exists
    if url.startswith("file://"):
        path = Path(url[7:].lstrip("/"))
    elif Path(url).exists():
        path = Path(url)
    else:
        path = None
    if path is not None and path.is_file():
        return _path_to_base64(path, max_size)
    cache_path = _image_cache_path(Path(cache_dir) if cache_dir else None, url) if url else None
    if cache_path and cache_path.exists():
        return _path_to_base64(cache_path, max_size)
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = _resize_image_bytes(r.content, max_size)
        b64 = base64.b64encode(data).decode("ascii")
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(data)
            except Exception:
                pass
        return b64
    except Exception as e:
        print(f"  [Картинка] Не удалось загрузить по ссылке: {e}", file=sys.stderr)
        return None


def ensure_image_cached(
    url: str,
    cache_dir: Path | str,
    max_size: int = MAX_IMAGE_SIZE,
    timeout: int = 30,
) -> Path | None:
    """Скачать картинку по URL в кэш (если ещё нет) и вернуть путь к файлу. Для отображения в UI."""
    if not url:
        return None
    path = _image_cache_path(Path(cache_dir), url)
    if path and path.exists():
        return path
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = _resize_image_bytes(r.content, max_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
    except Exception as e:
        print(f"  [Кэш] Не удалось сохранить картинку: {e}", file=sys.stderr)
        return None


def _path_to_base64(path: str | Path, max_size: int = MAX_IMAGE_SIZE) -> str | None:
    try:
        data = Path(path).read_bytes()
        data = _resize_image_bytes(data, max_size)
        return base64.b64encode(data).decode("ascii")
    except Exception as e:
        print(f"  [Картинка] Не удалось прочитать файл: {e}", file=sys.stderr)
        return None


# ── Ollama call ───────────────────────────────────────────────────────────────

def normalize_ollama_url(url: str) -> str:
    """Use 127.0.0.1 instead of localhost to avoid IPv6 connection issues."""
    if not url or not url.strip():
        return url or ""
    parsed = urlparse(url.strip())
    if parsed.hostname and parsed.hostname.lower() == "localhost":
        netloc = "127.0.0.1"
        if parsed.port:
            netloc += f":{parsed.port}"
        parsed = parsed._replace(netloc=netloc)
        return urlunparse(parsed)
    return url.strip()


def ollama_root_health_timeout_s(ollama_url: str, *, default_s: float = 3.0, pool_proxy_s: float = 12.0) -> float:
    """Таймаут для GET {base}/ (проверка «Ollama доступен»). Прокси-пул на :11435 часто отвечает >3s — иначе ложный красный баннер."""
    base = normalize_ollama_url(ollama_url or "").strip()
    if not base:
        return default_s
    try:
        if urlparse(base.rstrip("/")).port == 11435:
            return pool_proxy_s
    except Exception:
        pass
    return default_s


def ollama_list_models(ollama_url: str, timeout: int = 5) -> list[str]:
    """GET /api/tags — все модели, установленные в Ollama (по имени). Для выпадающего списка «Модель»."""
    base = normalize_ollama_url(ollama_url or "").rstrip("/")
    try:
        r = requests.get(f"{base}/api/tags", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        models = data.get("models") or []
        names = []
        for m in models:
            n = m.get("name") or m.get("model")
            if n and n not in names:
                names.append(n)
        return sorted(names)
    except Exception:
        return []


def ollama_loaded_models(ollama_url: str, timeout: int = 5) -> list[dict]:
    """GET /api/ps — список моделей, сейчас загруженных в память Ollama. Каждый элемент: name, size_vram (bytes), expires_at."""
    base = normalize_ollama_url(ollama_url or "").rstrip("/")
    try:
        r = requests.get(f"{base}/api/ps", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        models = data.get("models") or []
        # Нормализуем имя: в ответе может быть "name" или "model"
        for m in models:
            if "name" not in m and "model" in m:
                m["name"] = m["model"]
        return models
    except Exception:
        return []


def ollama_unload_model(ollama_url: str, model_name: str, timeout: int = 30) -> tuple[bool, str]:
    """Выгрузить модель из памяти Ollama (keep_alive=0). Возвращает (успех, сообщение)."""
    base = normalize_ollama_url(ollama_url or "").rstrip("/")
    try:
        r = requests.post(
            f"{base}/api/generate",
            json={"model": model_name, "prompt": "", "keep_alive": 0},
            timeout=timeout,
            headers=pool_trace_headers(
                ollama_url, project="image_description", label=f"unload {model_name}"[:120]
            ),
        )
        r.raise_for_status()
        return True, f"Модель **{model_name}** выгружена из памяти."
    except requests.RequestException as e:
        err = getattr(e, "response", None)
        text = err.text[:200] if err and hasattr(err, "text") else str(e)
        return False, f"Ошибка: {text}"


def _ollama_error_message(e: Exception, response_text: str | None = None) -> str:
    """Человекочитаемое сообщение об ошибке Ollama."""
    # Сначала проверяем тело ответа — оно информативнее, чем строка исключения
    if response_text:
        try:
            body = json.loads(response_text)
            api_err = body.get("error", "")
            if api_err:
                if "model failed to load" in api_err or "resource" in api_err.lower():
                    return (
                        f"Модель не загрузилась: {api_err}. "
                        "Возможно, не хватает VRAM или архитектура не поддерживается (Qwen3-VL в старых Ollama). "
                        "Попробуйте clothes-detector-v1 или обновите Ollama."
                    )
                return f"Ollama: {api_err}"
        except Exception:
            pass
    err = str(e).strip()
    # Connection refused / недоступен — только если это реально сетевая ошибка (нет response body)
    is_conn_err = (
        "Connection refused" in err
        or "ConnectionError" in err
        or "Failed to establish" in err
        or "Max retries exceeded" in err
        or ("11434" in err and not response_text)
    )
    if is_conn_err:
        return "Ollama недоступен (проверь URL: пул :11435 или прямой :11434). Запустите Ollama и при необходимости прокси."
    if "Connection" in err or "Pool" in err or "refused" in err.lower():
        return f"Нет связи с Ollama: {err[:120]}"
    return err[:200]


def _ollama_use_think_false(model: str) -> bool:
    """
    Ollama для Qwen3 умеет отключать reasoning через **верхнеуровневое** поле think=false в /api/chat
    (не в options — иначе на части версий игнорируется).
    См. https://ollama.com/blog/thinking и issue ollama про generate vs chat.
    """
    m = (model or "").lower()
    return "qwen3" in m


def _ollama_chat(
    prompt: str,
    image_b64: str | None,
    model: str,
    ollama_url: str,
    timeout: int = 120,
    system: str | None = None,
    vision_profile_calls: list | None = None,
    vision_profile_task: str = "ollama",
    extra_images: list[str] | None = None,
) -> str:
    # Native Ollama API: "content" is the text, "images" is array of base64 (not content array)
    t_wall = _time.perf_counter()

    # Дополнительно к API think=false (см. payload ниже): жёсткий system для моделей 9b/35b.
    if "9b" in model or "35b" in model:
        if not system:
            system = "CRITICAL: You MUST respond directly in the requested JSON format. Do NOT use thinking mode. Do NOT show reasoning. Output ONLY the JSON response immediately."
        else:
            # Добавляем к существующему системному промпту
            system = "CRITICAL: Do NOT use thinking mode. Output ONLY JSON directly.\n\n" + system.strip()

    messages = []
    if system and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    msg = {"role": "user", "content": prompt or ""}
    # Собираем список всех изображений (основное + дополнительные для all_images)
    all_images: list[str] = []
    if image_b64:
        all_images.append(image_b64)
    if extra_images:
        all_images.extend(b for b in extra_images if b)
    if all_images:
        msg["images"] = all_images
    messages.append(msg)
    has_vision = bool(all_images)
    # 9b/35b: thinking + JSON; для vision num_predict увеличен (см. ниже), иначе обрезка → пустой парс.
    if "9b" in model or "35b" in model:
        # thinking + vision JSON: 3000 часто обрезает ответ → пустой парс и 0% уверенности
        num_predict = 8192 if has_vision else 4096
    elif has_vision:
        # Vision + JSON: 512 часто обрезает ответ у 2b/4b/7b → пустой/битый JSON и 0% уверенности.
        num_predict = 1536
    else:
        # Текст без картинки: надписи, батч-перевод value — длинный JSON-массив не должен обрезаться.
        num_predict = 2048
    
    options = {
        "num_predict": num_predict,
    }
    
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if _ollama_use_think_false(model):
        payload["think"] = False
    base = normalize_ollama_url(ollama_url or "").rstrip("/")
    url = f"{base}/api/chat"
    _trace = pool_trace_headers(
        ollama_url or base,
        project="image_description",
        label=(vision_profile_task or "api_chat")[:200],
    )
    last_e = None
    last_response_text = None
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=timeout, headers=_trace)
            r.raise_for_status()
            _vision_profile_append(
                vision_profile_calls,
                {
                    "task": vision_profile_task,
                    "backend": "ollama",
                    "ms": round((_time.perf_counter() - t_wall) * 1000, 2),
                },
            )
            return (r.json().get("message") or {}).get("content") or ""
        except requests.RequestException as e:
            last_e = e
            if hasattr(e, "response") and e.response is not None:
                try:
                    last_response_text = e.response.text
                except Exception:
                    pass
            if attempt < 2:
                _time.sleep(2)
    _vision_profile_append(
        vision_profile_calls,
        {
            "task": vision_profile_task,
            "backend": "ollama",
            "ms": round((_time.perf_counter() - t_wall) * 1000, 2),
            "failed": True,
        },
    )
    raise RuntimeError(_ollama_error_message(last_e, last_response_text)) from last_e


_ATTRIBUTE_VALUE_LATIN_RE = re.compile(r"[A-Za-z]{3,}")


def attribute_value_needs_llm_translate(value: str) -> bool:
    """True, если в строке value после глоссария осталась «заметная» латиница (отдельный LLM-перевод)."""
    return bool(value and _ATTRIBUTE_VALUE_LATIN_RE.search(value))


def _extract_json_array_from_text(text: str) -> list | None:
    """Первый сбалансированный JSON-массив из ответа модели (учёт кавычек и escape)."""
    if not text or not str(text).strip():
        return None
    t = str(text).strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t).strip()
    start = t.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    quote_char = ""
    for i in range(start, len(t)):
        ch = t[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
                quote_char = ""
            continue
        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    arr = json.loads(t[start : i + 1])
                    return arr if isinstance(arr, list) else None
                except json.JSONDecodeError:
                    return None
    return None


def _batch_translate_attribute_values_llm(
    strings: list[str],
    model_name: str,
    ollama_url: str,
    timeout: int,
    vision_profile_calls: list | None,
) -> list[str] | None:
    """Один запрос: массив строк → массив переводов той же длины. При ошибке парсинга — None."""
    if not strings:
        return []
    payload = json.dumps(strings, ensure_ascii=False)
    system = (
        "Ты переводишь короткие значения атрибутов для русскоязычной карточки товара. "
        "Отвечай ТОЛЬКО JSON-массивом строк той же длины, без markdown, без пояснений до или после."
    )
    user = (
        "Переведи каждый элемент массива на русский: нейтрально и кратко, как в интернет-магазине (одежда, обувь, аксессуары). "
        "Если элемент уже по-русски — верни его почти без изменений (только опечатки). "
        "Не добавляй пояснений. Сохрани порядок и ровно такое же число элементов.\n\n"
        f"{payload}"
    )
    adapter = _resolve_adapter_path(model_name)
    try:
        if adapter is not None:
            raw = _local_adapter_chat(
                user,
                None,
                adapter,
                system=system,
                max_new_tokens=2048,
                vision_profile_calls=vision_profile_calls,
                vision_profile_task="attr_value_translate",
            )
        else:
            raw = _ollama_chat(
                user,
                None,
                model_name,
                ollama_url,
                timeout,
                system=system,
                vision_profile_calls=vision_profile_calls,
                vision_profile_task="attr_value_translate",
            )
    except Exception:
        return None
    arr = _extract_json_array_from_text(raw)
    if not isinstance(arr, list) or len(arr) != len(strings):
        return None
    out: list[str] = []
    for x in arr:
        out.append(str(x).strip() if x is not None else "")
    return out


def apply_llm_translate_remaining_latin_inplace(
    direction_results: dict,
    *,
    enabled: bool,
    translate_model: str,
    vision_model: str,
    ollama_url: str,
    timeout: int,
    vision_profile_calls: list | None = None,
) -> None:
    """
    После глоссария: для value с латиницей (≥3 буквы подряд) — батч через Ollama/локальный адаптер.
    translate_model пустой → используется vision_model.
    """
    if not enabled or not direction_results:
        return
    model_use = (translate_model or "").strip() or (vision_model or "").strip()
    if not model_use:
        return

    refs: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    for did, dr in direction_results.items():
        if not isinstance(dr, dict) or dr.get("error"):
            continue
        did_s = str(did)
        for ak, ent in dr.items():
            if ak == "error" or not isinstance(ent, dict):
                continue
            val = ent.get("value")
            if not isinstance(val, str) or not val.strip():
                continue
            if not attribute_value_needs_llm_translate(val):
                continue
            if val not in refs:
                refs[val] = []
                order.append(val)
            refs[val].append((did_s, str(ak)))

    if not order:
        return

    MAX_BATCH = 48
    MAX_CHARS = 12000
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for s in order:
        add = len(s) + 3
        if cur and (len(cur) >= MAX_BATCH or cur_len + add > MAX_CHARS):
            batches.append(cur)
            cur = []
            cur_len = 0
        cur.append(s)
        cur_len += add
    if cur:
        batches.append(cur)

    translations: dict[str, str] = {}
    per_batch_timeout = max(30, min(int(timeout), 180))
    for batch in batches:
        tr = _batch_translate_attribute_values_llm(
            batch, model_use, ollama_url, per_batch_timeout, vision_profile_calls
        )
        if tr is None:
            continue
        for orig, newv in zip(batch, tr):
            translations[orig] = newv if newv else orig

    for orig, pairs in refs.items():
        newv = translations.get(orig)
        if newv is None or newv == orig:
            continue
        for did_s, ak in pairs:
            dr = direction_results.get(did_s)
            if isinstance(dr, dict):
                ent = dr.get(ak)
                if isinstance(ent, dict):
                    ent["value"] = newv


def _extract_json(text: str) -> dict:
    """Извлекает один JSON-объект из текста (первый сбалансированный {...})."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = re.sub(r"^[.\s]+", "", text)
    text = text.strip()
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _strip_model_reasoning_noise(text: str) -> str:
    """
    Qwen3 / др. перед JSON часто вставляют блоки рассуждений — мешают поиску и съедают num_predict.
    """
    if not (text or "").strip():
        return text or ""
    t = text
    # Ограждения ```thinking``` / ```reasoning``` (часто у VL в Ollama). Общий ```…``` не трогаем — там может быть JSON.
    t = re.sub(r"```(?:thinking|reasoning|analysis)[\s\S]*?```", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<think>[\s\S]*?</think>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^[\s\W]*(?:here is|below is|json:?)\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def _collect_balanced_json_dicts(text: str) -> list[dict]:
    """Все успешно распарсенные объекты {...} слева направо."""
    text = (text or "").strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = re.sub(r"^[.\s]+", "", text)
    out: list[dict] = []
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            out.append(obj)
                    except json.JSONDecodeError:
                        pass
                    pos = i + 1
                    break
        else:
            break
    return out


def _parse_vision_json_response(raw: str, expected_keys: list[str]) -> dict:
    """
    Слияние всех JSON-объектов в ответе; если ожидаемых ключей нет — берём последний объект,
    где они есть (типично: сначала мусор/черновик, в конце итоговый JSON).
    """
    text = _strip_model_reasoning_noise(raw or "")
    objs = _collect_balanced_json_dicts(text)
    if not objs:
        return {}
    merged: dict = {}
    for o in objs:
        merged.update(o)
    exp = {k for k in (expected_keys or []) if k}
    if exp and not (set(merged.keys()) & exp):
        for o in reversed(objs):
            if set(o.keys()) & exp:
                return dict(o)
    return merged


def _extract_and_merge_all_json(text: str) -> dict:
    """
    Извлекает все сбалансированные {...} из ответа и сливает в один dict.
    Дообученные модели иногда выдают несколько объектов подряд — нужен полный список атрибутов.
    """
    objs = _collect_balanced_json_dicts(text)
    merged = {}
    for o in objs:
        merged.update(o)
    return merged


def _parse_fallback_attributes(raw: str, attribute_keys: list[str]) -> dict:
    """
    Дообученные модели могут возвращать не JSON, а текст. Пытаемся вытащить пары ключ-значение
    по ожидаемым ключам атрибутов. Возвращаем dict как из _extract_json: { key: value или {value, confidence} }.
    """
    if not raw or not attribute_keys:
        return {}
    text = raw.strip()
    result = {}
    for key in attribute_keys:
        if not key:
            continue
        # Паттерны: "sleeve_length": "short"  или  sleeve_length: short  или  "sleeve_length":"short"
        key_esc = re.escape(key)
        for pattern in [
            rf'["\']?{key_esc}["\']?\s*:\s*["\']([^"\']+)["\']',
            rf'["\']?{key_esc}["\']?\s*:\s*(\w+)',
            rf'["\']?{key_esc}["\']?\s*=\s*["\']?([^"\',\s}}]+)',
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                result[key] = {"value": m.group(1).strip(), "confidence": 75}
                break
    return result


# ── Text detection ─────────────────────────────────────────────────────────────

def _text_prompt(product_name: str | None = None) -> str:
    base = (
        "На фото может быть несколько предметов одежды. "
        "Внимательно рассмотри изображение и прочитай надписи, принты, текст."
    )
    if product_name and product_name.strip():
        base = (
            f"На фото модель может быть в полном образе (несколько предметов). "
            f"Нас интересует ТОЛЬКО товар из названия: «{product_name.strip()}». "
            f"Определи, где на изображении этот предмет (обычно один конкретный — топ, блузка, брюки и т.д.), "
            f"и прочитай надписи/принты ТОЛЬКО на нём. Остальную одежду не описывай."
        )
    return (
        f"{base}\n\n"
        "Источник истины — изображение: переноси в ответ только текст, который реально виден на предмете; "
        "не выдумывай надписи из названия товара, если их нет на фото.\n\n"
        "Язык надписи: на российском рынке чаще всего **русский или английский**. "
        "Если слово логичнее читать как кириллицу (типичные русские буквы, окончания) — фиксируй **кириллицей**. "
        "Если это очевидно латинское слово/бренд — оставь латиницей. "
        "Не «улучшай» текст английскими заменами там, где на фото кириллица.\n\n"
        "Ответь строго в формате JSON (без markdown, без пояснений):\n"
        "{\n"
        '  "text_found": true или false,\n'
        '  "texts": ["текст 1", "текст 2"],\n'
        '  "confidence": число от 0 до 100\n'
        "}\n\n"
        "Если текста нет — texts пустой список, confidence — уверенность. Только JSON."
    )


def _inline_inscription_json_suffix() -> str:
    """Добавляется к промпту атрибутов: те же поля, что в отдельном запросе, но в одном JSON."""
    return (
        "\n\n=== ТЕКСТ НА ИЗДЕЛИИ (в том же корневом JSON) ===\n"
        "Добавь в тот же объект JSON поля (рядом с атрибутами):\n"
        '\"text_found\": true или false,\n'
        '\"texts\": [\"точный видимый текст на товаре\", ...],\n'
        '\"text_read_confidence\": число от 0 до 100 (уверенность в прочтении текста).\n'
        "Если надписей нет — text_found: false, texts: [].\n"
        "Источник — только фото; не копируй текст из названия товара, если его нет на изделии.\n"
        "Язык: кириллица vs латиница — см. правила в общем задании по надписям (русский каталог: не подменяй кириллицу английскими «аналогами»)."
    )


def _pop_inline_inscription_from_parsed(parsed: dict | None, raw: str) -> dict | None:
    """Достаёт поля надписей из ответа; возвращает словарь как у detect_text или None."""
    if not parsed:
        return None
    tf = bool(parsed.pop("text_found", False))
    txs = parsed.pop("texts", None)
    if txs is None:
        txs = []
    if not isinstance(txs, list):
        txs = [str(txs)] if txs else []
    txs = [str(t).strip() for t in txs if str(t).strip()]
    trc_raw = parsed.pop("text_read_confidence", None)
    try:
        trc = int(trc_raw) if trc_raw is not None else 0
    except (TypeError, ValueError):
        trc = 0
    return {
        "text_found": tf or bool(txs),
        "texts": txs,
        "confidence": max(0, min(100, trc)),
        "raw": raw,
        "error": None,
    }


def resolve_inscription_model(config: dict, main_model: str) -> str:
    """Модель для отдельного запроса надписей; пустая настройка → основная модель."""
    m = (config.get("inscription_model") or "").strip()
    return m if m else (main_model or DEFAULT_MODEL)


def inscription_mode_is_same_prompt(config: dict) -> bool:
    mode = (config.get("inscription_mode") or "separate_call").strip().lower()
    if mode in ("same_prompt", "same", "inline", "one_call"):
        return True
    # Старый баг Gradio Radio: в конфиг попала русская подпись второй опции.
    return "тот же" in mode or "один запрос" in mode


def detect_text(
    image_b64: str | None,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
    product_name: str | None = None,
    adapter_path: Path | None = None,
    vision_profile_calls: list | None = None,
    vision_profile_task: str = "text",
    extra_images_b64: list[str] | None = None,
) -> dict:
    """Returns {text_found, texts, confidence, raw, error}."""
    try:
        prompt = _text_prompt(product_name)
        if adapter_path is not None:
            raw = _local_adapter_chat(
                prompt,
                image_b64,
                adapter_path,
                system=None,
                vision_profile_calls=vision_profile_calls,
                vision_profile_task=vision_profile_task,
            )
        else:
            raw = _ollama_chat(
                prompt,
                image_b64,
                model,
                ollama_url,
                timeout,
                vision_profile_calls=vision_profile_calls,
                vision_profile_task=vision_profile_task,
                extra_images=extra_images_b64,
            )
        parsed = _extract_json(_strip_model_reasoning_noise(raw or ""))
        return {
            "text_found": bool(parsed.get("text_found", False)),
            "texts": parsed.get("texts") or [],
            "confidence": int(parsed.get("confidence", 0)),
            "raw": raw,
            "error": None,
        }
    except Exception as e:
        return {"text_found": False, "texts": [], "confidence": 0, "raw": "", "error": str(e)}


# ── Attributes by direction (config-driven, allow extra keys) ─────────────────

def _filter_dirs_attrs_by_keys(directions: list, target_keys: set[str]) -> list:
    """Оставляет в направлениях только атрибуты с ключами из target_keys."""
    import copy

    out = []
    for d in directions:
        nd = copy.deepcopy(d)
        attrs = nd.get("attributes") or []
        nd["attributes"] = [a for a in attrs if (a.get("key") or "").strip() in target_keys]
        out.append(nd)
    return out


def parse_target_attribute_lines_to_keys(lines: list[str]) -> list[str]:
    """
    Строки с вкладки «Запуск»: «Цвет металла (metal_color)» или «metal_color» → ключи JSON.
    """
    keys: list[str] = []
    for line in lines or []:
        line = (line or "").strip()
        if not line:
            continue
        m = re.search(r"\(([^)]+)\)\s*$", line)
        if m:
            k = m.group(1).strip()
            if k:
                keys.append(k)
            continue
        if re.match(r"^[\w.-]+$", line):
            keys.append(line)
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def infer_latin_keys_in_parentheses(text: str) -> list[str]:
    """
    Латинские идентификаторы в скобках в свободном тексте задания: «...(metal_color)».
    """
    if not (text or "").strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\(([a-z][a-z0-9_]*)\)", text, re.IGNORECASE):
        k = m.group(1).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def canonicalize_target_attribute_line(line: str) -> str:
    """
    Из свободного текста («цвет металла») делает строку с латинским ключом: «цвет металла (metal_color)».
    Уже корректная строка с `(key)` не ломает.
    """
    s = (line or "").strip()
    if not s:
        return s
    m = re.search(r"\(([a-z][a-z0-9_]*)\)\s*$", s, re.IGNORECASE)
    if m:
        return s
    keys = parse_target_attribute_lines_to_keys([s])
    key = keys[0] if len(keys) == 1 else None
    if not key:
        fb = fallback_keys_from_freeform_target_lines([s])
        key = fb[0] if len(fb) == 1 else None
    if not key:
        return s
    label = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip() or key
    return f"{label} ({key})"


def canonicalize_target_attribute_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines or []:
        c = canonicalize_target_attribute_line(line)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def fallback_keys_from_freeform_target_lines(lines: list[str]) -> list[str]:
    """
    Строка без скобок «(key)», напр. «цвет металла» — даёт латинский ключ JSON, чтобы парсер
    не отбрасывал ответ модели (частая ошибка: только кириллица в поле «атрибут»).
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines or []:
        line = (raw or "").strip()
        if not line or re.search(r"\([^)]+\)\s*$", line):
            continue
        low = line.lower()
        k = None
        if any(
            p in low
            for p in (
                "цвет металла",
                "цветметалла",
                "металл",
                "metal color",
                "metal colour",
                "metal_color",
            )
        ) or ("metal" in low and "color" in low):
            k = "metal_color"
        elif any(p in low for p in ("камен", "вставк", "gem", "кристалл")):
            k = "gem_related"
        elif any(p in low for p in ("размер", "size", "габарит")):
            k = "size"
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def resolve_attributes_for_prompt(
    base_attrs: list[dict],
    task_target_list: list[str],
    custom_prompt: str,
    full_prompt_text: str,
) -> list[dict]:
    """
    Если в направлении нет атрибутов в конфиге, но задан промпт (задание / полный промпт),
    восстанавливаем ожидаемые ключи JSON из строк «Запуск» или из текста — иначе ответ модели
    отбрасывается целиком (типичный кейс: вертикаль «не одежда», убрано clothing, остался other с []).
    """
    attrs = list(base_attrs or [])
    if attrs:
        return attrs
    blob = "\n".join(
        x
        for x in (
            "\n".join(task_target_list or []),
            custom_prompt or "",
            full_prompt_text or "",
        )
        if (x or "").strip()
    )
    if not blob.strip():
        return attrs
    keys = parse_target_attribute_lines_to_keys(task_target_list or [])
    if not keys:
        keys = parse_target_attribute_lines_to_keys(blob.splitlines())
    if not keys:
        keys = infer_latin_keys_in_parentheses(blob)
    if not keys:
        keys = fallback_keys_from_freeform_target_lines(task_target_list or [])
    if not keys:
        keys = fallback_keys_from_freeform_target_lines(blob.splitlines())
    if keys:
        return [{"key": k, "label": k, "options": []} for k in keys]
    return attrs


def filter_clothing_standard_attributes_for_extraction(attrs: list[dict], config: dict) -> list[dict]:
    """
    Вертикаль «Одежда», направление clothing: убрать шаблонные ключи (из DEFAULT_DIRECTIONS),
    не отмеченные в clothing_standard_keys_enabled. Свои ключи из JSON направлений не трогаем.
    clothing_standard_keys_enabled is None — без изменений (все шаблонные, как в directions).
    """
    enabled = config.get("clothing_standard_keys_enabled")
    if enabled is None:
        return list(attrs or [])
    import project_manager as _pm

    std = frozenset(_pm.default_clothing_standard_keys())
    enabled_set = {str(x).strip() for x in (enabled or []) if str(x).strip()}
    out: list[dict] = []
    for a in attrs or []:
        k = (a.get("key") or "").strip()
        if not k:
            continue
        if k in std:
            if k in enabled_set:
                out.append(a)
        else:
            out.append(a)
    return out


def _hoist_nested_attribute_json(parsed: dict, expected_keys: list[str]) -> dict:
    """
    Модели часто заворачивают атрибуты: {"result": {"metal_color": {...}}}.
    Поднимаем внутренний объект, если на верхнем уровне нет ни одного ожидаемого ключа.
    """
    if not isinstance(parsed, dict) or not expected_keys:
        return parsed
    exp = {k for k in expected_keys if k}
    if not exp:
        return parsed
    top = set(parsed.keys()) - {"value", "confidence", "error", "raw"}
    if top & exp:
        return parsed
    for w in ("result", "data", "response", "output", "attributes", "answer", "items"):
        inner = parsed.get(w)
        if isinstance(inner, dict) and set(inner.keys()) & exp:
            return dict(inner)
    for _k, v in parsed.items():
        if isinstance(v, dict) and set(v.keys()) & exp:
            return dict(v)
    return parsed


def _verbatim_from_image_mode(
    task: str,
    user_constraints: str,
    user_examples: str,
    required_json_keys: list[str] | None,
    target_lines: list[str] | None,
) -> bool:
    """
    True — нужен дословный текст/код с объекта (любой алфавит), без жёсткого «value только по-русски».
    Срабатывает по ключам атрибутов или по формулировке задания; **не** привязано к вертикали (аптека/одежда/авто).
    """
    keys = [(x or "").strip().lower() for x in (required_json_keys or []) if (x or "").strip()]
    for k in keys:
        if "original_name" in k or k in ("trade_name", "brand_name") or k.endswith("_trade_name"):
            return True
        if k in ("article", "mpn", "asin"):
            return True
        if any(
            s in k
            for s in (
                "sku",
                "ean",
                "gtin",
                "upc",
                "isbn",
                "oem",
                "vin",
                "serial",
                "articul",
                "artikul",
                "part_no",
                "part_number",
                "model_code",
                "label_text",
                "marking",
                "stamp",
            )
        ):
            return True

    for line in target_lines or []:
        low = (line or "").lower()
        if "original_name" in low or "trade_name" in low:
            return True
        if any(
            x in low
            for x in (
                "артикул",
                "штрихкод",
                "vin",
                "oem",
                "серийн",
            )
        ):
            return True

    blob = "\n".join(
        x.strip() for x in (task, user_constraints, user_examples) if (x or "").strip()
    ).lower()
    packaging_or_label = (
        "упаков",
        "блистер",
        "препарат",
        "торговое название",
        "с упаковки",
        "на упаковке",
        "самую крупную",
        "самая крупная",
        "крупнейш",
        "крупную надпись",
        "крупная надпись",
        "самую большую",
        "самая большая",
        "большую надпись",
    )
    any_domain_literal = (
        "дословн",
        "выписать",
        "переписать",
        "как на фото",
        "с фотограф",
        "с изображен",
        "с картин",
        "ocr",
        "маркиров",
        "артикул",
        "part number",
        "номер детал",
        "номер запчаст",
        "штрихкод",
        "barcode",
        "с бирк",
        "на бирк",
        "шильдик",
        "табличк",
        "гравиров",
        " vin",
        "vin ",
        "vin:",
    )
    if any(t in blob for t in packaging_or_label):
        return True
    if any(t in blob for t in any_domain_literal):
        return True
    if re.search(r"\bvin\b", blob, re.IGNORECASE):
        return True
    if "этикет" in blob and any(
        x in blob for x in ("крупн", "назван", "надпис", "бренд", "строк", "текст", "латин")
    ):
        return True
    return False


def compose_task_prompt_blocks(
    task: str,
    *,
    vertical: str | None = None,
    direction_name: str = "",
    product_name: str | None = None,
    user_constraints: str = "",
    user_examples: str = "",
    target_attribute: str = "",
    target_attributes: list[str] | None = None,
    required_json_keys: list[str] | None = None,
) -> str:
    """
    Собирает полный текст задания для vision-модели: короткая «ЗАДАЧА» пользователя
    + шаблонные блоки (источник истины — фото, формат JSON, язык значений, уверенность)
    + опциональные ограничения и примеры от пользователя.
    """
    task = (task or "").strip()
    if not task:
        return ""

    parts: list[str] = []

    v = (vertical or "").strip()
    if v and v != "Одежда":
        parts.append(f"Вертикаль / контекст партнёра: {v}.")
    if (direction_name or "").strip():
        parts.append(f"Направление анализа: {direction_name.strip()}.")

    parts.append("=== ЗАДАЧА (формулировка пользователя) ===")
    parts.append(task)

    parts.append("=== ПРИОРИТЕТ ФОРМУЛИРОВКИ ЗАДАЧИ ===")
    parts.append(
        "Текст в блоке «ЗАДАЧА» задан пользователем и имеет **наивысший приоритет**: "
        "если там указано, **какой именно** текст или признак с фото извлечь (например только **самая крупная** надпись на упаковке, только бренд, "
        "только верхняя строка этикетки, артикул на детали, VIN/номер на шильдике, размер на бирке) — выполняй это **буквально**. "
        "Общие шаблонные абзацы ниже **не отменяют** и **не расширяют** смысл задания: не подменяй узкую инструкцию «общими» правилами."
    )

    tas: list[str] = []
    if target_attributes:
        tas.extend([x.strip() for x in target_attributes if (x or "").strip()])
    ta0 = (target_attribute or "").strip()
    if ta0:
        for line in ta0.split("\n"):
            line = line.strip()
            if line and line not in tas:
                tas.append(line)

    rjk_for_mode = [x.strip() for x in (required_json_keys or []) if (x or "").strip()]
    verbatim_mode = _verbatim_from_image_mode(
        task,
        user_constraints or "",
        user_examples or "",
        rjk_for_mode or None,
        tas or None,
    )

    if tas:
        parts.append("=== ИЗВЛЕКАЕМЫЕ АТРИБУТЫ ===")
        for i, ta in enumerate(tas, 1):
            if verbatim_mode:
                parts.append(
                    f"{i}. {ta} — сфокусируйся на этом признаке по фото. В JSON используй ключ атрибута из настроек направления проекта "
                    "(латиницей, как в конфиге; если в скобках указан ключ — он приоритетен). "
                    "Строковое **value** — **дословно главная крупная строка текста с объекта** (упаковка, этикетка, бирка, шильдик, деталь), как на фото "
                    "(кириллица или **латиница в том же виде**, что на изображении; **не переводи** дословную маркировку в «удобное» русское описание). "
                    "**Не** вставляй сюда весь мелкий текст с той же поверхности; для полного текста есть отдельный запрос про надписи — сюда только строка из задания."
                )
            else:
                parts.append(
                    f"{i}. {ta} — сфокусируйся на этом признаке по фото. В JSON используй ключ атрибута из настроек направления проекта "
                    "(латиницей, как в конфиге; если в скобках указан ключ — он приоритетен). "
                    "Строковое **value**: для **классовых** признаков (цвет, материал, фасон, тип рисунка) — коротко **по-русски**, нейтрально для карточки. "
                    "Если атрибут — **код, артикул, бренд или текст с бирки/принта** — переноси **дословно**, **в той же раскладке**, что на фото (латиница/кириллица/цифры). "
                    "Длинный многострочный текст с той же этикетки не дублируй сюда целиком, если для него в проекте есть отдельный запрос «надписи»."
                )

    if product_name and product_name.strip():
        parts.append("=== ФОКУС НА ТОВАРЕ ===")
        parts.append(
            f"На изображении может быть несколько предметов. Нужен только товар из оффера: «{product_name.strip()}». "
            "Найди этот предмет на фото и отвечай только по нему; фон, людей и чужие предметы не описывай, если это не нужно для задачи."
        )
        pl = product_name.strip().lower()
        if any(
            x in pl
            for x in (
                "набор",
                "комплект",
                " пар",
                "пар ",
                "пар)",
                "пар,",
                "пар в",
                "штук",
                " шт",
                "×",
                " x",
            )
        ):
            parts.append(
                "**Набор или несколько единиц в одном оффере:** если на фото видно **несколько предметов этого товара** "
                "разных цветов (например несколько пар носков), в полях цвета перечисли **все** различимые цвета **через запятую**, "
                "не ограничивайся одним–двумя оттенками одной гаммы."
            )

    parts.append("=== ИСТОЧНИК ИСТИНЫ ===")
    parts.append(
        "Главный источник правды — изображение. Суди по тому, что реально видно на фото. "
        "Название товара и текст с фида — только чтобы понять, какой предмет на фото в фокусе; "
        "значения атрибутов (цвет, материал и т.д.) не выводи из названия — только из того, что видно. "
        "**Если атрибут — дословное название / текст с упаковки, блистера, этикетки:** переписывай **только то, что читаешь на фото** "
        "(OCR по изображению). **Не копируй и не слегка перефразируй** маркетинговое название из фида, даже если оно почти совпадает с тем, что на упаковке — "
        "сверяй буквы и цифры именно с фото. "
        "**Запрещено** подставлять в value **общие описания** вместо запрошенной маркировки: например «коробка с лекарством», «упаковка товара», «изделие на фото», "
        "«товар без надписи» — если нужна конкретная строка с объекта, укажи **читаемые слова/цифры с фото** или оставь пусто с низкой уверенностью. "
        "**Не вставляй** в value слова и цифры из названия/описания оффера, которых **нет** на изображении (не галлюцинируй объём «200 мл», «300 мл» и т.п., если такого крупно не видно). "
        "Если признак на фото не виден или нет оснований — value можно оставить пустым с confidence 0; "
        "если по фото есть слабая, но лучшая догадка — укажи её с низкой уверенностью (не из названия товара)."
    )

    parts.append("=== ОГРАНИЧЕНИЯ (системные) ===")
    parts.append(
        "Не придумывай детали, которых нет на фото. Не добавляй в JSON ключи, которых нет в списке атрибутов задания. "
        "Не заполняй значения из названия или описания оффера, если их нельзя подтвердить по изображению "
        "(для текста с упаковки — только визуально прочитанное). "
        "Не распространяйся о фоне и людях без необходимости для задачи."
    )

    uc = (user_constraints or "").strip()
    if uc:
        parts.append("=== ОГРАНИЧЕНИЯ (от пользователя) ===")
        parts.append(uc)

    ex = (user_examples or "").strip()
    if ex:
        parts.append("=== ПРИМЕРЫ И УТОЧНЕНИЯ (от пользователя) ===")
        parts.append(
            "Ниже — пояснения пользователя (не факты из базы, а как интерпретировать задачу). Учти их при ответе."
        )
        parts.append(ex)

    rjk = [x.strip() for x in (required_json_keys or []) if (x or "").strip()]
    if rjk:
        parts.append("=== ИМЕНА ПОЛЕЙ В JSON (обязательно) ===")
        parts.append(
            "Используй в корне ответа **ровно эти ключи** (латиница, как в проекте): "
            + ", ".join(f"`{k}`" for k in rjk)
            + ". Не переводи их на русский и не заменяй синонимами."
        )

    parts.append("=== ФОРМАТ ОТВЕТА ===")
    if verbatim_mode:
        parts.append(
            "Ответь строго одним JSON-объектом, без markdown-ограждений и без текста до или после JSON. "
            "Имена ключей в JSON — **только как в настройках проекта** (как в поле key у атрибутов, обычно латиницей); не переводи ключи и не придумывай свои имена полей. "
            "Строковые **значения** (поле value) для дословной маркировки/названия с объекта — **повторяй доминирующую крупную строку с фото** в том же написании, что на изображении "
            "(**латиница или кириллица как на фото**; **не подменяй** латиницу «удобным» русским переводом). "
            "**Не запихивай** в value весь абзац мелкого текста с той же поверхности: только **одна** главная строка, как в блоке «ЗАДАЧА»; полный текст — в отдельном запросе про надписи, сюда не дублируй. "
            "**ОБЯЗАТЕЛЬНО укажи уверенность от 0 до 100 для каждого атрибута.** Формат: для каждого ключа объект вида {\"value\": \"...\", \"confidence\": число_от_0_до_100}. "
            "Если уверенность не указана — будет подставлено 75% по умолчанию, что нежелательно. "
            "Оценивай уверенность честно: если признак виден чётко — 85-95%; если виден, но не идеально — 70-84%; если сомневаешься — 50-69%; если почти не видно — 30-49%; если не видно — 0-29%. "
            "Если для какого-то пункта данных недостаточно оснований на фото — отрази это в формулировке значения и в низкой уверенности."
        )
    else:
        parts.append(
            "Ответь строго одним JSON-объектом, без markdown-ограждений и без текста до или после JSON. "
            "Имена ключей в JSON — **только как в настройках проекта** (как в поле key у атрибутов, обычно латиницей); не переводи ключи и не придумывай свои имена полей. "
            "Строковые **значения** (поле value): для **категориальных** признаков (цвет, материал, фасон, тип рисунка, посадка) — **коротко по-русски**, нейтрально для карточки "
            "(например: круглый воротник, хлопок, унисекс, графический принт). "
            "Если задание или атрибут требуют **текст, код или маркировку с объекта** — пиши **как на фото**, **без** принудительного перевода на русский (латиница/кириллица/цифры). "
            "**Не подставляй** длинное название из оффера вместо того, что видно на изображении. "
            "Для **чисто классовых** полей не заполняй value целым абзацем мелкого текста с этикетки — оставь короткую характеристику; полный текст этикетки — в запросе про надписи, если он есть в проекте. "
            "**ОБЯЗАТЕЛЬНО укажи уверенность от 0 до 100 для каждого атрибута.** Формат: для каждого ключа объект вида {\"value\": \"...\", \"confidence\": число_от_0_до_100}. "
            "Если уверенность не указана — будет подставлено 75% по умолчанию, что нежелательно. "
            "Оценивай уверенность честно: если признак виден чётко — 85-95%; если виден, но не идеально — 70-84%; если сомневаешься — 50-69%; если почти не видно — 30-49%; если не видно — 0-29%. "
            "Если для какого-то пункта данных недостаточно оснований на фото — отрази это в формулировке значения и в низкой уверенности."
        )

    return "\n\n".join(parts)


def _build_attributes_prompt(
    direction_name: str,
    attributes: list[dict],
    custom_prompt: str = "",
    product_name: str | None = None,
    vertical: str | None = None,
    dynamic_subset: bool = False,
    task_constraints: str = "",
    task_examples: str = "",
    task_target_attribute: str = "",
    task_target_attributes: list[str] | None = None,
) -> str:
    """Build prompt from config attributes; allow model to add more keys."""
    # Для вертикали "Одежда" префикс не добавляем — стандартное извлечение атрибутов одежды.
    v = (vertical or "").strip()
    prefix = (f"Вертикаль: {v}. Учти специфику сферы партнёра. " if v and v != "Одежда" else "")

    focus = ""
    if product_name and product_name.strip():
        focus = (
            f"На фото может быть несколько предметов. Нас интересует ТОЛЬКО товар: «{product_name.strip()}». "
            "Определи, где этот предмет на изображении, и опиши только его характеристики (остальное на фото игнорируй). "
        )
    if custom_prompt and custom_prompt.strip():
        ta_list = list(task_target_attributes or [])
        if not ta_list and (task_target_attribute or "").strip():
            ta_list = [x.strip() for x in (task_target_attribute or "").split("\n") if x.strip()]
        resolved_keys = [a.get("key", "").strip() for a in attributes if a.get("key")]
        return compose_task_prompt_blocks(
            custom_prompt.strip(),
            vertical=vertical,
            direction_name=direction_name,
            product_name=product_name,
            user_constraints=task_constraints or "",
            user_examples=task_examples or "",
            target_attribute="",
            target_attributes=ta_list if ta_list else None,
            required_json_keys=resolved_keys or None,
        )
    if not attributes:
        return ""

    keys_list = [a.get("key", "").strip() for a in attributes if a.get("key")]
    keys_str = ", ".join(keys_list)
    g_ex = pm.load_attribute_glossary()
    example_parts = []
    for a in attributes:
        key = a.get("key", "").strip()
        opts = a.get("options", [])
        raw_opt = str(opts[0]) if opts else "short"
        val = pm.translate_attribute_value(raw_opt, g_ex)
        val_j = json.dumps(val, ensure_ascii=False)
        example_parts.append(f'"{key}":{{"value":{val_j},"confidence":80}}')
    example_line = "{" + ",".join(example_parts) + "}"
    example_minimal = '{"sleeve_length":{"value":"короткий","confidence":85},"material":{"value":"хлопок","confidence":80}}'

    color_hint = ""
    if any((a.get("key") or "").strip() in ("color", "color_shade") for a in attributes):
        color_hint = (
            "\n**Цвет (`color`, `color_shade`):** описывай цвета **именно того товара**, который в названии оффера "
            "(если название задаёт предмет — не цвет фона и не чужие вещи на фото).\n"
            "- **Одно изделие одного цвета** — один базовый цвет в `color`; при необходимости нюанс в `color_shade`.\n"
            "- **Набор / комплект / несколько пар или штук** в названии **и на фото видно несколько предметов этого товара разных цветов** "
            "(например набор носков столбиком) — перечисли **все** различимые цвета **через запятую** в `color` "
            "(в порядке сверху вниз или слева направо). **Не останавливайся на одном–двух** оттенках одной гаммы, если дальше по фото "
            "явно другие цвета (белый, розовый, бежевый и т.д.).\n"
            "- В `color_shade` можно дать **список через запятую в том же порядке**, что и в `color`, либо **опусти** ключ, "
            "если достаточно базовых названий в `color`.\n"
        )

    no_unknown_hint = (
        "\n**Запрещено** в любых value: «неизвестно», «неизвестный», «неузнаваемый», «известняк», "
        "«невозможно определить», «не могу определить» и любые похожие отмазки — **не пиши такое**; "
        "вместо этого **полностью убери ключ** из JSON или оставь только осмысленное значение.\n"
    )

    collar_hint = ""
    if any((a.get("key") or "").strip() == "collar" for a in attributes):
        collar_hint = (
            "\n**Воротник (`collar`):** **mandarin / стойка** — только **стоячий ворот без отворота**, плотно у шеи "
            "(рубашка-мао, некоторые платья). **Не** называй стойкой **crew** — круглое горло свитера/футболки/джемпера "
            "с обтачкой; **не** путай с **notch** — **отложной воротник рубашки или пиджака с лацканами**. "
            "Жилет/жакет **без воротника**, только обтачка горловины или V от запаха — **none** или **v-neck**, **не** «стойка». "
            "**Запрещено** выдумывать **«оборванный»** ворот: аккуратный **сырой край** (raw edge) у горловины — это **не** порванная ткань; "
            "для пальто с лацканами — **notch**.\n"
        )

    fastener_hint = ""
    if any((a.get("key") or "").strip() == "fastener" for a in attributes):
        fastener_hint = (
            "\n**Застёжка (`fastener`):** указывай тип **только если он явно виден** на фото. "
            "Если не видно — **none** или опусти ключ; **не** угадывай молнию.\n"
            "- **tie** — **завязка**: **узкая полоска той же ткани**, **завязанная узлом** или бантом (часто у жилета/халата/кардигана); "
            "**не** называй это **ремень**, если нет **кожи/плотного ремня с пряжкой**.\n"
            "- **belt** — **ремень**: отдельная **кожаная или плотная лента** с **пряжкой** (металл), тянется через шлёвки.\n"
            "- **sash** — **пояс**: **широкая декоративная лента** на талии (часто из ткани платья), может завязываться, **без** типичной ременной пряжки.\n"
            "- **drawstring** — **шнурок** в **кулиске** (стягивает капюшон/пояс).\n"
            "- **buttons** — **швейные пуговицы**; **snaps** — **пресс-кнопки** (не путать с пуговицами).\n"
            "- **buckle** — только **металлическая дужка/рамка пряжки** (ремня). **Крупные круглые плоские швейные пуговицы** на планке — это **пуговицы**, **не** пряжка.\n"
            "- **Ряд круглых пришивных элементов** по центру у кардигана/жакета/пальто — почти всегда **пуговицы** или **пресс-кнопки**; **не** называй это **ремнём**, если нет **отдельной ленты через шлёвки**.\n"
            "**Формулировки:** коротко (**завязка**, **пуговицы**, **нет**). **Запрещено**: «наличие», «отсутствие», «присутствуют», «карманы — наличие».\n"
            "Пиши value **по-русски**, без латиницы вперемешку.\n"
        )

    pockets_hint = ""
    if any((a.get("key") or "").strip() == "pockets" for a in attributes):
        pockets_hint = (
            "\n**Карманы (`pockets`):** только **да** / **нет** или тип (**накладной**, **втачной**). "
            "**Запрещено**: «присутствуют», «наличие», «отсутствие» — так не пиши.\n"
        )

    print_pattern_hint = ""
    if any((a.get("key") or "").strip() == "print_pattern" for a in attributes):
        print_pattern_hint = (
            "\n**Паттерн / принт (`print_pattern`):** только по **фото**. "
            "**Запрещено** в одном value одновременно **однотонный** и любой **узор** (клетка, полоска, горошек, гусиная лапка и т.д.) — выбери **одно**: "
            "либо реальный узор, либо однотонность **без** слова «узор».\n"
            "- **Меланж / marl / heather:** пряжа **пёстрая**, вкрапления другого цвета по всему полотну — это **не** «однотонный»; укажи **melange** / **меланж**.\n"
            "- **Букле / nubby bouclé / «барашек»:** **петельная кудрявая** фактура, **узелки и петли** по всей поверхности (платье/пиджак из твида букле) — "
            "это **не** гладкий однотон; укажи **boucle knit** / **букле** / **петельная фактура (барашек)**, не **plain**.\n"
            "- **Экомех / тедди / плюш с ворсом:** виден **ворс, «овечка», тёплый ворс** по всей поверхности — **teddy texture** / **текстура барашек**, не **однотонный** как у гладкой ткани.\n"
            "- **Трикотаж / вязка:** видна **вязальная структура** (рибана, косы) — **knit texture** / **трикотажный рисунок**, не «однотонный».\n"
            "- **Ёлочка** (**herringbone** / ёлочка): **повторяющиеся стрелки / ступеньки** из **наклонных полос** "
            "как **ломаная «V»** вдоль ткани (часто у пальто, костюмной шерсти). **Не** путай с гусиной лапкой.\n"
            "- **Гусиная лапка** (houndstooth, pied-de-poule): **ломаная «рваная» сетка** из **четырёхугольников** "
            "двух цветов, **зубчатые края** чешуек; **не** ровные параллельные стрелки ёлочки и **не** крупная клетка.\n"
            "- **Клетка / plaid:** **ровная шахматная сетка** квадратов или **перекрёстные полосы** — **checked** / **plaid**.\n"
            "- **Люрекс** — блеск **нитей в полотне**; **пайетки** — **отдельные диски** на поверхности. Не смешивай термины.\n"
            "- **Горошек:** пиши по-русски **горошек** (не оставляй **polka dots** латиницей).\n"
            "- **Фактура атласа:** при однотонном цвете можно **однотонный, атлас**; если есть узор — **без** «однотонный».\n"
            "- **Пайетки:** если покрывают полотно — **sequin pattern**; если только вставка — основной узор в `print_pattern`, пайетки в `details_decor`.\n"
            "- **Стразы** — в `details_decor`, не путать с люрексом и пайетками.\n"
        )

    details_decor_hint = ""
    if any((a.get("key") or "").strip() == "details_decor" for a in attributes):
        details_decor_hint = (
            "\n**Детали и декор (`details_decor`):** отдели **локальный декор** от **узора всего полотна**. "
            "**Пайетки** как нашивка/полоса/воротник → **sequins**; **люрекс** по всей поверхности — в `print_pattern`. "
            "**Стразы** → **rhinestones** / стразы. **Все value — целиком по-русски**, без фраз вроде «contrast panel», "
            "«inserts», «beading» — используй **контрастная вставка**, **вставки**, **бисер** и т.п.\n"
        )

    if dynamic_subset:
        return (
            f"{prefix}{focus}"
            f"Проанализируй предмет одежды/аксессуар на фото. Направление: {direction_name}.\n\n"
            "Шаг 1: Определи тип изделия (футболка, джинсы, куртка, платье, нижнее бельё, купальник, носки, обувь, сумка и т.п.).\n"
            "Шаг 2: Из РАЗРЕШЁННЫХ ключей ниже выведи только те, которые реально применимы к ЭТОМУ изделию. "
            "Если признак для такого типа не существует — полностью опусти ключ из JSON.\n"
            "Верни ОДИН JSON только с подходящими ключами. Каждое значение: "
            '{"value":"<строка на русском>","confidence":<0-100>}. '
            "**Все value — на русском.** Только JSON, без markdown и без пояснений.\n"
            "Для **длины рукава** и **длины изделия** (мини/миди): пиши **длинный / длинная** (о размере), "
            "не **долгий** — это слово про время, не про физическую длину."
            f"{color_hint}{collar_hint}{fastener_hint}{pockets_hint}{print_pattern_hint}{details_decor_hint}{no_unknown_hint}\n"
            f"Разрешённые ключи (подмножество): {keys_str}.\n"
            f"Пример (подмножество): {example_minimal}"
        )

    # Упрощённый промпт для ускорения (меньше текста = быстрее токенизация и генерация)
    return (
        f"{prefix}{focus}Одежда на фото. Направление: {direction_name}. "
        f"Верни JSON со всеми ключами: {keys_str}. "
        "Подходи к опциям из настроек или своей краткой формулировке; **все value — на русском**. Только JSON, без markdown. "
        "Для длины рукава и длины платья/юбки: **длинный/длинная** (размер), не «долгий» (о времени). "
        "Не заполняй рукав, воротник, капюшон и «длину платья» для изделий без рукавов (носки, джинсы, обувь, сумки и т.п.) — опусти эти ключи. "
        f"{color_hint}{collar_hint}{fastener_hint}{pockets_hint}{print_pattern_hint}{details_decor_hint}{no_unknown_hint}"
        f"Пример: {example_line}"
    )


def _clothing_category_tail(category: str) -> str:
    """Последний сегмент пути категории фида: «Женская / Джинсы» → «джинсы»."""
    s = (category or "").strip().lower()
    if " / " in s:
        s = s.rsplit(" / ", 1)[-1].strip()
    elif "/" in s:
        s = s.rsplit("/", 1)[-1].strip()
    return s


# Верх изделия: рукав / воротник / капюшон осмысленны.
_UPPER_BODY_CATEGORY_MARKERS: tuple[str, ...] = (
    "плать",
    "блуз",
    "рубаш",
    "футбол",
    "худи",
    "лонгслив",
    "лонг",
    "свитер",
    "джемпер",
    "кардиган",
    "жилет",
    "жакет",
    "жилет",
    "куртк",
    "пальто",
    "пиджак",
    "топ",
    "майк",
    "боди",
    "комбинезон",
    "сарафан",
    "туник",
    "поло",
    "анорак",
    "ветровк",
    "пуховик",
    "бомбер",
    "смокинг",
    "пуловер",
    "кофта",
    "свитшот",
    "костюм",
    "блузк",
    "водолазк",
)

# Длина в смысле «мини/миди/макси» — для платьев и юбок (не для джинс/носков).
_DRESS_SKIRT_LENGTH_MARKERS: tuple[str, ...] = ("плать", "юбк", "сарафан")


def category_has_upper_body_garment(tail: str) -> bool:
    t = (tail or "").strip().lower()
    return bool(t) and any(m in t for m in _UPPER_BODY_CATEGORY_MARKERS)


def category_needs_dress_skirt_length(tail: str) -> bool:
    t = (tail or "").strip().lower()
    return bool(t) and any(m in t for m in _DRESS_SKIRT_LENGTH_MARKERS)


def inapplicable_clothing_attribute_keys(category: str) -> frozenset[str]:
    """
    Ключи атрибутов направления clothing, которые не показываем для данной категории фида.
    (Джинсы/носки — без рукава; «длина юбки/платья» — только платье/юбка/сарафан.)
    """
    tail = _clothing_category_tail(category)
    if not tail:
        return frozenset()
    drop: set[str] = set()
    if not category_has_upper_body_garment(tail):
        drop.update(("sleeve_length", "hood", "collar"))
    if not category_needs_dress_skirt_length(tail):
        drop.add("length")
    return frozenset(drop)


def strip_inapplicable_clothing_attributes(
    category: str,
    vertical: str,
    direction_id: str,
    direction_attrs: dict,
) -> dict:
    """
    Убирает из ответа модели ключи, неприменимые к категории (вертикаль «Одежда», направление clothing).
    Сохраняет error и __inscription__.
    """
    if (vertical or "").strip() != "Одежда":
        return direction_attrs
    if (direction_id or "").strip() != "clothing":
        return direction_attrs
    if not isinstance(direction_attrs, dict):
        return direction_attrs
    drop = inapplicable_clothing_attribute_keys(category)
    if not drop:
        return direction_attrs
    out: dict = {}
    for k, v in direction_attrs.items():
        if k in drop:
            continue
        out[k] = v
    return out


def _norm_attr_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    return re.sub(r"[^\w\u0400-\u04FF]+", "", s)


def _key_match_tokens(s: str) -> set[str]:
    """Токены из ключа для сопоставления синонимов (color_metal ↔ metal_color)."""
    n = _norm_attr_key(s)
    if not n:
        return set()
    parts = re.split(r"_+", n)
    return {p for p in parts if len(p) >= 2}


def _best_expected_key_by_token_overlap(
    pk: str, expected: list[str], label_for: dict[str, str]
) -> str | None:
    pt = _key_match_tokens(pk)
    if len(pt) < 2:
        return None
    scored: list[tuple[str, int]] = []
    for ek in expected:
        et = _key_match_tokens(ek) | _key_match_tokens(label_for.get(ek, ""))
        if not et:
            continue
        inter = len(pt & et)
        if inter >= 2:
            scored.append((ek, inter))
    if not scored:
        return None
    best = max(s for _, s in scored)
    winners = [ek for ek, s in scored if s == best]
    return winners[0] if len(winners) == 1 else None


def _normalize_parsed_attribute_keys(parsed: dict, attributes: list[dict]) -> dict:
    """
    Приводит ключи из ответа модели к ключам из конфига (латиница в attributes).
    Прямое совпадение / нормализация + пересечение токенов (без хардкода по конкретным атрибутам).
    """
    if not parsed or not attributes:
        return parsed
    expected = [a.get("key", "").strip() for a in attributes if a.get("key")]
    label_for = {a.get("key", "").strip(): (a.get("label") or "").strip() for a in attributes if a.get("key")}
    if not expected:
        return parsed

    canon_by_variant: dict[str, str] = {}
    for ek in expected:
        lab = label_for.get(ek, "")
        for v in {ek, ek.lower(), _norm_attr_key(ek), _norm_attr_key(lab) if lab else ""}:
            if v:
                canon_by_variant[v] = ek
        compact = _norm_attr_key(ek).replace("_", "")
        if len(compact) >= 4:
            canon_by_variant.setdefault(compact, ek)
        lab_n = _norm_attr_key(lab) if lab else ""
        if lab_n:
            lab_compact = lab_n.replace("_", "")
            if len(lab_compact) >= 4:
                canon_by_variant.setdefault(lab_compact, ek)

    out: dict = {}
    reserved = {"value", "confidence", "error"}
    for pk, val in parsed.items():
        if pk in reserved:
            out[pk] = val
            continue
        ck = (
            canon_by_variant.get(pk)
            or canon_by_variant.get((pk or "").lower())
            or canon_by_variant.get(_norm_attr_key(pk))
        )
        if not ck:
            ck = _best_expected_key_by_token_overlap(pk, expected, label_for)
        if ck:
            if ck not in out:
                out[ck] = val
            else:
                out[pk] = val
        else:
            out[pk] = val
    return out


def _normalize_freeform_attribute_value(v: str, preserve_case: bool = False) -> str:
    """Единая нормализация строкового значения атрибута (без доменных подмен).

    preserve_case=True — для verbatim-атрибутов (Original_name, артикул, VIN …):
    не переводим в нижний регистр, оставляем регистр как вернула модель.
    """
    if not isinstance(v, str):
        return v
    t = v.strip()
    if not preserve_case:
        t = t.lower()
    t = re.sub(r"[\-_]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_FAILED_EXTRACTION_VALUES = frozenset(
    {
        "",
        "unknown",
        "known",
        "?",
        "n/a",
        "na",
        "нет",
        "неизвестно",
        "неизвестен",
        "неизвестна",
        "неизвестный",
        "неизвестные",
        "известный",
        "известно",
        "неузнаваемый",
        "неузнаваемая",
        "известняк",
        "нет металла",
        "невозможно определить",
        "нельзя определить",
    }
)


def _coerce_failed_extraction_value(v: str, confidence: int) -> tuple[str, int]:
    """Нет данных по фото — пустое значение и нулевая уверенность (без подстановок из названия)."""
    if pm.attribute_value_is_placeholder_noise(v):
        return "", 0
    vl = (v or "").strip().lower()
    if vl in _FAILED_EXTRACTION_VALUES:
        return "", 0
    return v, confidence


# Если title оффера в промпт не передаётся — напоминание к «полному промпту» вручную (без автоблоков compose).
_FULL_PROMPT_IMAGE_ONLY_SUFFIX = (
    "\n\n---\n**Системное напоминание:** значения полей JSON бери **только с изображения** "
    "(видимый текст и признаки на фото). Не подставляй длинное название товара из фида/карточки вместо того, что **прочитал с фото** "
    "(не копируй title оффера вместо OCR), если только текст выше явно не поручает иное."
)


def detect_attributes_for_direction(
    image_b64: str | None,
    direction_id: str,
    direction_name: str,
    attributes: list[dict],
    custom_prompt: str,
    model: str,
    ollama_url: str,
    timeout: int = 120,
    product_name: str | None = None,
    adapter_path: Path | None = None,
    vertical: str | None = None,
    dynamic_subset: bool = False,
    task_constraints: str = "",
    task_examples: str = "",
    task_target_attribute: str = "",
    task_target_attributes: list[str] | None = None,
    full_prompt_override: str = "",
    vision_profile_calls: list | None = None,
    vision_profile_task: str = "direction",
    inline_inscriptions: bool = False,
    extra_images_b64: list[str] | None = None,
) -> dict:
    """
    Returns dict: { attr_key: { value, confidence, label }, ... }, plus "error" if any.
    Если full_prompt_override задан — он уходит в модель как есть (после подстановки {product_name}),
    без автосборки блоков из полей задания.
    При inline_inscriptions=True в ответе ожидаются text_found, texts, text_read_confidence — кладутся в ключ __inscription__.
    """
    ov = (full_prompt_override or "").strip()
    if ov:
        pn = (product_name or "").strip()
        prompt = ov.replace("{product_name}", pn)
        if not pn:
            prompt = f"{prompt}{_FULL_PROMPT_IMAGE_ONLY_SUFFIX}"
    else:
        prompt = _build_attributes_prompt(
            direction_name,
            attributes,
            custom_prompt,
            product_name,
            vertical,
            dynamic_subset=dynamic_subset,
            task_constraints=task_constraints or "",
            task_examples=task_examples or "",
            task_target_attribute=task_target_attribute or "",
            task_target_attributes=task_target_attributes,
        )
    if inline_inscriptions and prompt:
        prompt = f"{prompt}{_inline_inscription_json_suffix()}"
    if not prompt:
        return {"error": None}

    label_by_key = {a.get("key", ""): a.get("label", "") for a in attributes}

    try:
        if adapter_path is not None:
            raw = _local_adapter_chat(
                prompt,
                image_b64,
                adapter_path,
                system=None,
                vision_profile_calls=vision_profile_calls,
                vision_profile_task=vision_profile_task,
            )
        else:
            raw = _ollama_chat(
                prompt,
                image_b64,
                model,
                ollama_url,
                timeout,
                vision_profile_calls=vision_profile_calls,
                vision_profile_task=vision_profile_task,
                extra_images=extra_images_b64,
            )
        expected_keys = [a.get("key", "").strip() for a in attributes if a.get("key")]
        parsed = _parse_vision_json_response(raw, expected_keys)
        if not parsed and (raw or "").strip():
            parsed = _parse_fallback_attributes(raw, expected_keys)
        if not parsed and (raw or "").strip():
            parsed = _extract_json(raw)  # один объект на случай одного {...}
        if parsed and expected_keys:
            parsed = _hoist_nested_attribute_json(parsed, expected_keys)
        inscription_meta = None
        if parsed:
            parsed = _normalize_parsed_attribute_keys(parsed, attributes)
            # Удаляем лишние поля после нормализации (object_type, type, category и т.д.)
            known_extra_fields = {"object_type", "type", "category", "item_type", "product_type"}
            parsed = {k: v for k, v in parsed.items() if k.lower() not in known_extra_fields or k in expected_keys}
            if inline_inscriptions:
                inscription_meta = _pop_inline_inscription_from_parsed(parsed, raw or "")
        result = {"error": None}
        if not parsed and (raw or "").strip():
            preview = (raw.strip()[:200] + "…") if len(raw.strip()) > 200 else raw.strip()
            result["error"] = f"Модель вернула ответ без JSON. Ответ: {preview!r}"
        DEFAULT_CONFIDENCE_WHEN_MISSING = 75
        verbatim_keys = frozenset(
            k for k in expected_keys if _verbatim_from_image_mode("", "", "", [k], None)
        )
        # Плоский ответ {"value": "...", "confidence": N} — трактуем как один атрибут
        # Или {"key": "value", "confidence": N} — тоже плоский формат
        if parsed and set(parsed.keys()) <= {"value", "confidence"} and expected_keys:
            v_raw = str(parsed.get("value", "") or "").strip()
            first_key = expected_keys[0]
            v = _normalize_freeform_attribute_value(v_raw, preserve_case=first_key in verbatim_keys) if v_raw else ""
            c = int(parsed.get("confidence", 0)) or DEFAULT_CONFIDENCE_WHEN_MISSING
            v, c = _coerce_failed_extraction_value(v, c)
            if v and c <= 0:
                c = DEFAULT_CONFIDENCE_WHEN_MISSING
            result[first_key] = {"value": v, "confidence": c, "label": label_by_key.get(first_key, first_key)}
            if not dynamic_subset:
                for k in expected_keys[1:]:
                    result[k] = {"value": "", "confidence": 0, "label": label_by_key.get(k, k)}
            if inscription_meta:
                result["__inscription__"] = inscription_meta
            return result
        # Если confidence на верхнем уровне, используем его для всех атрибутов
        global_confidence = None
        if "confidence" in parsed and isinstance(parsed["confidence"], (int, float)):
            global_confidence = int(parsed["confidence"])

        for key, val in parsed.items():
            # Пропускаем служебные ключи и ключи, которых нет в конфиге
            if key in ("value", "confidence", "error") and key not in label_by_key:
                continue
            # Пропускаем лишние поля, которые не являются атрибутами (object_type, etc)
            if key not in label_by_key and key not in expected_keys:
                known_extra_fields = {"object_type", "type", "category", "item_type"}
                if key.lower() in known_extra_fields:
                    continue
                continue
            if isinstance(val, dict):
                rv = val.get("value", "")
                v = "" if rv is None else str(rv)
                c = int(val.get("confidence", 0))
                if c <= 0 and v and not pm.attribute_value_is_placeholder_noise(v):
                    c = DEFAULT_CONFIDENCE_WHEN_MISSING
            else:
                v = str(val)
                # Если confidence на верхнем уровне, используем его
                c = global_confidence if global_confidence is not None else (DEFAULT_CONFIDENCE_WHEN_MISSING if v else 0)
            v_raw = v.strip() if isinstance(v, str) else str(v)
            v = _normalize_freeform_attribute_value(v_raw, preserve_case=key in verbatim_keys) if v_raw else ""
            v, c = _coerce_failed_extraction_value(v, c)
            if v and c <= 0:
                c = DEFAULT_CONFIDENCE_WHEN_MISSING
            result[key] = {
                "value": v,
                "confidence": c,
                "label": label_by_key.get(key, key),
            }
        if not dynamic_subset:
            for k in expected_keys:
                if k not in result:
                    result[k] = {"value": "", "confidence": 0, "label": label_by_key.get(k, k)}
        if inscription_meta:
            result["__inscription__"] = inscription_meta
        return result
    except Exception as e:
        return {"error": str(e)}


# ── Model warmup ───────────────────────────────────────────────────────────────

def warmup_ollama_model(config: dict, timeout: int = 60) -> None:
    """One short request to load the model into memory so first real request is faster. No-op for local adapter."""
    model = config.get("model", DEFAULT_MODEL)
    if _is_adapter_path(model):
        return
    ollama_url = config.get("ollama_url", DEFAULT_OLLAMA_URL)
    print(f"[Ollama] Warming up model: {model}")
    try:
        _ollama_chat("Ответь одним словом: ок", None, model, ollama_url, timeout=timeout)
        print(f"[Ollama] Model {model} loaded in memory.")
    except Exception:
        pass


# ── Combined: parallel text + all directions ───────────────────────────────────


def parse_task_target_list_from_config(config: dict) -> tuple[list[str], str]:
    task_target_list: list[str] = []
    raw_tta = config.get("task_target_attributes")
    if isinstance(raw_tta, list):
        task_target_list = [str(x).strip() for x in raw_tta if str(x).strip()]
    elif isinstance(raw_tta, str) and raw_tta.strip():
        task_target_list = [x.strip() for x in raw_tta.split("\n") if x.strip()]
    task_target_one = (config.get("task_target_attribute") or "").strip()
    if not task_target_list and task_target_one:
        task_target_list = [x.strip() for x in task_target_one.split("\n") if x.strip()]
    task_target_list = canonicalize_target_attribute_lines(task_target_list)
    task_target_one = "\n".join(task_target_list)
    return task_target_list, task_target_one


def prepare_visual_analysis_plan(config: dict) -> dict:
    """
    Тот же расчёт направлений, что и в analyze_offer — для лога и тестов без дублирования логики.
    Ключи: directions, task_target_list, task_target_attribute, full_prompt_text, use_full_prompt, vertical.
    """
    import copy

    directions = list(config.get("directions") or [])

    def _has_work(dirs):
        if not dirs:
            return False
        if any(d.get("text_enabled", False) for d in dirs):
            return True
        return any((d.get("attributes") or []) or ((d.get("custom_prompt") or "").strip()) for d in dirs)

    if not _has_work(directions):
        import project_manager as _pm

        v0 = (config.get("vertical") or "").strip()
        if not v0 or v0 == _pm.VERTICAL_CLOTHING:
            directions = copy.deepcopy(_pm.DEFAULT_DIRECTIONS)
        else:
            directions = [
                {
                    "id": "other",
                    "name": "Другое",
                    "text_enabled": False,
                    "attributes": [],
                    "custom_prompt": "",
                }
            ]

    vertical = (config.get("vertical") or "").strip()
    if vertical and vertical != "Одежда":
        directions = [d for d in directions if d.get("id") != "clothing"]

    use_full_prompt = bool(config.get("use_full_prompt_edit"))
    full_prompt_text = (config.get("full_prompt_text") or "").strip() if use_full_prompt else ""

    task_target_list, task_target_attribute = parse_task_target_list_from_config(config)
    target_keys = set(parse_target_attribute_lines_to_keys(task_target_list))
    # Для «Одежда» строки «Извлекаемые атрибуты» с вкладки Запуск НЕ сужают направления:
    # иначе после ювелирки остаётся (metal_color) в last.json → выкидываются рукав/капюшон и т.д.
    if target_keys and vertical != "Одежда":
        directions = _filter_dirs_attrs_by_keys(directions, target_keys)
        non_empty_dirs = [d for d in directions if (d.get("attributes") or [])]
        if non_empty_dirs:
            # Оставляем только те направления, у которых остались атрибуты после фильтрации;
            # пустые (напр. clothing после отбора по target_keys) не запускаем — иначе они
            # уходят в модель с direction_name="Одежда" и порождают галлюцинации.
            directions = non_empty_dirs
        elif not full_prompt_text:
            directions = [
                {
                    "id": "other",
                    "name": "Другое",
                    "text_enabled": False,
                    "attributes": [{"key": k, "label": k, "options": []} for k in sorted(target_keys)],
                    "custom_prompt": "",
                }
            ]

    return {
        "directions": directions,
        "task_target_list": task_target_list,
        "task_target_attribute": task_target_attribute,
        "full_prompt_text": full_prompt_text,
        "use_full_prompt": use_full_prompt,
        "vertical": vertical,
    }


def _select_best_image_b64(
    images_b64: list[str],
    product_name: str | None,
    task_hint: str | None,
    model: str,
    ollama_url: str,
    timeout: int = 60,
    profile_calls: list | None = None,
) -> str | None:
    """
    Из нескольких base64-изображений одного товара выбирает лучшее для анализа атрибутов.
    Использует небольшой vision-запрос к модели.
    Возвращает b64 лучшего изображения или images_b64[0] при ошибке.
    """
    if not images_b64:
        return None
    if len(images_b64) == 1:
        return images_b64[0]

    n = len(images_b64)
    hint = ""
    if product_name:
        hint += f" of '{product_name}'"
    if task_hint:
        hint += f" for task: {task_hint}"

    prompt = (
        f"You will see {n} product images (numbered 1 to {n}). "
        f"Which image{hint} shows the main product most clearly and completely, "
        "with the best visibility for attribute analysis? "
        f"Reply with ONLY a single digit between 1 and {n}."
    )

    try:
        base = normalize_ollama_url(ollama_url or "").rstrip("/")
        messages = [{"role": "user", "content": prompt, "images": images_b64}]
        payload: dict = {"model": model, "messages": messages, "stream": False}
        if _ollama_use_think_false(model):
            payload["think"] = False

        t0 = _time.perf_counter()
        r = requests.post(f"{base}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        data = r.json()
        msg = data.get("message") or {}
        text = (msg.get("content") if isinstance(msg, dict) else "").strip()

        _vision_profile_append(profile_calls, {
            "task": "best_image_select",
            "model": model,
            "n_images": n,
            "response": text[:60],
            "elapsed_ms": round(elapsed_ms, 1),
        })

        m = re.search(r'\b([1-9]\d*)\b', text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n:
                return images_b64[idx]
    except Exception as e:
        print(f"  [SelectBestImage] Error: {e}", file=sys.stderr)

    return images_b64[0]


def analyze_offer(
    offer: dict,
    config: dict,
    timeout: int = 120,
    reanalyze_keys: Iterable[str] | None = None,
) -> dict:
    """
    Полный разбор одного оффера: текст и атрибуты по всем направлениям запускаются параллельно.
    offer: { offer_id, name, picture_urls, category }
    config: конфиг проекта (directions, model, ollama_url, image_max_size).
    reanalyze_keys: если задано — только выбранные ключи. Спец. ключ «__text__» — только надписи
    (отдельный vision-запрос); остальные — как key атрибутов в направлениях. Для слияния с БД вызывайте
    из приложения и объединяйте с сохранённой строкой.
    Returns:
      text_detection, direction_attributes: { direction_id: { attr_key: { value, confidence, label } } },
      avg_confidence, error.
    При IMAGE_DESC_PROFILE=1 в ответе поле _profile: тайминги image_prep, vision_calls, total_wall_ms.
    """
    t_analyze_start = _time.perf_counter()
    profiling = image_analysis_profiling_enabled()
    profile_calls: list | None = [] if profiling else None

    model = config.get("model", DEFAULT_MODEL)
    ollama_url = config.get("ollama_url", DEFAULT_OLLAMA_URL)
    adapter_path = _resolve_adapter_path(model)
    max_size = config.get("image_max_size", MAX_IMAGE_SIZE)

    plan = prepare_visual_analysis_plan(config)
    directions = plan["directions"]
    task_target_list = plan["task_target_list"]
    task_target_attribute = plan["task_target_attribute"]
    full_prompt_text = plan["full_prompt_text"]
    use_full_prompt = plan["use_full_prompt"]
    vertical = plan["vertical"]

    rk_set: frozenset[str] | None = None
    want_text_partial = False
    attr_keys_partial: frozenset[str] | None = None
    if reanalyze_keys is not None:
        raw = {str(x).strip() for x in reanalyze_keys if x is not None and str(x).strip()}
        if raw:
            rk_set = frozenset(raw)
            want_text_partial = "__text__" in rk_set
            attr_keys_partial = frozenset(x for x in rk_set if x != "__text__")

    extract_inscriptions = bool(config.get("extract_inscriptions", True))
    task_instruction = (config.get("task_instruction") or "").strip()

    t_img0 = _time.perf_counter()
    pic_urls = offer.get("picture_urls") or []
    cache_dir = config.get("image_cache_dir")
    if cache_dir:
        cache_dir = Path(cache_dir)

    multi_image_mode = (config.get("multi_image_mode") or "first_only").strip()
    product_name_raw = (offer.get("name") or "").strip() or None
    proj_omit_title = bool(config.get("omit_offer_title_in_prompt"))
    base_product_name = None if proj_omit_title else product_name_raw

    # Мульти-картинки: загружаем нужный набор
    extra_images_b64: list[str] = []  # дополнительные для режима all_images
    if multi_image_mode in ("best_select", "all_images") and len(pic_urls) > 1:
        all_b64 = [
            _url_to_base64(u, max_size=max_size, cache_dir=cache_dir)
            for u in pic_urls
        ]
        all_b64 = [b for b in all_b64 if b]
        if multi_image_mode == "best_select" and len(all_b64) > 1:
            task_hint = (config.get("task_instruction") or "").strip() or None
            best = _select_best_image_b64(
                all_b64,
                product_name=base_product_name,
                task_hint=task_hint,
                model=model,
                ollama_url=ollama_url,
                timeout=min(timeout, 60),
                profile_calls=profile_calls,
            )
            image_b64 = best
            pic_url = None  # URL уже не нужен
        elif multi_image_mode == "all_images" and len(all_b64) > 1:
            image_b64 = all_b64[0]
            extra_images_b64 = all_b64[1:]
            pic_url = pic_urls[0] if pic_urls else None
        else:
            pic_url = pic_urls[0] if pic_urls else None
            image_b64 = all_b64[0] if all_b64 else None
    else:
        pic_url = pic_urls[0] if pic_urls else None
        image_b64 = _url_to_base64(pic_url, max_size=max_size, cache_dir=cache_dir) if pic_url else None

    image_prep_ms = (_time.perf_counter() - t_img0) * 1000

    any_text = extract_inscriptions and any(d.get("text_enabled", False) for d in directions)

    # Для вертикали "Одежда" без задания — стандартное извлечение атрибутов; задание проекта не подставляем.
    use_project_task = bool(task_instruction and vertical != "Одежда")
    dynamic_clothing = bool(config.get("dynamic_clothing_attributes", True))
    task_constraints = (config.get("task_constraints") or "").strip()
    task_examples = (config.get("task_examples") or "").strip()

    use_same_prompt_text = bool(any_text and inscription_mode_is_same_prompt(config))
    chosen_inline_dir_id = None
    if use_same_prompt_text:
        for d in directions:
            if not d.get("text_enabled"):
                continue
            attrs = d.get("attributes") or []
            custom = (d.get("custom_prompt") or "").strip() or (task_instruction if use_project_task else "")
            attrs = resolve_attributes_for_prompt(
                attrs,
                task_target_list,
                custom,
                full_prompt_text if use_full_prompt else "",
            )
            if (vertical or "").strip() == "Одежда" and (d.get("id") or "").strip() == "clothing":
                attrs = filter_clothing_standard_attributes_for_extraction(attrs, config)
            attrs = _filter_vision_attributes(attrs, d.get("id"))
            use_dyn_chk = dynamic_clothing and bool(attrs) and not (custom.strip()) and not full_prompt_text
            if attrs or custom.strip() or full_prompt_text:
                chosen_inline_dir_id = d.get("id")
                break
    if use_same_prompt_text and chosen_inline_dir_id is None:
        use_same_prompt_text = False

    if rk_set:
        use_same_prompt_text = False

    if rk_set:
        use_separate_text = want_text_partial
    else:
        use_separate_text = bool(any_text and not use_same_prompt_text)
    text_model = resolve_inscription_model(config, model)
    tasks = []

    if use_separate_text:
        text_adapter = _resolve_adapter_path(text_model)
        tasks.append(
            (
                "text",
                None,
                detect_text,
                (
                    image_b64,
                    text_model,
                    ollama_url,
                    timeout,
                    base_product_name,
                    text_adapter,
                    profile_calls,
                    "text",
                    extra_images_b64 if extra_images_b64 else None,
                ),
            )
        )

    for d in directions:
        attrs = d.get("attributes") or []
        custom = (d.get("custom_prompt") or "").strip() or (task_instruction if use_project_task else "")
        attrs = resolve_attributes_for_prompt(
            attrs,
            task_target_list,
            custom,
            full_prompt_text if use_full_prompt else "",
        )
        if (vertical or "").strip() == "Одежда" and (d.get("id") or "").strip() == "clothing":
            attrs = filter_clothing_standard_attributes_for_extraction(attrs, config)
        attrs = _filter_vision_attributes(attrs, d.get("id"))
        if rk_set:
            if not attr_keys_partial:
                continue
            attrs = [a for a in attrs if a.get("key") in attr_keys_partial]
            attrs = _filter_vision_attributes(attrs, d.get("id"))
            if not attrs:
                continue
        use_dynamic_subset = dynamic_clothing and bool(attrs) and not (custom.strip()) and not full_prompt_text
        do_inline = bool(
            use_same_prompt_text and chosen_inline_dir_id is not None and d.get("id") == chosen_inline_dir_id
        )
        if attrs or custom.strip() or full_prompt_text:
            dir_omit = bool(d.get("omit_offer_title_in_prompt"))
            pn_for_dir = None if dir_omit else base_product_name
            tasks.append((
                "direction",
                d.get("id"),
                detect_attributes_for_direction,
                (
                    image_b64,
                    d.get("id", ""),
                    d.get("name", ""),
                    attrs,
                    custom,
                    model,
                    ollama_url,
                    timeout,
                    pn_for_dir,
                    adapter_path,
                    vertical or None,
                    use_dynamic_subset,
                    task_constraints,
                    task_examples,
                    task_target_attribute,
                    task_target_list if task_target_list else None,
                    full_prompt_text,
                    profile_calls,
                    f"direction:{d.get('id', '')}",
                    do_inline,
                    extra_images_b64 if extra_images_b64 else None,
                ),
            ))

    text_result = {}
    direction_results = {}

    try:
        cap_raw = config.get("max_parallel_vision", 0)
        cap = int(cap_raw) if cap_raw is not None else 0
    except (TypeError, ValueError):
        cap = 0
    n_tasks = len(tasks)
    if n_tasks == 0 and rk_set:
        return {
            "offer_id": offer.get("offer_id", ""),
            "name": offer.get("name", ""),
            "category": offer.get("category", ""),
            "picture_url": pic_url or "",
            "text_detection": {},
            "direction_attributes": {},
            "avg_confidence": 0,
            "error": "Частичный прогон: ни один ключ не совпал с атрибутами направлений",
        }
    if cap > 0:
        workers = max(1, min(cap, n_tasks)) if n_tasks else 1
    else:
        workers = max(1, n_tasks) if n_tasks else 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, *args): (kind, did) for kind, did, fn, args in tasks}
        for fut in as_completed(futures):
            kind, direction_id = futures[fut]
            try:
                data = fut.result()
                if kind == "text":
                    text_result = data
                else:
                    direction_results[direction_id] = data
            except Exception as e:
                if kind == "text":
                    text_result = {"error": str(e), "texts": [], "confidence": 0}
                else:
                    direction_results[direction_id] = {"error": str(e)}

    if use_same_prompt_text and chosen_inline_dir_id:
        dr = direction_results.get(chosen_inline_dir_id)
        if isinstance(dr, dict) and "__inscription__" in dr:
            text_result = dr.pop("__inscription__") or {}

    cat_for_strip = offer.get("category") or ""
    v_for_strip = vertical or ""
    for _did, _dr in list(direction_results.items()):
        if isinstance(_dr, dict) and not _dr.get("error"):
            direction_results[_did] = strip_inapplicable_clothing_attributes(
                cat_for_strip, v_for_strip, str(_did or ""), _dr
            )

    pm.strip_forbidden_attribute_keys_inplace(direction_results)

    # Просим RU в промпте; если модель всё же дала EN — дожимаем глоссарием. Надписи (text_detection) не трогаем.
    pm.translate_direction_attribute_values_inplace(direction_results)
    pm.strip_placeholder_attribute_values_inplace(direction_results)

    _gs_tr = pm.get_global_settings()
    apply_llm_translate_remaining_latin_inplace(
        direction_results,
        enabled=bool(_gs_tr.get("attribute_value_llm_translate")),
        translate_model=str(_gs_tr.get("attribute_value_translate_model") or ""),
        vision_model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        vision_profile_calls=profile_calls,
    )
    pm.strip_placeholder_attribute_values_inplace(direction_results)
    pm.normalize_presence_like_attribute_values_inplace(direction_results)
    pm.sanitize_print_pattern_in_direction_inplace(direction_results)

    confidence_scores = []
    if text_result.get("confidence") is not None:
        confidence_scores.append(text_result["confidence"])
    for dr in direction_results.values():
        if dr.get("error"):
            continue
        for k, v in dr.items():
            if k == "error":
                continue
            if isinstance(v, dict) and "confidence" in v:
                confidence_scores.append(v["confidence"])
    avg_confidence = int(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0

    err = text_result.get("error")
    for dr in direction_results.values():
        if dr.get("error"):
            err = err or dr["error"]

    out = {
        "offer_id": offer.get("offer_id", ""),
        "name": offer.get("name", ""),
        "category": offer.get("category", ""),
        "picture_url": pic_url or "",
        "text_detection": text_result,
        "direction_attributes": direction_results,
        "avg_confidence": avg_confidence,
        "error": err,
    }
    if profiling:
        calls_snapshot = list(profile_calls or [])
        sum_vision = sum(
            float(c["ms"])
            for c in calls_snapshot
            if isinstance(c, dict) and isinstance(c.get("ms"), (int, float))
        )
        _prof = {
            "total_wall_ms": round((_time.perf_counter() - t_analyze_start) * 1000, 2),
            "image_prep_ms": round(image_prep_ms, 2),
            "max_parallel_vision": workers,
            "vision_calls": calls_snapshot,
            "vision_calls_sum_ms": round(sum_vision, 2),
        }
        out["_profile"] = _prof
        print(
            "[IMAGE_DESC_PROFILE] "
            f"offer_id={offer.get('offer_id', '')!r} "
            f"total_ms={_prof['total_wall_ms']} "
            f"image_prep_ms={_prof['image_prep_ms']} "
            f"sum_vision_ms={_prof['vision_calls_sum_ms']} "
            f"workers={workers} "
            f"n_calls={len(calls_snapshot)}",
            file=sys.stderr,
        )
    return out
