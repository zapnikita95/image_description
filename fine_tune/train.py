#!/usr/bin/env python3
"""
LoRA fine-tuning via unsloth.

Requirements:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps trl peft accelerate bitsandbytes

Usage (from command line):
  python fine_tune/train.py --dataset fine_tune/dataset/train.jsonl \
      --model Qwen/Qwen2.5-VL-3B-Instruct --output fine_tune/lora_out
  # Or Qwen3-VL (stronger vision): Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3-VL-32B-Instruct
"""

import argparse
import base64
import io
import json
import logging
import os
import random
import sys
import threading
import warnings
from pathlib import Path

# Отключаем torch.compile (inductor) — на Windows triton несовместим с текущим PyTorch
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Меньше шума: Unsloth Zoo GRPO не ставится на Windows/Triton — мы его не используем
logging.getLogger("unsloth_zoo.log").setLevel(logging.ERROR)


def check_gpu() -> tuple[bool, str]:
    """Return (available, message)."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            return True, f"GPU: {gpu} ({mem} GB VRAM)"
        # CUDA не видна — чаще всего установлен CPU-only PyTorch
        import shutil
        nvidia_ok = shutil.which("nvidia-smi") is not None
        if nvidia_ok:
            return False, (
                "CUDA в PyTorch недоступна, хотя nvidia-smi есть — скорее всего установлен PyTorch без CUDA.\n\n"
                "Переустановите PyTorch с CUDA (в том же venv):\n"
                "  pip uninstall torch -y\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n\n"
                "Для CUDA 11.8 замените cu121 на cu118. После установки перезапустите приложение."
            )
        return False, (
            "NVIDIA GPU не обнаружена. Для дообучения нужна видеокарта NVIDIA и драйвер.\n\n"
            "1) Установите драйвер NVIDIA с https://www.nvidia.com/drivers\n"
            "2) Переустановите PyTorch с CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    except ImportError:
        return False, "torch not installed"


def check_unsloth() -> tuple[bool, str]:
    try:
        import unsloth  # noqa: F401
        return True, "unsloth available"
    except ImportError:
        return False, (
            "unsloth not installed.\n"
            "Install: pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n"
            "         pip install --no-deps trl peft accelerate bitsandbytes"
        )


def train(
    dataset_path: str | Path,
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    output_dir: str | Path = "fine_tune/lora_out",
    max_steps: int = 200,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    progress_callback=None,
    max_train_examples: int = 0,
    resume_from_adapter: str | Path | None = None,
    log_file: str | Path | None = None,
    stop_event: threading.Event | None = None,
    skip_first_n: int = 0,
) -> dict:
    """
    Run LoRA fine-tuning.
    progress_callback(step, total, loss) — optional, called each step.
    stop_event: if set, training stops after current step.
    Returns {success, output_dir, error}.
    """
    gpu_ok, gpu_msg = check_gpu()
    if not gpu_ok:
        return {"success": False, "error": gpu_msg}

    unsloth_ok, unsloth_msg = check_unsloth()
    if not unsloth_ok:
        return {"success": False, "error": unsloth_msg}

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file) if log_file else None

    def _log(msg: str):
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
        print(msg)

    if not dataset_path.exists():
        return {"success": False, "error": f"Dataset not found: {dataset_path}"}

    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments

        # Load examples
        examples = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        if not examples:
            return {"success": False, "error": "Dataset is empty"}

        total_in_file = len(examples)
        if skip_first_n > 0:
            examples = examples[skip_first_n:]
            _log(f"[LoRA] Пропущено уже использованных: {skip_first_n}. Осталось в выборке: {len(examples)}")
        if not examples:
            return {"success": False, "error": f"Все примеры уже использованы (skip_first_n={skip_first_n}). Добавьте новые в датасет или сбросьте счётчик."}
        random.shuffle(examples)
        if max_train_examples and max_train_examples > 0:
            examples = examples[:max_train_examples]

        has_images = any(
            any(isinstance(v, dict) and v.get("type") == "image" for v in (c.get("value") or []) if isinstance(c.get("value"), list))
            for ex in examples for c in ex.get("conversations", [])
        )

        resume_path = Path(resume_from_adapter) if resume_from_adapter else None
        # При дозагрузке берём базу из конфига адаптера — иначе размеры весов не совпадут (адаптер с 7B, база 3B и т.д.)
        if resume_path and resume_path.exists():
            adapter_config_path = resume_path / "adapter_config.json"
            if adapter_config_path.exists():
                try:
                    cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                    base_from_adapter = cfg.get("base_model_name_or_path") or cfg.get("base_model")
                    if base_from_adapter and isinstance(base_from_adapter, str) and base_from_adapter.strip():
                        base_model = base_from_adapter.strip()
                        _log(f"[LoRA] База из конфига адаптера: {base_model}")
                except Exception:
                    pass
            _log(f"[LoRA] Resuming from adapter: {resume_path}")
        grad_accum = 4
        effective_batch = batch_size * grad_accum
        _log(f"[LoRA] Dataset: {len(examples)} examples | base: {base_model} | VL (images): {has_images}")
        _log(f"[LoRA] max_steps={max_steps} | batch_size={batch_size} | grad_accum={grad_accum} | effective_batch={effective_batch}")
        _log("[LoRA] Loading model...")

        if has_images:
            # ─── Vision-language path: use FastVisionModel and UnslothVisionDataCollator ───
            try:
                from PIL import Image
                from datasets import Dataset
                from unsloth import FastVisionModel
                try:
                    from unsloth.trainer import UnslothVisionDataCollator
                except ImportError:
                    from unsloth_zoo.vision_utils import UnslothVisionDataCollator
                from trl import SFTConfig, SFTTrainer
            except ImportError as e:
                return {
                    "success": False,
                    "error": (
                        "VL training requires: unsloth with FastVisionModel and UnslothVisionDataCollator. "
                        f"Import failed: {e}. Upgrade unsloth or use dataset without images (uncheck 'Включить картинки')."
                    ),
                }

            def _b64_to_pil(b64: str):
                if not b64:
                    return None
                try:
                    raw = base64.b64decode(b64)
                    return Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    return None

            def sharegpt_to_messages(ex):
                """Convert ShareGPT conversations to Unsloth messages: content with PIL images."""
                messages = []
                for c in ex.get("conversations", []):
                    role = c.get("role", "")
                    val = c.get("value", "")
                    if role == "system":
                        continue
                    if isinstance(val, list):
                        content = []
                        for v in val:
                            if not isinstance(v, dict):
                                continue
                            if v.get("type") == "text":
                                content.append({"type": "text", "text": v.get("text", "")})
                            elif v.get("type") == "image":
                                img_b64 = v.get("image") or v.get("url") or ""
                                pil_img = _b64_to_pil(img_b64)
                                if pil_img is not None:
                                    content.append({"type": "image", "image": pil_img})
                        if content:
                            messages.append({"role": role, "content": content})
                    else:
                        messages.append({"role": role, "content": [{"type": "text", "text": str(val)}]})
                return {"messages": messages} if messages else None

            vl_examples = []
            for e in examples:
                out = sharegpt_to_messages(e)
                if out and any(
                    any(bloc.get("type") == "image" for bloc in m.get("content", []))
                    for m in out["messages"]
                ):
                    vl_examples.append(out)
            if not vl_examples:
                return {
                    "success": False,
                    "error": "Dataset has image blocks but no valid image could be decoded (check base64). Rebuild dataset with 'Включить картинки'.",
                }
            _log(f"[LoRA] VL examples with images: {len(vl_examples)} (skipped {len(examples) - len(vl_examples)} without image)")

            # Убираем спам от Qwen3.5 MoE: в чекпоинте веса не tied, в конфиге — tied; на обучение не влияет
            warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
            warnings.filterwarnings("ignore", message=".*tied weights mapping.*")
            warnings.filterwarnings("ignore", message=".*so we will NOT tie them.*")
            # LoRA грузит веса из HuggingFace/Unsloth (4bit/bf16) + градиенты и активации — это тяжелее,
            # чем инференс qwen3.5:27b/35b в Ollama (GGUF Q4 на диске ~17–23 ГБ). На ~24 GB 27B/35B часто упираются в OOM.
            try:
                import torch
                vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3) if torch.cuda.is_available() else 0
            except Exception:
                vram_gb = 0
            if vram_gb < 32 and ("35B" in base_model or "27B" in base_model):
                _log(
                    f"[LoRA] Предупреждение: GPU ~{vram_gb} GB — для VL 27B/35B обучение часто не влезает в VRAM "
                    "(не путать с **запуском** тех же имён в Ollama: там другие веса и меньше памяти). "
                    "Надёжный вариант на 24 GB — **9B/4B**; если ниже будет OOM — уменьши модель или увеличь VRAM."
                )
            # 32–40 GB: 35B/27B в 4bit; 40+ GB: bf16
            if "35B" in base_model or "27B" in base_model:
                load_4bit = vram_gb < 40
                if load_4bit and "35B" in base_model:
                    _log(f"[LoRA] VRAM {vram_gb} GB — для 35B включаем 4bit.")
            else:
                load_4bit = True
            # На 24 GB для 9B/4B уменьшаем batch, чтобы точно влезло
            if vram_gb > 0 and vram_gb < 32:
                batch_size = min(batch_size, 1)
                grad_accum = 4
                _log(f"[LoRA] VRAM {vram_gb} GB — batch_size=1 для экономии памяти.")
            model, processor = None, None
            attempts = [base_model]
            # На HuggingFace 35B — только Qwen/Qwen3.5-35B-A3B (отдельного Qwen3.5-35B-Instruct нет)
            if "35B" in base_model and "A3B" not in base_model:
                attempts.insert(0, "Qwen/Qwen3.5-35B-A3B")
            if "-Instruct" in base_model:
                alt = base_model.replace("-Instruct", "")
                if alt not in attempts:
                    attempts.append(alt)
            for attempt_name in attempts:
                try:
                    kwargs = dict(
                        model_name=attempt_name,
                        max_seq_length=2048,
                        dtype=None,
                        load_in_4bit=load_4bit,
                    )
                    if not load_4bit:
                        kwargs["load_in_16bit"] = True
                    model, processor = FastVisionModel.from_pretrained(**kwargs)
                    if attempt_name != base_model:
                        _log(f"[LoRA] Загружено как {attempt_name}")
                    break
                except Exception as e:
                    err = str(e)
                    if "No config file found" in err or "config file" in err.lower():
                        if attempt_name == attempts[-1]:
                            raise RuntimeError(
                                f"Unsloth Zoo не содержит конфиг для этой модели. Выбери Qwen3.5-9B или Qwen3.5-27B, "
                                f"или обнови: pip install -U unsloth unsloth_zoo. Ошибка: {e}"
                            ) from e
                        _log(f"[LoRA] Нет конфига для {attempt_name}, пробуем следующий вариант...")
                        continue
                    raise
            if model is None:
                raise RuntimeError("Не удалось загрузить VL-модель")
            _log("[LoRA] VL model loaded. Applying LoRA or loading adapter...")
            try:
                if resume_path and resume_path.exists():
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, str(resume_path))
                else:
                    model = FastVisionModel.get_peft_model(
                        model,
                        r=lora_rank,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        lora_alpha=lora_rank,
                        lora_dropout=0,
                        bias="none",
                        use_gradient_checkpointing="unsloth",
                        random_state=42,
                        finetune_vision_layers=True,
                        finetune_language_layers=True,
                    )
                FastVisionModel.for_training(model)
            except Exception as e:
                err_msg = str(e)
                if "meta tensor" in err_msg.lower() or "no data" in err_msg.lower():
                    raise RuntimeError(
                        "Не хватает видеопамяти: модель частично на meta-устройстве. "
                        "Попробуй **Qwen3.5-9B** или **Qwen3.5-4B**; для 27B/35B нужна большая VRAM, чем для инференса тех же имён в Ollama."
                    ) from e
                if "dispatched on the cpu" in err_msg.lower() or "dispatched on the disk" in err_msg.lower():
                    raise RuntimeError(
                        "Модель не влезла в VRAM (часть на CPU/диске). Для LoRA на ~24 GB чаще подходят **9B/4B**; "
                        "27B/35B — риск OOM (это не то же самое, что запуск qwen3.5:27b/35b в Ollama)."
                    ) from e
                raise

            hf_dataset = Dataset.from_list(vl_examples)
            data_collator = UnslothVisionDataCollator(model, processor, max_seq_length=2048)

            callbacks = []
            if progress_callback or log_path or stop_event:
                from transformers import TrainerCallback

                def _on_log(step, total, loss):
                    pct = int(round(100 * step / total)) if total else 0
                    line = f"Step {step}/{total} ({pct}%)  loss={loss:.4f}"
                    if progress_callback:
                        progress_callback(step, total, loss)
                    _log(line)

                class _CB(TrainerCallback):
                    def on_log(self, args, state, control, logs=None, **kwargs):
                        if stop_event and stop_event.is_set():
                            control.should_training_stop = True
                        if logs is not None:
                            _on_log(state.global_step, max_steps, logs.get("loss", 0))

                callbacks.append(_CB())

            trainer = SFTTrainer(
                model=model,
                tokenizer=processor,
                train_dataset=hf_dataset,
                data_collator=data_collator,
                args=SFTConfig(
                    remove_unused_columns=False,
                    dataset_text_field="",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=4,
                    warmup_steps=10,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    bf16=True,
                    logging_steps=1,
                    output_dir=str(output_dir / "checkpoints"),
                    save_strategy="no",
                    report_to="none",
                ),
                callbacks=callbacks,
            )
            _log(f"[LoRA] Train examples: {len(vl_examples)}. Starting VL training (with images)...")
            trainer.train()
            # Сохраняем адаптер в любом случае (и при остановке, и при успехе)
            _log("[LoRA] Saving adapter...")
            adapter_dir = output_dir / "lora_adapter"
            model.save_pretrained(str(adapter_dir))
            processor.save_pretrained(str(adapter_dir))
            if stop_event and stop_event.is_set():
                _log("[LoRA] Остановлено по запросу пользователя. Адаптер сохранён — можно дообучить с него дальше.")
                return {"success": False, "output_dir": str(adapter_dir), "error": "Остановлено пользователем. Адаптер сохранён.", "examples_used": getattr(trainer.state, "global_step", max_steps)}
            _log(f"[LoRA] Done. Adapter: {adapter_dir}")
        else:
            # ─── Text-only path (no images) — supports Qwen3.5, Qwen2.5, Llama, etc. ───
            from unsloth import FastLanguageModel
            from datasets import Dataset
            from trl import SFTTrainer, SFTConfig

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            _log("[LoRA] Model loaded. Applying LoRA or loading adapter...")
            if resume_path and resume_path.exists():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(resume_path))
            else:
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=lora_rank,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                    lora_alpha=lora_rank,
                    lora_dropout=0,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=42,
                )
            FastLanguageModel.for_training(model)

            def sharegpt_to_messages(ex):
                msgs = []
                for c in ex.get("conversations", []):
                    role = c.get("role", "")
                    val = c.get("value", "")
                    if isinstance(val, list):
                        val = " ".join(
                            v.get("text", "") for v in val if isinstance(v, dict) and v.get("type") == "text"
                        )
                    if role in ("system", "user", "assistant") and str(val).strip():
                        msgs.append({"role": role, "content": str(val)})
                return msgs

            def format_example(ex):
                msgs = sharegpt_to_messages(ex)
                if not msgs:
                    return {"text": ""}
                try:
                    text = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception:
                    text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
                return {"text": text}

            formatted = [format_example(e) for e in examples]
            formatted = [x for x in formatted if x["text"].strip()]
            hf_dataset = Dataset.from_list(formatted)
            _log(f"[LoRA] Train examples: {len(formatted)}. Starting training...")

            callbacks = []
            if progress_callback or log_path or stop_event:
                from transformers import TrainerCallback

                def _on_log(step, total, loss):
                    pct = int(round(100 * step / total)) if total else 0
                    line = f"Step {step}/{total} ({pct}%)  loss={loss:.4f}"
                    if progress_callback:
                        progress_callback(step, total, loss)
                    _log(line)

                class _CB(TrainerCallback):
                    def on_log(self, args, state, control, logs=None, **kwargs):
                        if stop_event and stop_event.is_set():
                            control.should_training_stop = True
                        if logs is not None:
                            _on_log(state.global_step, max_steps, logs.get("loss", 0))

                callbacks.append(_CB())

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=hf_dataset,
                args=SFTConfig(
                    dataset_text_field="text",
                    max_seq_length=2048,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=4,
                    warmup_steps=10,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    fp16=False,
                    bf16=True,
                    logging_steps=1,
                    output_dir=str(output_dir / "checkpoints"),
                    save_strategy="no",
                    report_to="none",
                ),
                callbacks=callbacks,
            )
            trainer.train()
            adapter_dir = output_dir / "lora_adapter"
            _log("[LoRA] Saving adapter...")
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            if stop_event and stop_event.is_set():
                _log("[LoRA] Остановлено по запросу пользователя. Адаптер сохранён.")
                return {"success": False, "output_dir": str(adapter_dir), "error": "Остановлено пользователем. Адаптер сохранён.", "examples_used": getattr(trainer.state, "global_step", max_steps)}
            _log(f"[LoRA] Done. Adapter: {adapter_dir}")

        return {"success": True, "output_dir": str(output_dir / "lora_adapter"), "error": None, "examples_used": len(examples)}

    except Exception as e:
        _log(f"[LoRA] Error: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output", default="fine_tune/lora_out")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--max-train-examples", type=int, default=0, help="Max examples from dataset (0=all)")
    p.add_argument("--resume-from-adapter", default="", help="Path to previous LoRA adapter to continue training")
    args = p.parse_args()

    result = train(
        dataset_path=args.dataset,
        base_model=args.model,
        output_dir=args.output,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        progress_callback=lambda step, total, loss: print(f"  Step {step}/{total}  loss={loss:.4f}"),
        max_train_examples=args.max_train_examples,
        resume_from_adapter=args.resume_from_adapter or None,
    )
    if result["success"]:
        print(f"Training complete. Adapter saved to: {result['output_dir']}")
    else:
        print(f"Training failed: {result['error']}", file=sys.stderr)
        sys.exit(1)
