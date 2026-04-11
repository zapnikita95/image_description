#!/usr/bin/env python3
"""Проверка, использует ли Ollama GPU."""
import requests
import sys

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11434"
base = url.rstrip("/")

try:
    # Проверка загруженных моделей (показывает VRAM usage)
    r = requests.get(f"{base}/api/ps", timeout=5)
    r.raise_for_status()
    models = r.json().get("models") or []
    print(f"Models in memory: {len(models)}")
    total_vram = 0
    for m in models:
        name = m.get("name") or m.get("model", "?")
        vram = m.get("size_vram", 0)
        total_vram += vram
        print(f"  {name}: {vram / (1024**3):.2f} GB VRAM")
    if total_vram > 0:
        print(f"\nTotal VRAM used: {total_vram / (1024**3):.2f} GB")
        print("✅ GPU is being used (VRAM allocation detected)")
    else:
        print("\n⚠️  No VRAM usage detected — model might be on CPU")
except Exception as e:
    print(f"Error checking Ollama: {e}")
