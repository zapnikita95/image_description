"""Check prompt size per direction for DeepFashion."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "generated/deepfashion/deepfashion_directions.json"
dirs = json.loads(p.read_text(encoding="utf-8"))
for i, d in enumerate(dirs):
    attrs = d.get("attributes", [])
    keys = [a.get("key", "") for a in attrs]
    keys_str = ", ".join(keys)
    example_parts = [f'"{k}":{{"value":"yes","confidence":80}}' for k in keys]
    example_line = "{" + ",".join(example_parts) + "}"
    base = f"Analyze the clothing in the image. Direction: {d.get('name','')}. You MUST return ONE JSON object containing ALL these keys (no omissions): "
    full = base + keys_str + ". For each key use a value from the suggested options when it fits, or suggest your own value. Use English. Reply with ONLY this JSON. Example: " + example_line
    print(f"Chunk {i}: {len(attrs)} keys, prompt={len(full)} chars, ~{len(full)//4} tokens")
print("Total directions:", len(dirs))
