# === PATCH loader.py ===
import json
import os

def load_musicbench_jsonl(file_path):
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        head = f.read(1024)
        f.seek(0)
        # Heuristique : si ça commence par '[' -> JSON array, sinon JSONL
        if head.lstrip().startswith('['):
            data = json.load(f)
            if isinstance(data, list):
                entries = data
        else:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries
