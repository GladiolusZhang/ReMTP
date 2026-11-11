#!/usr/bin/env bash
# Normalize one upstream tokenizer metadata field for Transformers 4.57.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/FastMTP}"
python3 - "$MODEL_PATH" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

model = Path(sys.argv[1])
config_path = model / "config.json"
backup_path = model / "config.hf-original.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
value = config.get("tokenizer_class")
if value == "Qwen2Tokenizer":
    print(f"FastMTP tokenizer metadata ready: {config_path}")
    raise SystemExit(0)
if value != ["Qwen2Tokenizer"]:
    raise SystemExit(f"Unexpected FastMTP tokenizer_class: {value!r}")
if backup_path.exists():
    original = json.loads(backup_path.read_text(encoding="utf-8"))
    if original != config:
        raise SystemExit(f"Existing backup differs from current source: {backup_path}")
else:
    shutil.copy2(config_path, backup_path)
config["tokenizer_class"] = "Qwen2Tokenizer"
fd, temporary = tempfile.mkstemp(prefix=".config.", suffix=".json", dir=model)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, config_path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(f"Normalized tokenizer_class; original saved at {backup_path}")
PY
