#!/usr/bin/env bash
# Download Qwen2.5-0.5B-Instruct into a local directory. Does not commit weights.
# Usage: scripts/download-qwen25-0.5b.sh [dest] [--no-weights]
set -euo pipefail

DEST="${1:-models/Qwen2.5-0.5B-Instruct}"
if [[ "${1:-}" == "--no-weights" ]]; then
  DEST="models/Qwen2.5-0.5B-Instruct"
fi
NO_WEIGHTS=0
for arg in "$@"; do
  if [[ "$arg" == "--no-weights" ]]; then
    NO_WEIGHTS=1
  fi
done

mkdir -p "$DEST"
BASE="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main"

download() {
  local name="$1"
  local path="$DEST/$name"
  if [[ -f "$path" && -s "$path" ]]; then
    echo "present $path"
    return
  fi
  echo "downloading $name"
  curl -L --fail --retry 5 --retry-delay 2 -o "$path.partial" "$BASE/$name"
  mv "$path.partial" "$path"
}

download config.json
download generation_config.json
download tokenizer_config.json
download tokenizer.json
if [[ "$NO_WEIGHTS" -eq 0 ]]; then
  download model.safetensors
fi

echo "ready $DEST"
