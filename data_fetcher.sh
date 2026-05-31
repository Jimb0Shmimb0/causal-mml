#!/usr/bin/env bash
# Download selected datasets from cmu-phil/example-causal-datasets
# Source: https://github.com/cmu-phil/example-causal-datasets

set -euo pipefail

BASE_URL="https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real"
DATA_DIR="data"

download() {
  local dataset="$1"
  local filename="$2"

  local dest="$DATA_DIR/$dataset"

  mkdir -p "$dest"

  local url="$BASE_URL/$dataset/data/$filename"
  local out="$dest/$filename"

  if [ -f "$out" ]; then
    echo "  [skip] $out already exists"
    return
  fi

  echo "  Downloading $filename..."
  if command -v curl &>/dev/null; then
    curl -fsSL "$url" -o "$out"
  elif command -v wget &>/dev/null; then
    wget -q "$url" -O "$out"
  else
    echo "ERROR: neither curl nor wget found" >&2
    exit 1
  fi
}

# --- Datasets ---
# echo "Fetching: Asia network"
download "sachs" "sachs.2005.discrete.txt"
echo "Done. Data saved to ./$DATA_DIR/"