#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "$(uname -m)" == "aarch64" ]] &&
   [[ -f /etc/nv_tegra_release ]] &&
   grep -q '^# R39 ' /etc/nv_tegra_release; then
  # R39 Jetson: system Python lacks ensurepip, and the host CUDA driver is 13.2.
  UV_BIN="${UV_BIN:-$(command -v uv || true)}"
  if [[ -z "$UV_BIN" ]] && [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  fi
  if [[ -z "$UV_BIN" ]] || [[ ! -x "$UV_BIN" ]]; then
    echo "uv is required. Install it in your user directory: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi

  if [[ ! -x .venv/bin/python ]]; then
    "$UV_BIN" venv --python /usr/bin/python3.12 --seed --no-python-downloads .venv
  fi
  "$UV_BIN" pip install --python .venv/bin/python --torch-backend=cu130 \
    'https://github.com/vllm-project/vllm/releases/download/v0.18.0/vllm-0.18.0%2Bcu130-cp38-abi3-manylinux_2_35_aarch64.whl'
  "$UV_BIN" pip install --python .venv/bin/python 'bitsandbytes==0.50.2'

  # Triton compiles a small Python extension at runtime. R39's minimal image
  # has no Python.h, so unpack the Ubuntu headers locally without sudo.
  PYTHON_DEV_DIR="${REMTP_PYTHON_DEV_DIR:-$HOME/.local/share/remtp/python3.12-dev}"
  if [[ ! -f /usr/include/python3.12/Python.h ]] &&
     [[ ! -f "$PYTHON_DEV_DIR/usr/include/python3.12/Python.h" ]]; then
    package_dir="$(mktemp -d)"
    (cd "$package_dir" && apt download libpython3.12-dev)
    dpkg-deb -x "$package_dir"/libpython3.12-dev_*_arm64.deb "$PYTHON_DEV_DIR"
    rm -rf "$package_dir"
  fi
else
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U uv
  uv pip install 'vllm==0.18.0' --torch-backend=auto
fi

source .venv/bin/activate
source "$PROJECT_DIR/scripts/cuda_env.sh"
python -c 'import vllm; print("vLLM installed:", vllm.__version__)'
