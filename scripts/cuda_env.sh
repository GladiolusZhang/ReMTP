#!/usr/bin/env bash

# Some nightly wheels ship a CUDA runtime under site-packages/nvidia/cu*/lib.
# Add it lazily so vLLM's compiled extension can locate libcudart.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  for cuda_library_dir in \
    "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/cu*/lib
  do
    if [[ -d "$cuda_library_dir" ]]; then
      export LD_LIBRARY_PATH="${cuda_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
  done
fi

# Jetson R39 base images may omit Python.h, which Triton needs for its JIT
# driver module. scripts/install.sh extracts the Ubuntu headers under $HOME.
if [[ "$(uname -m)" == "aarch64" ]] &&
   [[ -f /etc/nv_tegra_release ]] &&
   grep -q '^# R39 ' /etc/nv_tegra_release &&
   [[ ! -f /usr/include/python3.12/Python.h ]]; then
  python_dev_dir="${REMTP_PYTHON_DEV_DIR:-$HOME/.local/share/remtp/python3.12-dev}"
  if [[ -f "$python_dev_dir/usr/include/python3.12/Python.h" ]]; then
    export CPATH="$python_dev_dir/usr/include:$python_dev_dir/usr/include/python3.12${CPATH:+:$CPATH}"
  fi
fi
