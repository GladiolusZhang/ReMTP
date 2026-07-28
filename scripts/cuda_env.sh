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
