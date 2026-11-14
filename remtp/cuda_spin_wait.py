"""Opt-in spin waiting for host-device synchronisation (engineering only).

``REMTP_CUDA_SPIN_WAIT=1`` sets ``CU_CTX_SCHED_SPIN`` on the device's primary
context before vLLM initialises it.  A host thread that waits for the GPU
(``.item()``, ``.tolist()``, ``.cpu()``) then polls instead of sleeping.

Why: with CUDA graphs a decode step launches the whole target forward in
~0.2 ms and the engine thread then waits ~50 ms for the GPU.  If it sleeps,
the Jetson's ``schedutil`` CPU governor lowers the core clock, and the
latency-critical Python that follows (verification, tree construction, MTP
launches) runs at a reduced clock.  The governor cannot be changed without
root; polling keeps the core busy.  No kernel, value, RNG draw or launch order
changes: only how the host thread waits.
"""
from __future__ import annotations

import ctypes
import os

CU_CTX_SCHED_SPIN = 0x01
CU_CTX_SCHED_MASK = 0x07
_INSTALLED = False


def _check(code: int, what: str) -> None:
    if code != 0:
        raise RuntimeError(f"CUDA driver call {what} failed with error {code}")


def install_cuda_spin_wait(device_index: int | None = None) -> None:
    global _INSTALLED
    if _INSTALLED or os.getenv("REMTP_CUDA_SPIN_WAIT", "0") != "1":
        return
    if device_index is None:
        device_index = int(os.getenv("LOCAL_RANK", "0"))
    lib = ctypes.CDLL("libcuda.so.1")
    _check(lib.cuInit(0), "cuInit")
    device = ctypes.c_int()
    _check(lib.cuDeviceGet(ctypes.byref(device), device_index), "cuDeviceGet")
    flags, active = ctypes.c_uint(), ctypes.c_int()
    _check(lib.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active)),
           "cuDevicePrimaryCtxGetState")
    new_flags = (flags.value & ~CU_CTX_SCHED_MASK) | CU_CTX_SCHED_SPIN
    setter = getattr(lib, "cuDevicePrimaryCtxSetFlags_v2", None) or lib.cuDevicePrimaryCtxSetFlags
    _check(setter(device, ctypes.c_uint(new_flags)), "cuDevicePrimaryCtxSetFlags")
    _check(lib.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active)),
           "cuDevicePrimaryCtxGetState")
    if flags.value & CU_CTX_SCHED_MASK != CU_CTX_SCHED_SPIN:
        raise RuntimeError(f"primary context scheduling flags not applied: {flags.value:#x}")
    _INSTALLED = True
    print(f"[ReMTP][CudaSpinWait] primary context sched=SPIN (flags={flags.value:#x}, "
          f"active_before={bool(active.value)})", flush=True)
