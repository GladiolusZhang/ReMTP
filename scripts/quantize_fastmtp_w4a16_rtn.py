#!/usr/bin/env python3
"""Stream a FastMTP checkpoint into a GPTQ-format W4A16 target (round-to-nearest).

Only the 36 target decoder linear layers (q/k/v/o, gate/up/down) are quantized:
4-bit, symmetric, group size 128, no act-order.  Embeddings, the target LM
head, norms, biases and the physical MTP layer (``model.mtp_layers.*``) are
copied unchanged in BF16, so the draft head stays exactly as in the
bitsandbytes setup.

This is a calibration-free proxy.  It loads through vLLM's ``gptq_marlin``
path with the same kernel and memory traffic as a GPTQ/AWQ checkpoint, so it
is valid for kernel feasibility and speed measurements.  Its accuracy is not
that of a calibrated GPTQ/AWQ model and must not be reported as such.

Tensors are processed one at a time from memory-mapped shards; peak memory is
about one tensor plus one output shard.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LINEAR = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)
BITS = 4
PACK = 32 // BITS
ZERO = 1 << (BITS - 1)  # symmetric uint4b8: w = (q - 8) * scale


def _to_int32(packed64: torch.Tensor) -> torch.Tensor:
    wrapped = torch.where(packed64 >= 2**31, packed64 - 2**32, packed64)
    return wrapped.to(torch.int32)


def quantize_linear(weight: torch.Tensor, group: int) -> dict[str, torch.Tensor]:
    """Return GPTQ v1 tensors for a PyTorch [out, in] weight."""
    out_features, in_features = weight.shape
    if in_features % group or in_features % PACK or out_features % PACK:
        raise ValueError(f"unsupported shape {tuple(weight.shape)} for group {group}")
    w = weight.to(torch.float32).t().contiguous()  # [in, out]
    grouped = w.reshape(in_features // group, group, out_features)
    scales = grouped.abs().amax(dim=1) / (ZERO - 1)  # [in/group, out]
    scales = scales.clamp_min(1e-10).to(torch.float16)
    q = torch.round(grouped / scales.to(torch.float32)[:, None, :]) + ZERO
    q = q.clamp(0, 2**BITS - 1).to(torch.int64).reshape(in_features, out_features)
    shifts = (torch.arange(PACK, device=q.device, dtype=torch.int64) * BITS)
    qweight = (q.reshape(in_features // PACK, PACK, out_features) << shifts[None, :, None]).sum(dim=1)
    zeros = torch.full(
        (in_features // group, out_features // PACK, PACK), ZERO - 1,
        dtype=torch.int64, device=q.device,
    )  # GPTQ v1 stores zero-1; ignored by vLLM for sym=True
    qzeros = (zeros << shifts[None, None, :]).sum(dim=2)
    g_idx = torch.arange(in_features, dtype=torch.int32, device=q.device) // group
    return {
        "qweight": _to_int32(qweight).cpu(),
        "qzeros": _to_int32(qzeros).cpu(),
        "scales": scales.cpu(),
        "g_idx": g_idx.cpu(),
    }


def dequantize(tensors: dict[str, torch.Tensor], group: int) -> torch.Tensor:
    """Reference dequantization back to a PyTorch [out, in] float32 weight."""
    qweight = tensors["qweight"].to(torch.int64) & 0xFFFFFFFF
    in_packed, out_features = qweight.shape
    shifts = torch.arange(PACK, dtype=torch.int64) * BITS
    q = (qweight[:, None, :] >> shifts[None, :, None]) & (2**BITS - 1)
    q = q.reshape(in_packed * PACK, out_features).to(torch.float32)
    scales = tensors["scales"].to(torch.float32).repeat_interleave(group, dim=0)
    return ((q - ZERO) * scales).t().contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--shard-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing {output}")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    output.mkdir(parents=True)

    weight_map: dict[str, str] = {}
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    shard_id = 0
    written: list[str] = []
    quantized = 0
    max_rel_error = 0.0

    def flush() -> None:
        nonlocal pending, pending_bytes, shard_id
        if not pending:
            return
        shard_id += 1
        name = f"model-w4a16-{shard_id:05d}.safetensors"
        save_file(pending, str(output / name), metadata={"format": "pt"})
        for key in pending:
            weight_map[key] = name
        written.append(name)
        pending, pending_bytes = {}, 0

    for shard in shards:
        with safe_open(str(source / shard), framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if LINEAR.match(key):
                    prefix = key[: -len(".weight")]
                    parts = quantize_linear(tensor.to(args.device), args.group_size)
                    if quantized % 36 == 0:
                        restored = dequantize(parts, args.group_size)
                        reference = tensor.to(torch.float32)
                        rel = float((restored - reference).norm() / reference.norm())
                        max_rel_error = max(max_rel_error, rel)
                        print(f"  {key}: relative Frobenius error {rel:.4f}", flush=True)
                    for suffix, value in parts.items():
                        pending[f"{prefix}.{suffix}"] = value
                        pending_bytes += value.numel() * value.element_size()
                    quantized += 1
                else:
                    pending[key] = tensor.contiguous()
                    pending_bytes += tensor.numel() * tensor.element_size()
                if pending_bytes >= args.shard_bytes:
                    flush()
    flush()

    total = 0
    for name in written:
        total += (output / name).stat().st_size
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n"
    )
    for item in source.iterdir():
        if item.name.endswith(".safetensors") or item.name in (
            "model.safetensors.index.json", "config.json"
        ):
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination, symlinks=True)
        else:
            shutil.copy2(item, destination)
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "gptq",
        "bits": BITS,
        "group_size": args.group_size,
        "desc_act": False,
        "sym": True,
        "lm_head": False,
        "checkpoint_format": "gptq",
        "remtp_note": "round-to-nearest proxy for kernel/speed tests; not calibrated GPTQ",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "quantized_linear_layers": quantized,
        "checked_max_relative_error": round(max_rel_error, 5),
        "shards": len(written),
        "bytes": total,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
