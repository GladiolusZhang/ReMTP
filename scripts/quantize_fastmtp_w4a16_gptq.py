#!/usr/bin/env python3
"""Layer-streamed GPTQ for the FastMTP target (W4A16, symmetric, group 128) on one Jetson.

Same output format as ``quantize_fastmtp_w4a16_rtn.py`` (GPTQ v1 tensors, loaded by
vLLM ``gptq_marlin``); only the 36 target decoder linear layers are quantized.
Embeddings, the target LM head, norms, biases and the physical MTP layer are copied
unchanged in BF16, so the draft stays exactly as before.

Algorithm: GPTQ (Frantar et al.) with true-sequential order inside each decoder layer
(q/k/v -> o -> gate/up -> down, every stage fed by the already-quantized weights),
per-group scale chosen by an MSE grid over clipping factors, 1% dampening, lazy batch
updates of 128 columns.  The calibration activations always come from the quantized
prefix of the network.  Only one decoder layer is resident in BF16 at a time.

Calibration and evaluation text come from files on this machine that are unrelated to
any benchmark: Python sources and package descriptions (``*.dist-info/METADATA``) in the
virtual environment.  Files are split into disjoint calibration / held-out sets by a
hash of their path; half of each set is wrapped in the chat template.

The same held-out tokens also run through four parallel streams -- BF16 reference,
this GPTQ model, the RTN proxy and an emulated bitsandbytes FP4 (block 64, the deployed
bnb configuration) -- and the script reports, per stream, KL(BF16 || stream) of the
next-token distribution at T=1 and T=0.6, top-1 agreement with BF16 and the NLL of the
actual next token.  That quantifies output-space quality without a benchmark run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_fastmtp_w4a16_rtn import LINEAR, PACK, ZERO, _to_int32  # noqa: E402

BITS = 4
QMIN, QMAX = -ZERO, ZERO - 1          # signed codes -8..7, stored as code + 8
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


# --------------------------------------------------------------------------- data
def local_corpus(root: Path, tokenizer, *, calib: int, evaluation: int, seq_len: int, seed: int):
    site = next((root / ".venv/lib").glob("python3*/site-packages"))
    code = sorted(p for p in site.rglob("*.py") if 4_000 <= p.stat().st_size <= 400_000)
    prose = sorted(p for p in site.glob("*.dist-info/METADATA") if p.stat().st_size >= 4_000)

    def held_out(path: Path) -> bool:
        return int(hashlib.sha256(str(path.relative_to(site)).encode()).hexdigest(), 16) % 8 == 0

    rng = random.Random(seed)

    def draw(pool: list[Path], count: int) -> list[torch.Tensor]:
        rng.shuffle(pool)
        out = []
        for index, path in enumerate(pool):
            if len(out) >= count:
                break
            text = path.read_text(errors="ignore")[:60_000]
            if index % 2:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Explain and continue the following material."},
                     {"role": "assistant", "content": text}],
                    tokenize=True, enable_thinking=False)
            else:
                ids = tokenizer(text)["input_ids"]
            if len(ids) < seq_len:
                continue
            start = rng.randrange(0, len(ids) - seq_len + 1) if index % 2 == 0 else 0
            out.append(torch.tensor(ids[start:start + seq_len], dtype=torch.int64))
        if len(out) < count:
            raise RuntimeError(f"only {len(out)} usable files for {count} samples")
        return out

    sets = {}
    for name, count, keep in (("calib", calib, False), ("eval", evaluation, True)):
        c = [p for p in code if held_out(p) == keep]
        d = [p for p in prose if held_out(p) == keep]
        sets[name] = torch.stack(draw(c, count - count // 2) + draw(d, count // 2))
    return sets["calib"], sets["eval"]


# ------------------------------------------------------------------------- model
class Weights:
    """Memory-mapped access to the source checkpoint."""

    def __init__(self, source: Path):
        self.source = source
        self.index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {}

    def get(self, key: str, device="cuda") -> torch.Tensor:
        shard = self.index[key]
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.source / shard), framework="pt")
        return self.handles[shard].get_tensor(key).to(device)


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.to(torch.float32)
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


def rotary(seq_len: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16)[None, None], emb.sin().to(torch.bfloat16)[None, None]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def attention_core(x_norm, w, cfg, cos, sin):
    """Normalized input -> attention output before o_proj."""
    b, t, _ = x_norm.shape
    hd, nh, nkv = cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    q = F.linear(x_norm, w["self_attn.q_proj"], w["self_attn.q_proj.bias"]).view(b, t, nh, hd).transpose(1, 2)
    k = F.linear(x_norm, w["self_attn.k_proj"], w["self_attn.k_proj.bias"]).view(b, t, nkv, hd).transpose(1, 2)
    v = F.linear(x_norm, w["self_attn.v_proj"], w["self_attn.v_proj.bias"]).view(b, t, nkv, hd).transpose(1, 2)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    k = k.repeat_interleave(nh // nkv, dim=1)
    v = v.repeat_interleave(nh // nkv, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).reshape(b, t, nh * hd)


def layer_forward(x, w, cfg, cos, sin):
    eps = cfg["rms_norm_eps"]
    attn = attention_core(rms(x, w["input_layernorm"], eps), w, cfg, cos, sin)
    h = x + F.linear(attn, w["self_attn.o_proj"])
    m = rms(h, w["post_attention_layernorm"], eps)
    return h + F.linear(F.silu(F.linear(m, w["mlp.gate_proj"])) * F.linear(m, w["mlp.up_proj"]), w["mlp.down_proj"])


# --------------------------------------------------------------------- quantizers
def _group_scales(block: torch.Tensor, grid: tuple[float, ...]) -> torch.Tensor:
    """Per-row symmetric scale for one [out, group] block, chosen by MSE over clip factors."""
    absmax = block.abs().amax(dim=1).clamp_min(1e-10)
    best_scale = (absmax / QMAX).to(torch.float16).to(torch.float32)
    best_err = torch.full_like(absmax, float("inf"))
    for factor in grid:
        scale = (absmax * factor / QMAX).to(torch.float16).to(torch.float32).clamp_min(1e-10)
        err = (block - scale[:, None] * torch.clamp(torch.round(block / scale[:, None]), QMIN, QMAX)).pow(2).sum(1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, scale, best_scale)
    return best_scale


def gptq(weight: torch.Tensor, hessian: torch.Tensor, *, group: int, damp: float, grid: tuple[float, ...]):
    """Return (codes [out,in] int8 signed, scales [in/group, out] fp32 (fp16-exact), dequantized [out,in])."""
    w = weight.to(torch.float32).clone()
    out_features, in_features = w.shape
    h = hessian.clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h.diagonal().add_(damp * torch.mean(torch.diag(h)))
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    codes = torch.zeros((out_features, in_features), dtype=torch.int8, device=w.device)
    scales = torch.zeros((in_features // group, out_features), dtype=torch.float32, device=w.device)
    for i1 in range(0, in_features, group):          # block size == group size
        i2 = i1 + group
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        h1 = hinv[i1:i2, i1:i2]
        scale = _group_scales(w1, grid)
        scales[i1 // group] = scale
        for i in range(group):
            col = w1[:, i]
            q = torch.clamp(torch.round(col / scale), QMIN, QMAX)
            codes[:, i1 + i] = q.to(torch.int8)
            e = (col - q * scale) / h1[i, i]
            w1[:, i:] -= e[:, None] * h1[i, i:][None, :]
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    deq = codes.to(torch.float32) * scales.repeat_interleave(group, dim=0).t()
    return codes, scales, deq


def rtn(weight: torch.Tensor, group: int) -> torch.Tensor:
    """Dequantized RTN proxy exactly as quantize_fastmtp_w4a16_rtn.py writes it."""
    w = weight.to(torch.float32)
    out_features, in_features = w.shape
    g = w.reshape(out_features, in_features // group, group)
    scale = (g.abs().amax(dim=2, keepdim=True) / QMAX).clamp_min(1e-10).to(torch.float16).to(torch.float32)
    return (torch.clamp(torch.round(g / scale), QMIN, QMAX) * scale).reshape(out_features, in_features)


def bnb_fp4(weight: torch.Tensor) -> torch.Tensor:
    import bitsandbytes.functional as bnbf
    packed, state = bnbf.quantize_4bit(weight.to(torch.bfloat16).contiguous(), blocksize=64, quant_type="fp4")
    return bnbf.dequantize_4bit(packed, state).to(torch.float32)


def pack_gptq(codes: torch.Tensor, scales: torch.Tensor, group: int) -> dict[str, torch.Tensor]:
    """codes [out,in] signed int8, scales [in/group,out] -> GPTQ v1 tensors (uint4b8, sym)."""
    out_features, in_features = codes.shape
    q = (codes.to(torch.int64) + ZERO).t().contiguous()                     # [in, out] 0..15
    shifts = torch.arange(PACK, device=q.device, dtype=torch.int64) * BITS
    qweight = (q.reshape(in_features // PACK, PACK, out_features) << shifts[None, :, None]).sum(dim=1)
    zeros = torch.full((in_features // group, out_features // PACK, PACK), ZERO - 1, dtype=torch.int64, device=q.device)
    qzeros = (zeros << shifts[None, None, :]).sum(dim=2)
    return {"qweight": _to_int32(qweight).cpu(), "qzeros": _to_int32(qzeros).cpu(),
            "scales": scales.to(torch.float16).cpu(),
            "g_idx": (torch.arange(in_features, dtype=torch.int32) // group)}


# --------------------------------------------------------------------------- main
def accumulate(hessian, x):
    flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
    hessian.addmm_(flat.t(), flat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=Path("models/FastMTP"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=8, help="samples per forward chunk")
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=None, help="debug: stop after N layers, write nothing")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing {output}")
    cfg = json.loads((source / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"] if args.layers is None else args.layers
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    calib, evaluation = local_corpus(root, tokenizer, calib=args.calib_samples, evaluation=args.eval_samples,
                                     seq_len=args.seq_len, seed=args.seed)
    print(f"calibration {tuple(calib.shape)}, held-out {tuple(evaluation.shape)}", flush=True)
    grid = tuple(1.0 - 0.025 * i for i in range(9))      # clip factors 1.0 .. 0.8
    weights = Weights(source)
    dev = "cuda"
    cos, sin = rotary(args.seq_len, cfg["head_dim"], cfg["rope_theta"], dev)

    embed = weights.get("model.embed_tokens.weight")
    x = F.embedding(calib.to(dev), embed)                              # calibration stream (quantized prefix)
    streams = {name: F.embedding(evaluation.to(dev), embed) for name in ("bf16", "gptq", "rtn", "bnb_fp4")}
    del embed
    torch.cuda.empty_cache()

    if args.layers is None:
        output.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    report = {"layers": []}
    started = time.time()
    chunks = range(0, x.shape[0], args.chunk)
    for layer in range(n_layers):
        t0 = time.time()
        prefix = f"model.layers.{layer}."
        w = {name: weights.get(prefix + name + ".weight") for name in LINEARS}
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            w[name + ".bias"] = weights.get(prefix + name + ".bias")
        for name in ("input_layernorm", "post_attention_layernorm"):
            w[name] = weights.get(prefix + name + ".weight")
        eps = cfg["rms_norm_eps"]
        wq = dict(w)                                                  # progressively quantized copy
        packed: dict[str, torch.Tensor] = {}
        errors = {}

        def quantize(names, hessian):
            for name in names:
                codes, scales, deq = gptq(w[name], hessian, group=args.group_size, damp=args.damp, grid=grid)
                ref = w[name].to(torch.float32)
                errors[name] = dict(
                    weight_rel=float((deq - ref).norm() / ref.norm()),
                    rtn_weight_rel=float((rtn(w[name], args.group_size) - ref).norm() / ref.norm()))
                wq[name] = deq.to(torch.bfloat16)
                for suffix, value in pack_gptq(codes, scales, args.group_size).items():
                    packed[f"{prefix}{name}.{suffix}"] = value
                del codes, scales, deq

        # Stage A: q/k/v share the attention-norm input.
        hessian = torch.zeros(cfg["hidden_size"], cfg["hidden_size"], device=dev)
        for s in chunks:
            accumulate(hessian, rms(x[s:s + args.chunk], w["input_layernorm"], eps))
        quantize(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), hessian / (x.shape[0] * x.shape[1]))
        # Stage B: o_proj input from quantized q/k/v.
        attn_out = torch.empty_like(x)
        hessian.zero_()
        for s in chunks:
            attn_out[s:s + args.chunk] = attention_core(rms(x[s:s + args.chunk], w["input_layernorm"], eps), wq, cfg, cos, sin)
            accumulate(hessian, attn_out[s:s + args.chunk])
        quantize(("self_attn.o_proj",), hessian / (x.shape[0] * x.shape[1]))
        # Stage C: gate/up share the MLP-norm input of the quantized attention block.
        hidden = torch.empty_like(x)
        hessian.zero_()
        for s in chunks:
            hidden[s:s + args.chunk] = x[s:s + args.chunk] + F.linear(attn_out[s:s + args.chunk], wq["self_attn.o_proj"])
            accumulate(hessian, rms(hidden[s:s + args.chunk], w["post_attention_layernorm"], eps))
        quantize(("mlp.gate_proj", "mlp.up_proj"), hessian / (x.shape[0] * x.shape[1]))
        del attn_out, hessian
        # Stage D: down_proj input from quantized gate/up.
        hessian = torch.zeros(cfg["intermediate_size"], cfg["intermediate_size"], device=dev)
        for s in chunks:
            m = rms(hidden[s:s + args.chunk], w["post_attention_layernorm"], eps)
            accumulate(hessian, F.silu(F.linear(m, wq["mlp.gate_proj"])) * F.linear(m, wq["mlp.up_proj"]))
        quantize(("mlp.down_proj",), hessian / (x.shape[0] * x.shape[1]))
        del hessian
        # Next layer's calibration input comes from the fully quantized layer.
        for s in chunks:
            m = rms(hidden[s:s + args.chunk], w["post_attention_layernorm"], eps)
            x[s:s + args.chunk] = hidden[s:s + args.chunk] + F.linear(
                F.silu(F.linear(m, wq["mlp.gate_proj"])) * F.linear(m, wq["mlp.up_proj"]), wq["mlp.down_proj"])
        del hidden
        # Held-out streams, each with its own weights.
        variants = {
            "bf16": w, "gptq": wq,
            "rtn": {**w, **{n: rtn(w[n], args.group_size).to(torch.bfloat16) for n in LINEARS}},
            "bnb_fp4": {**w, **{n: bnb_fp4(w[n]).to(torch.bfloat16) for n in LINEARS}},
        }
        for name, stream in streams.items():
            for s in range(0, stream.shape[0], args.chunk):
                stream[s:s + args.chunk] = layer_forward(stream[s:s + args.chunk], variants[name], cfg, cos, sin)
        del variants
        drift = {n: float((streams[n] - streams["bf16"]).float().norm() / streams["bf16"].float().norm())
                 for n in ("gptq", "rtn", "bnb_fp4")}
        report["layers"].append(dict(layer=layer, seconds=time.time() - t0, hidden_drift=drift, linears=errors))
        print(f"layer {layer:2d} {time.time() - t0:5.1f}s  hidden drift vs BF16: "
              + " ".join(f"{k}={v:.4f}" for k, v in drift.items())
              + f"  q_proj w-err gptq={errors['self_attn.q_proj']['weight_rel']:.3f} rtn={errors['self_attn.q_proj']['rtn_weight_rel']:.3f}",
              flush=True)
        if args.layers is None:
            name = f"model-gptq-layer-{layer:02d}.safetensors"
            save_file(packed, str(output / name), metadata={"format": "pt"})
            weight_map.update({key: name for key in packed})
        del w, wq, packed
        torch.cuda.empty_cache()

    # Output-space comparison on held-out text.
    del x
    torch.cuda.empty_cache()
    if args.layers is None or args.layers == cfg["num_hidden_layers"]:
        norm = weights.get("model.norm.weight")
        head = weights.get("lm_head.weight")
        targets = evaluation[:, 1:].to(dev)
        metrics = {n: dict(kl_t1=0.0, kl_t06=0.0, top1_agree=0.0, nll=0.0) for n in streams}
        count = 0
        for b in range(evaluation.shape[0]):
            logits = {n: F.linear(rms(streams[n][b:b + 1, :-1], norm, cfg["rms_norm_eps"]), head).float()[0]
                      for n in streams}
            ref1 = F.log_softmax(logits["bf16"], -1)
            ref06 = F.log_softmax(logits["bf16"] / 0.6, -1)
            top = logits["bf16"].argmax(-1)
            for n, value in logits.items():
                lp1 = F.log_softmax(value, -1)
                lp06 = F.log_softmax(value / 0.6, -1)
                metrics[n]["kl_t1"] += float((ref1.exp() * (ref1 - lp1)).sum())
                metrics[n]["kl_t06"] += float((ref06.exp() * (ref06 - lp06)).sum())
                metrics[n]["top1_agree"] += float((value.argmax(-1) == top).sum())
                metrics[n]["nll"] += float(-lp1.gather(-1, targets[b][:, None]).sum())
            count += targets.shape[1]
            del logits
        report["output_space"] = {n: {k: v / count for k, v in m.items()} for n, m in metrics.items()}
        report["held_out_tokens"] = count
        print(json.dumps(report["output_space"], indent=1), flush=True)

    if args.layers is not None:
        print("debug run: nothing written", flush=True)
        return 0
    # Copy every non-quantized tensor unchanged.
    rest: dict[str, torch.Tensor] = {}
    rest_bytes, rest_id = 0, 0

    def flush():
        nonlocal rest, rest_bytes, rest_id
        if rest:
            rest_id += 1
            name = f"model-rest-{rest_id:05d}.safetensors"
            save_file(rest, str(output / name), metadata={"format": "pt"})
            weight_map.update({key: name for key in rest})
            rest, rest_bytes = {}, 0

    for key in weights.index:
        if LINEAR.match(key):
            continue
        tensor = weights.get(key, device="cpu").contiguous()
        rest[key] = tensor
        rest_bytes += tensor.numel() * tensor.element_size()
        if rest_bytes >= 2 * 1024**3:
            flush()
    flush()
    total = sum((output / name).stat().st_size for name in set(weight_map.values()))
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")
    for item in source.iterdir():
        if item.name.endswith(".safetensors") or item.name in ("model.safetensors.index.json", "config.json"):
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination, symlinks=True)
        else:
            shutil.copy2(item, destination)
    cfg["quantization_config"] = {
        "quant_method": "gptq", "bits": BITS, "group_size": args.group_size, "desc_act": False, "sym": True,
        "lm_head": False, "checkpoint_format": "gptq",
        "remtp_note": (f"GPTQ, true-sequential, MSE clip grid, damp {args.damp}, {args.calib_samples}x{args.seq_len} "
                       "local calibration tokens (venv Python sources and package descriptions; no benchmark data)"),
    }
    (output / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    report.update(seconds=time.time() - started, calib_tokens=int(calib.numel()), output=str(output))
    (output / "remtp_gptq_report.json").write_text(json.dumps(report, indent=1))
    print(f"done in {time.time() - started:.0f}s -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
