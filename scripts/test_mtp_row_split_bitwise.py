"""Plan A feasibility: is an M-row MTP pass bitwise equal to a 1-row + (M-1)-row pass?

Standalone (no vLLM): real BF16 MTP-layer and LM-head weights, every GEMM of the
layer fed with realistic chained activations.  For M = 2..10 the M-row result is
compared with the concatenation of a 1-row and an (M-1)-row call.
Usage: python scripts/test_mtp_row_split_bitwise.py [model_dir] [trials]
"""
import glob, sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

model = sys.argv[1] if len(sys.argv) > 1 else "models/FastMTP-W4A16-GPTQ"
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False  # as the server
dev = "cuda"
W = {}
for f in sorted(glob.glob(f"{model}/*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if "mtp_layers.0" in k or k in ("lm_head.weight", "model.embed_tokens.weight"):
                W[k.replace("model.mtp_layers.0.", "")] = s.get_tensor(k).to(dev)
qkv_w = torch.cat([W[f"self_attn.{n}_proj.weight"] for n in "qkv"])
qkv_b = torch.cat([W[f"self_attn.{n}_proj.bias"] for n in "qkv"])


def rms(x, w):
    v = x.float()
    return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * w


def split(fn, x):
    return torch.cat((fn(x[:1]), fn(x[1:])))


stats = {}
g = torch.Generator(device=dev).manual_seed(0)
print(torch.cuda.get_device_name(0), torch.__version__)
for trial in range(trials):
    for m in range(2, 11):
        ids = torch.randint(0, 151680, (m,), device=dev, generator=g)
        hidden = (torch.randn(m, 4096, device=dev, generator=g) * 3).bfloat16()
        emb = W["model.embed_tokens.weight"][ids]
        cat = torch.cat((rms(emb, W["token_layernorm.weight"]), rms(hidden, W["hidden_layernorm.weight"])), -1)
        stages = [("input_proj", lambda x: F.linear(x, W["input_proj.weight"]), cat)]
        x = stages[0][1](cat)
        n1 = rms(x, W["input_layernorm.weight"])
        stages.append(("qkv", lambda t: F.linear(t, qkv_w, qkv_b), n1))
        qkv = stages[-1][1](n1)
        attn = qkv[:, :4096]                      # stand-in attention output with real statistics
        stages.append(("o_proj", lambda t: F.linear(t, W["self_attn.o_proj.weight"]), attn))
        x2 = x + stages[-1][1](attn)
        n2 = rms(x2, W["post_attention_layernorm.weight"])
        stages.append(("gate_proj", lambda t: F.linear(t, W["mlp.gate_proj.weight"]), n2))
        stages.append(("up_proj", lambda t: F.linear(t, W["mlp.up_proj.weight"]), n2))
        act = F.silu(stages[-2][1](n2)) * stages[-1][1](n2)
        stages.append(("down_proj", lambda t: F.linear(t, W["mlp.down_proj.weight"]), act))
        out = rms(x2 + stages[-1][1](act), W["final_layernorm.weight"])
        stages.append(("lm_head", lambda t: F.linear(t, W["lm_head.weight"]), out))
        for name, fn, inp in stages:
            full, parts = fn(inp), split(fn, inp)
            s = stats.setdefault(name, dict(cases=0, equal=0, maxdiff=0.0, argmax_flip=0, tv=0.0))
            s["cases"] += 1
            s["equal"] += int(torch.equal(full, parts))
            s["maxdiff"] = max(s["maxdiff"], float((full.float() - parts.float()).abs().max()))
            if name == "lm_head":
                p, q = torch.softmax(full.float() / 0.6, -1), torch.softmax(parts.float() / 0.6, -1)
                s["argmax_flip"] += int((p.argmax(-1) != q.argmax(-1)).sum())
                s["tv"] = max(s["tv"], float((p - q).abs().sum(-1).max() / 2))
for name, s in stats.items():
    extra = f" argmax_flips={s['argmax_flip']} max_TV(T=0.6)={s['tv']:.2e}" if name == "lm_head" else ""
    print(f"{name:10s} bitwise-equal {s['equal']}/{s['cases']}  max|diff|={s['maxdiff']:.3e}{extra}")
