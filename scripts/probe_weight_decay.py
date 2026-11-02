"""Is the weight decay over-shrinking the run?

AdamW shrinks every weight by ``1 - lr*wd`` per step, so over the 15,423 steps of
the decay pass at lr ~2.7e-4 and wd 0.1 the weights have been scaled by about
``exp(-0.1 * sum(lr))``.  Two measurements, no training:

1. **What actually happened to the norms** between the step-58,213 checkpoint and
   step 73,636 -- if decay dominates, matrix norms fall.
2. **Does the shrink cost loss?**  Multiply every weight matrix by a constant
   (RMSNorm makes this meaningful: the map, not the activations, scales) and
   measure val bpb.  If bpb falls at a scale above 1, the run wants bigger
   weights than the decay term allows, which is the "wd is hurting" hypothesis
   stated as a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\Users\Icold\Desktop\kron_llm")
sys.path.insert(0, str(ROOT))

from kronllm.model import KronTransformer      # noqa: E402
from kronllm.tokenizer import BPETokenizer     # noqa: E402

VAL = ROOT / "data" / "bilingual" / "val.bpe.bin"
TOK = ROOT / "data" / "bilingual" / "tokenizer.json"
NEW = Path(r"E:\kron_out\session9\q10a\last.pt")
OLD = Path(r"E:\kron_out\session7\q10\last.pt")

old = torch.load(OLD, map_location="cpu", weights_only=False)
new = torch.load(NEW, map_location="cpu", weights_only=False)
print(f"old step {old['step']:,}  new step {new['step']:,}")

groups: dict[str, list[float]] = {}
for k, v in new["model"].items():
    if v.dim() < 2:
        continue
    base = k.split(".")[0]
    a = float(old["model"][k].float().norm())
    b = float(v.float().norm())
    groups.setdefault(base, []).append(b / max(a, 1e-12))
print("\nratio of weight norms, new/old, by module prefix:")
for base, rs in sorted(groups.items()):
    print(f"  {base:12s} n={len(rs):3d}  mean {sum(rs)/len(rs):.3f}  "
          f"min {min(rs):.3f}  max {max(rs):.3f}")

# --- does a global rescale improve val? --------------------------------------
data = np.fromfile(VAL, dtype=np.uint16).astype(np.int64)
tok = BPETokenizer(TOK, 8192, bytes_per_token=3.69, eot_id=8191)
cfg = new["cfg"]
seq = 256
model = KronTransformer(cfg)
model.load_state_dict(new["model"])
model.eval()
blocks = [700_000 + i * 9_000 for i in range(24)]
basis = {k: v.clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def val_bpb() -> float:
    total, nbytes = 0.0, 0
    for s in blocks:
        ids = torch.from_numpy(data[s:s + seq + 1])[None]
        out = model(ids[:, :-1], ids[:, 1:])
        total += float(out["loss"]) * (seq - 1)
        nbytes += len(tok.decode(ids[0, 1:].tolist()).encode("utf-8"))
    return total / nbytes / 0.6931471805599453


print("\nscale   val_bpb")
for scale in (0.90, 1.00, 1.05, 1.10, 1.20):
    sd = {k: (v * scale if v.dim() >= 2 and k in basis else v)
          for k, v in basis.items()}
    model.load_state_dict(sd)
    print(f"  {scale:.2f}   {val_bpb():.5f}", flush=True)
model.load_state_dict(basis)
