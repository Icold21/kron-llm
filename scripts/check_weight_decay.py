"""How hard is weight decay actually pulling?  Measure, do not guess.

Compares the decay-group weight scale at initialisation with the 2,000-step
test checkpoint, and prints what pure weight decay alone would have done over
that many steps.  Also reports the learned sigma, which is what would have to
compensate if the factors were being shrunk too hard.
"""
import math
import sys

sys.path.insert(0, ".")
import torch

from kronllm.config import experiment_config
from kronllm.model import KronTransformer
from kronllm.train import build_param_groups

cfg = experiment_config("Q10")
tc = cfg["training"]
model = KronTransformer(cfg)
groups = {g["name"]: g for g in build_param_groups(model, tc)}
named = dict(model.named_parameters())

ckpt_path = "data/kaggle_out/q10/last.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt)
step = ckpt.get("step", "?")
print("checkpoint step", step)

decay_names = [n for n, p in named.items()
               if any(p is q for q in groups["decay"]["params"])]
factor_names = [n for n in decay_names if n.endswith(".B") or ".B" in n]
other = [n for n in decay_names if n not in factor_names]
print(f"decay group: {len(decay_names)} tensors, "
      f"{len(factor_names)} Kronecker factor tensors, {len(other)} other")


def stats(names, key=None):
    vals = []
    for n in names:
        w = state[n] if key is None else state[key]
        vals.append(w.float().abs().mean().item())
    return sum(vals) / len(vals)


init_state = {n: p.detach().clone() for n, p in named.items()}


def compare(names, label):
    a = torch.stack([init_state[n].float().abs().mean() for n in names]).mean()
    b = torch.stack([state[n].float().abs().mean() for n in names]).mean()
    print(f"{label:<22} init {a:.4f} -> now {b:.4f} "
          f"({100 * (b / a - 1):+.1f} %)")


compare(factor_names, "kron factors .B")
if other:
    compare(other, "other decayed")

lr = float(tc["lr"])
wd = float(tc["weight_decay"])
n_steps = int(step) if isinstance(step, int) else 2000
pure = math.exp(-lr * wd * n_steps)
print(f"pure weight decay over {n_steps} steps: x{pure:.3f} "
      f"({100 * (pure - 1):+.1f} %)")

sig = [(n, float(w.float().mean())) for n, w in state.items() if n.endswith(".s")]
if sig:
    print("sigma stats:", [(n, round(v, 3)) for n, v in sig[:6]], "...")
    vals = [v for _, v in sig]
    print(f"sigma mean {sum(vals) / len(vals):.3f}, "
          f"min {min(vals):.3f}, max {max(vals):.3f}")

theta = [w for n, w in state.items() if n.endswith(".theta")]
if theta:
    t = torch.cat([w.float().flatten() for w in theta])
    print(f"theta: mean |.| {t.abs().mean():.5f}, rms {t.pow(2).mean().sqrt():.5f}")
