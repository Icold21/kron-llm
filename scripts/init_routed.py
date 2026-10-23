"""Warm-start a routed embedding from a factorised checkpoint.

The routed variant (Q3) has a different embedding parameterisation from Q2:
instead of ``E @ U`` it stores a dense bank and a top-k router.  This script
copies every shared transformer tensor from Q2 and initialises the routed table
so that its effective embeddings match Q2's factorised table.  The resulting
checkpoint can be fine-tuned with ``training.init_from``.

Example:
    python scripts/init_routed.py \
        --source runs/Q2/best.pt \
        --out runs/Q3warm/init.pt

The router is initialised with nearly orthogonal token codes.  The scale is
large enough that each token's own bank row dominates its top-k mixture; the
effective table then starts within ~1e-3 cosine similarity of Q2's table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/init_routed.py` from a fresh checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kronllm.config import experiment_config
from kronllm.model import KronTransformer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="runs/Q2/best.pt",
                    help="factorised checkpoint to copy the transformer from")
    ap.add_argument("--out", default="runs/Q3warm/init.pt",
                    help="output checkpoint path")
    ap.add_argument("--target-exp", default="Q3",
                    help="experiment whose routed config is used")
    ap.add_argument("--scale", type=float, default=2000.0,
                    help="router logit scale; larger means a sharper mixture")
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def main() -> None:
    """Build and save the warm-started routed checkpoint."""
    args = parse_args()
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    src_cfg = source["cfg"]
    src = KronTransformer(src_cfg)
    src.load_state_dict(source["model"])
    src.eval()

    cfg = experiment_config(args.target_exp)
    dst = KronTransformer(cfg)
    dst.eval()
    if dst.routed is None:
        raise SystemExit(f"{args.target_exp} does not use embed_mode='routed'")
    if src.embed_up is None:
        raise SystemExit("source checkpoint has no factorised embedding")

    # Copy every shared tensor (blocks, norms, ...).  Embedding and head keys
    # intentionally differ and are skipped.
    dst_state = dst.state_dict()
    copied = 0
    for name, value in src.state_dict().items():
        if name in dst_state and dst_state[name].shape == value.shape:
            dst_state[name] = value.clone()
            copied += 1
    dst.load_state_dict(dst_state)

    with torch.no_grad():
        table = src.embed.weight @ src.embed_up.weight.t()   # (V, d)
        vocab, d_model = table.shape
        v_eff = dst.routed.v_eff
        route_dim = dst.routed.route_dim
        gen = torch.Generator().manual_seed(args.seed)

        codes = torch.randn(vocab, route_dim, generator=gen)
        codes = codes / codes.norm(dim=1, keepdim=True)
        all_codes = torch.randn(v_eff, route_dim, generator=gen)
        all_codes[:vocab] = codes
        all_codes = all_codes / all_codes.norm(dim=1, keepdim=True)

        bank = torch.empty(v_eff, d_model)
        bank[:vocab] = table
        bank[vocab:] = torch.randn(v_eff - vocab, d_model, generator=gen) * table.std()

        dst.routed.route.weight.copy_(codes)
        dst.routed.route_proj.weight.copy_(all_codes * args.scale)
        dst.routed.bank.weight.copy_(bank)
        dst._eff_cache = None

        effective = dst.routed.effective_embeddings()
        cosine = torch.nn.functional.cosine_similarity(effective, table, dim=-1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dst.state_dict(), "cfg": cfg, "step": 0,
                "init_from": str(Path(args.source))}, out)
    print(f"copied {copied} shared tensors from {args.source}")
    print(f"effective-table cosine: mean {cosine.mean():.6f} "
          f"min {cosine.min():.6f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
