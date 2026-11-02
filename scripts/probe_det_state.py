"""Volume state of the Kron compositions: sigma_min, geo-mean gain, rank.

For every (layer, role) it materialises ``P = W_L ... W_1`` -- the map the block
actually applies -- and reports

* ``geo``   -- ``|det P|^(1/d)``, the geometric-mean gain.  Above 1 the map
  expands volume, below 1 it contracts.
* ``smin``  -- the smallest singular value: the direction that is closest to
  collapsed.  A map can expand on balance and still have lost a direction.
* ``rank``  -- effective rank (``exp(entropy(sigma))``) over ``d``.

The A/B needs this because bpb alone cannot say whether a penalty did what it
claims: a penalty that never moved ``smin`` is inert no matter how the loss
looked.

Usage::

    python scripts/probe_det_state.py --ckpt runs/AB_A_base_s42/best.pt \
        --label base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kronllm.diagnostics import ALL_ROLES, _module   # noqa: E402
from kronllm.model import KronTransformer            # noqa: E402
from kronllm.tokenizer import BPETokenizer           # noqa: E402


def geo(sv: torch.Tensor) -> float:
    return float(torch.exp(torch.log(sv.clamp_min(1e-12)).mean()))


def eff_rank(sv: torch.Tensor) -> float:
    p = sv.clamp_min(1e-12)
    p = p / p.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--tokenizer", default="data/bilingual/tokenizer.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"])
    model.load_state_dict(state["model"])
    model.eval()
    # Byte-level bench configs and the 8K BPE run alike: load_tokenizer picks
    # the right one from the config, so the probe never guesses a vocabulary.
    from kronllm.tokenizer import load_tokenizer
    tok = load_tokenizer(state["cfg"])
    ids = torch.tensor([tok.encode("The Roman Empire was one of the largest.")[:64]])

    with torch.no_grad():
        out = model(ids, collect=True)

    rows = []
    for i, entry in enumerate(out["trace"]):
        block = model.blocks[i]
        for role in ALL_ROLES:
            mod = _module(block, role)
            D = entry["D"].get(role)
            if mod is None or D is None:
                continue
            P = mod.compose(D).float()
            sv = torch.linalg.svdvals(P)
            rows.append({"layer": i, "role": role, "geo": geo(sv),
                         "smin": float(sv[-1]), "smax": float(sv[0]),
                         "rank": eff_rank(sv), "d": int(P.shape[0])})

    summary = {"label": args.label or Path(args.ckpt).parent.name,
               "step": int(state.get("step") or 0), "pairs": len(rows),
               "geo_mean": sum(r["geo"] for r in rows) / len(rows),
               "smin_mean": sum(r["smin"] for r in rows) / len(rows),
               "rank_frac": sum(r["rank"] / r["d"] for r in rows) / len(rows),
               "geo_under_1": sum(1 for r in rows if r["geo"] < 1.0),
               "smin_under_1": sum(1 for r in rows if r["smin"] < 1.0)}
    print(json.dumps(summary))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows},
                                             indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
