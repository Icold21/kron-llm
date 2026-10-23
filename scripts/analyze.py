"""Post-hoc analysis (section 7): spectra, sigma(theta) heatmaps, generation.

Usage
-----
    python scripts/analyze.py --run runs/E2
    python scripts/analyze.py --run runs/E2 --checkpoint best.pt --samples 10
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/analyze.py` from a fresh checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from kronllm.config import load_config
from kronllm.data import open_dataset
from kronllm.diagnostics import (eff_rank_heatmap, sigma_heatmap,
                                 sigma_theta_report, spectral_report,
                                 spearman)
from kronllm.model import KronTransformer
from kronllm.tokenizer import load_tokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_run(run_dir: Path, ckpt: str = "best.pt"):
    path = run_dir / ckpt
    if not path.exists():
        path = run_dir / "last.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or load_config(run_dir / "config.yaml")
    model = KronTransformer(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def heatmap_plots(model: KronTransformer, cfg: Dict[str, Any], out: Path) -> Dict[str, Any]:
    ds = open_dataset(cfg, "val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x, y = next(iter(ds.iter_blocks(4, device)))
    with torch.no_grad():
        res = model(x, y, collect=True)

    spec = spectral_report(model, res["trace"])
    sigma = sigma_theta_report(model)
    out.mkdir(parents=True, exist_ok=True)

    roles = [r for r in ("q", "k", "v", "o", "f", "fg")
             if any(row["role"] == r for row in spec["rows"])]
    fig, axes = plt.subplots(2, len(roles), figsize=(4 * len(roles), 7),
                             squeeze=False)
    for j, role in enumerate(roles):
        er = eff_rank_heatmap(spec, role)
        if er.numel():
            ax = axes[0][j]
            im = ax.imshow(er.numpy(), aspect="auto", origin="lower",
                           cmap="viridis", vmin=0, vmax=model.d)
            ax.set_title(f"eff. rank D[{role}]")
            ax.set_xlabel("channel l")
            ax.set_ylabel("layer i")
            fig.colorbar(im, ax=ax)
        sg = sigma_heatmap(sigma, role)
        if sg.numel():
            ax = axes[1][j]
            im = ax.imshow(sg.numpy(), aspect="auto", origin="lower",
                           cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"sigma(theta) [{role}]")
            ax.set_xlabel("channel l")
            ax.set_ylabel("layer i")
            fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png", dpi=140)
    plt.close(fig)

    # sigma(theta) distribution: H3 (fast vs slow memory channels)
    flat = [v for rows in sigma["sigma_theta_per_type"].values()
            for row in rows for v in row]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(flat, bins=24, range=(0, 1))
    ax.set_xlabel("sigma(theta)")
    ax.set_ylabel("count")
    ax.set_title(f"H3: var_l = {sigma['sigma_theta_var']:.4f}")
    fig.tight_layout()
    fig.savefig(out / "sigma_theta_hist.png", dpi=140)
    plt.close(fig)

    # H2: does the expressivity of the innovation grow with depth?
    summary0 = spec["summary"]
    per_layer_d = summary0.get("eff_rank_per_layer", [])
    per_layer_e = summary0.get("eff_rank_excess_per_layer", [])
    if per_layer_d:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(per_layer_d, marker="o", label="eff.rank(D)  (innovation)")
        ax.plot(per_layer_e, marker="s", label="eff.rank(P - s^L I)")
        ax.axhline(model.d, ls="--", c="grey", lw=1, label="d")
        ax.set_xlabel("layer i")
        ax.set_ylabel("effective rank")
        ax.set_title(f"H2: spearman(rank, depth) = "
                     f"{summary0.get('eff_rank_spearman', float('nan')):.2f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "rank_vs_depth.png", dpi=140)
        plt.close(fig)

    # spectrum of the innovation D (channel 0) at a few depths
    fig, ax = plt.subplots(figsize=(6, 3.5))
    n_layers = len(res["trace"])
    for i in (0, n_layers // 2, n_layers - 1):
        D = res["trace"][i]["D"].get("q")
        if D is None:
            continue
        sv = torch.linalg.svdvals(D[0].float()).cpu().numpy()
        ax.plot(sv, label=f"layer {i}")
    ax.set_yscale("log")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.set_title("spectrum of D[q, channel 0]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "spectra.png", dpi=140)
    plt.close(fig)

    # learned innovation scale tau and ReZero gate gamma, per layer
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for name, ax in (("tau", axes[0]), ("gate", axes[1])):
        for role in roles:
            vals = [row[name] for row in spec["rows"]
                    if row["role"] == role and row["channel"] == 0]
            if vals:
                ax.plot(vals, marker="o", label=role)
        ax.set_title(f"learned {name}")
        ax.set_xlabel("layer i")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "scales.png", dpi=140)
    plt.close(fig)


    summary = dict(spec["summary"])
    summary.update({k: v for k, v in sigma.items() if not isinstance(v, dict)})
    (out / "analysis.json").write_text(
        json.dumps(summary, indent=2, default=float, ensure_ascii=True),
        encoding="utf-8")
    return summary


def worst_tokens(model, cfg, out: Path, device: torch.device,
                 n_batches: int = 16, top: int = 20) -> None:
    ds = open_dataset(cfg, "val")
    tok = load_tokenizer(cfg)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for i, (x, y) in enumerate(ds.iter_blocks(16, device)):
            if i >= n_batches:
                break
            logits = model(x)["logits"].float()
            ls = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1),
                reduction="none").view(y.shape).cpu()
            for b in range(x.shape[0]):
                ids = x[b].cpu().tolist()
                text = tok.decode(ids)
                for t in range(y.shape[1]):
                    rows.append({
                        "loss": float(ls[b, t]),
                        "block": i * x.shape[0] + b,
                        "pos": t,
                        "context": text[max(0, t - 48):t],
                        "next_token": tok.decode([int(y[b, t])]),
                        "next_id": int(y[b, t]),
                    })
    rows.sort(key=lambda r: -r["loss"])
    (out / "worst_tokens.json").write_text(
        json.dumps(rows[:top], indent=2, ensure_ascii=True), encoding="utf-8")


@torch.no_grad()
def generate_samples(model, cfg, out: Path, device: torch.device,
                     n: int = 10, max_new_tokens: int = 150) -> List[str]:
    prompts = ["Once upon a time", "The little cat", "Lily and Tom",
               "One day, a dog", "The sun was"]
    tok = load_tokenizer(cfg)
    texts = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        idx = torch.tensor([tok.encode(prompt)],
                           dtype=torch.long, device=device)
        out_ids = model.generate(idx, max_new_tokens=max_new_tokens,
                                 temperature=1.0, top_k=8, stop_at_eot=True)
        text = tok.decode(out_ids[0].tolist())
        texts.append(text)
        print("-" * 60)
        print(text[:400])
    (out / "generation.txt").write_text("\n\n" + ("=" * 70 + "\n\n").join(texts),
                                        encoding="utf-8")
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run)
    model, cfg, ck = load_run(run_dir, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    out = run_dir / "analysis"
    summary = heatmap_plots(model, cfg, out)
    print(json.dumps(summary, indent=2, default=float)[:3000])

    if (Path(cfg["data"].get("data_dir", "data")) / "val.bin").exists() \
            or (Path(cfg["data"].get("data_dir", "data")) / "val.bpe.bin").exists():
        worst_tokens(model, cfg, out, device)
        if not args.skip_generation:
            generate_samples(model, cfg, out, device, n=args.samples)


if __name__ == "__main__":
    main()
