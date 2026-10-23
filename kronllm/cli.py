"""Command line interface.

Examples
--------
    python -m kronllm.cli train --exp E2
    python -m kronllm.cli train --exp E2 --steps 300 --run-dir runs/smoke
    python -m kronllm.cli generate --run runs/E2 --prompt "Once upon a time"
    python -m kronllm.cli eval --run runs/E2
    python -m kronllm.cli analyze --run runs/E2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
def _load_model(run_dir: str | Path, checkpoint: str = "best.pt"):
    from kronllm.config import load_config
    from kronllm.model import KronTransformer
    from kronllm.quantization import dequantize_state_dict

    run_dir = Path(run_dir)
    path = run_dir / checkpoint
    if not path.exists():
        path = run_dir / "last.pt"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or load_config(run_dir / "config.yaml")
    if ck.get("quantized"):
        state = dequantize_state_dict(ck["state"], int(ck["bits"]))
    else:
        state = ck["model"]
    model = KronTransformer(cfg)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, ck


def _device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> None:
    from kronllm.config import experiment_config, load_config
    from kronllm.train import train

    if args.exp:
        cfg = experiment_config(args.exp, args.config, args.override)
    else:
        cfg = load_config(args.config, args.override)

    if args.steps is not None:
        cfg["training"]["max_steps"] = args.steps
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.no_amp:
        cfg["training"]["amp"] = "none"
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    if args.wandb:
        cfg["logging"]["wandb"] = True

    run_dir = Path(args.run_dir or (Path(cfg["training"]["out_dir"])
                                    / (args.exp or "custom")))
    results = train(cfg, run_dir, resume=args.resume)
    print(json.dumps({k: results[k] for k in
                      ("experiment", "n_params", "val_ppl", "best_val_ppl",
                       "val_bpb", "time_per_epoch_sec", "peak_vram_gb")}, indent=2))


def cmd_generate(args: argparse.Namespace) -> None:
    from kronllm.generate import generate_texts
    from kronllm.tokenizer import load_tokenizer

    model, cfg, _ = _load_model(args.run, args.checkpoint)
    device = _device(args.device)
    model.to(device)
    texts = generate_texts(
        model, [args.prompt], max_new_tokens=args.max_new,
        temperature=args.temperature, top_k=args.top_k,
        n_samples=args.samples, seed=args.seed, device=device,
        tokenizer=load_tokenizer(cfg))
    for text in texts:
        print("-" * 70)
        print(text)


def cmd_eval(args: argparse.Namespace) -> None:
    from kronllm.data import open_dataset
    from kronllm.diagnostics import (comparison_report, sigma_theta_report)
    from kronllm.train import diagnostics, evaluate

    model, cfg, ck = _load_model(args.run, args.checkpoint)
    device = _device(args.device)
    model.to(device)
    ds = open_dataset(cfg, "val")
    out: Dict[str, Any] = {"checkpoint": str(args.run), "step": ck.get("step"),
                           **comparison_report(model, int(cfg["data"]["seq_len"]))}
    out.update(evaluate(model, ds, 32, device, args.batches, cfg["training"]["amp"]))
    dg = diagnostics(model, ds, 32, device, spectral=True,
                     amp=cfg["training"]["amp"])
    out.update({k: v for k, v in dg.items() if isinstance(v, (int, float))})
    print(json.dumps(out, indent=2, default=float))


def cmd_analyze(args: argparse.Namespace) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import analyze as analyze_mod

    argv = ["--run", str(args.run), "--checkpoint", args.checkpoint,
            "--samples", str(args.samples)]
    if args.skip_generation:
        argv.append("--skip-generation")
    old = sys.argv
    try:
        sys.argv = ["analyze.py"] + argv
        analyze_mod.main()
    finally:
        sys.argv = old


# ---------------------------------------------------------------------------
def cmd_quantize(args: argparse.Namespace) -> None:
    """Quantize a checkpoint to int8/int4 and report the quality delta."""
    from kronllm.data import open_dataset
    from kronllm.quantization import quantize_checkpoint
    from kronllm.tokenizer import bits_per_byte, load_tokenizer
    from kronllm.train import evaluate

    run_dir = Path(args.run)
    model, cfg, _ = _load_model(run_dir, args.checkpoint)
    device = _device(args.device)
    model.to(device)
    dataset = open_dataset(cfg, "val")
    bpt = load_tokenizer(cfg).bytes_per_token
    amp = cfg["training"]["amp"]

    base = evaluate(model, dataset, args.batch_size, device, args.batches, amp)
    base_bpb = bits_per_byte(base["val_loss"], bpt)
    print(f"[fp32] loss {base['val_loss']:.4f} | bpb {base_bpb:.4f}")

    src = run_dir / args.checkpoint
    if not src.exists():
        src = run_dir / "last.pt"
    reports = []
    for bits in args.bits:
        destination = run_dir / f"model_int{bits}.pt"
        report = quantize_checkpoint(src, destination, bits=bits)
        quantized_model, _, _ = _load_model(run_dir, destination.name)
        quantized_model.to(device)
        result = evaluate(quantized_model, dataset, args.batch_size, device,
                          args.batches, amp)
        result_bpb = bits_per_byte(result["val_loss"], bpt)
        report.update({"val_loss": result["val_loss"], "val_bpb": result_bpb,
                       "delta_bpb": result_bpb - base_bpb})
        print(f"[int{bits}] loss {result['val_loss']:.4f} | bpb {result_bpb:.4f} "
              f"| delta {report['delta_bpb']:+.4f} | "
              f"{report['size_after_mb']:.2f} MB ({report['compression']}x)")
        reports.append(report)
        del quantized_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {"checkpoint": str(src), "fp32_loss": base["val_loss"],
               "fp32_bpb": base_bpb, "reports": reports}
    out = run_dir / "quantization.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
def cmd_budget(args: argparse.Namespace) -> None:
    """Print the parameter budget of the E-series (all models with ``--all``)."""
    from .config import EXPERIMENTS, experiment_config
    from .model import KronTransformer

    limit = int(args.limit)
    if args.all:
        ids = list(EXPERIMENTS)
    else:
        ids = [exp_id for exp_id in EXPERIMENTS if exp_id.startswith("E")]
    print(f"{'E':>4} {'total':>10} {'matrix':>10} {'<=1M':>6}  description")
    worst = 0
    for exp_id in ids:
        model = KronTransformer(experiment_config(exp_id))
        worst = max(worst, model.n_params)
        ok = "ok" if model.n_params <= limit else "FAIL"
        print(f"{exp_id:>4} {model.n_params:>10,} {model.n_params_matrix:>10,} "
              f"{ok:>6}  {EXPERIMENTS[exp_id]['desc']}")
    print(f"\nmax {worst:,} / {limit:,} ({worst / limit:.1%})")
    if worst > limit:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kronllm")
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="train one model / experiment")
    tr.add_argument("--exp", default=None, help="E0..E12")
    tr.add_argument("--config", default=None)
    tr.add_argument("--run-dir", default=None)
    tr.add_argument("--resume", default=None)
    tr.add_argument("--steps", type=int, default=None)
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--batch-size", type=int, default=None)
    tr.add_argument("--lr", type=float, default=None)
    tr.add_argument("--device", default=None)
    tr.add_argument("--data-dir", default=None)
    tr.add_argument("--no-amp", action="store_true")
    tr.add_argument("--wandb", action="store_true")
    tr.add_argument("--override", action="append", default=[])
    tr.set_defaults(func=cmd_train)

    ge = sub.add_parser("generate", help="sample from a checkpoint")
    ge.add_argument("--run", required=True)
    ge.add_argument("--checkpoint", default="best.pt")
    ge.add_argument("--prompt", default="Once upon a time")
    ge.add_argument("--max-new", type=int, default=200)
    ge.add_argument("--temperature", type=float, default=1.0)
    ge.add_argument("--top-k", type=int, default=8)
    ge.add_argument("--samples", type=int, default=1)
    ge.add_argument("--seed", type=int, default=None)
    ge.add_argument("--device", default=None)
    ge.set_defaults(func=cmd_generate)

    ev = sub.add_parser("eval", help="evaluate a checkpoint")
    ev.add_argument("--run", required=True)
    ev.add_argument("--checkpoint", default="best.pt")
    ev.add_argument("--batches", type=int, default=64)
    ev.add_argument("--device", default=None)
    ev.set_defaults(func=cmd_eval)

    an = sub.add_parser("analyze", help="spectra / heatmaps / samples")
    an.add_argument("--run", required=True)
    an.add_argument("--checkpoint", default="best.pt")
    an.add_argument("--samples", type=int, default=10)
    an.add_argument("--skip-generation", action="store_true")
    an.set_defaults(func=cmd_analyze)

    qt = sub.add_parser("quantize", help="int8/int4 checkpoint quantization")
    qt.add_argument("--run", required=True)
    qt.add_argument("--checkpoint", default="best.pt")
    qt.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    qt.add_argument("--batches", type=int, default=32)
    qt.add_argument("--batch-size", type=int, default=32)
    qt.add_argument("--device", default=None)
    qt.set_defaults(func=cmd_quantize)

    bd = sub.add_parser("budget", help="print the parameter budget of E0..E13")
    bd.add_argument("--limit", type=int, default=1_000_000)
    bd.add_argument("--all", action="store_true",
                    help="also print Q1-Q3, D1 and S1 (deliberately above 1M)")
    bd.set_defaults(func=cmd_budget)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
