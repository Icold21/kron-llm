"""Run the E-series experiments (section 6.3) and collect a comparison table.

Usage
-----
    python scripts/run_experiments.py --exp E2 --steps 300        # smoke run
    python scripts/run_experiments.py --exp E0 E2 --epochs 3
    python scripts/run_experiments.py --all
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/run_experiments.py` from a fresh checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import time
from typing import Any, Dict, List

from kronllm.config import EXPERIMENTS, experiment_config, load_config
from kronllm.train import train

RESULT_COLS = [
    "experiment", "desc", "n_params", "n_params_matrix", "param_ratio_vs_E0",
    "flops_per_token", "val_loss", "val_ppl", "best_val_ppl", "val_bpb",
    "bytes_per_token", "vocab_size", "tokenizer",
    "time_per_epoch_sec", "sec_per_step", "peak_vram_gb", "steps",
    "sigma_theta_mean", "sigma_theta_var", "attn_entropy_mean",
    "q_drift_mean", "eff_rank_W", "eff_rank_D", "rank_excess",
    "eff_rank_P", "eff_rank_excess", "eff_rank_spearman",
    "eff_rank_excess_spearman", "eff_rank_growth", "D_rms", "tau_mean",
    "gate_mean", "sigma_min_W", "cond_max", "h_norm_ratio_mean",
]


def append_summary(results_dir: Path, result: Dict[str, Any]) -> None:
    path = results_dir / "summary.csv"
    exists = path.exists()
    row = {k: result.get(k) for k in RESULT_COLS}
    with open(path, "a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=RESULT_COLS)
        if not exists:
            wr.writeheader()
        wr.writerow(row)


def load_ratios(results_dir: Path) -> Dict[str, Any]:
    path = results_dir / "summary.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["experiment"]] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="cap optimizer steps (smoke runs)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--override", action="append", default=[],
                    help="extra dotted key=value override for every run")
    ap.add_argument("--rebuild-summary", action="store_true",
                    help="only rewrite results/summary.csv from results/E*.json")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if args.rebuild_summary:
        print(f"rewritten {rebuild_summary(results_dir)}")
        return
    ids: List[str] = (
        [e for e in EXPERIMENTS if e.startswith("E") and e[1:].isdigit()]
        if args.all else (args.exp or ["E2"]))
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    base = None
    for exp_id in ids:
        cfg = experiment_config(exp_id, args.config, args.override)
        if args.epochs is not None:
            cfg["training"]["epochs"] = args.epochs
        if args.steps is not None:
            cfg["training"]["max_steps"] = args.steps
        if args.batch_size is not None:
            cfg["training"]["batch_size"] = args.batch_size
        run_dir = Path(args.runs_dir or cfg["training"]["out_dir"]) / exp_id

        print("=" * 72)
        print(f"[{exp_id}] {EXPERIMENTS[exp_id]['desc']}")
        print("=" * 72)
        try:
            result = train(cfg, run_dir)
        except Exception as exc:                      # keep the series alive
            import traceback
            traceback.print_exc()
            print(f"[{exp_id}] FAILED: {type(exc).__name__}: {exc} -- "
                  f"continuing with the next experiment")
            continue
        result["time_started"] = t0

        summary = load_ratios(results_dir)
        base_ppl = None
        if "E0" in summary and summary["E0"].get("val_ppl"):
            try:
                base_ppl = float(summary["E0"]["val_ppl"])
            except ValueError:
                base_ppl = None
        if base_ppl is None and result.get("val_ppl"):
            base_ppl = float(result["val_ppl"])
        result["ppl_ratio_vs_E0"] = (float(result["val_ppl"]) / base_ppl
                                     if base_ppl else None)
        e0 = summary.get("E0") or {}
        try:
            result["param_ratio_vs_E0"] = (result["n_params_matrix"]
                                           / float(e0["n_params_matrix"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            result["param_ratio_vs_E0"] = None
        append_summary(results_dir, result)
        with open(results_dir / f"{exp_id}.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    rebuild_summary(results_dir)
    print(f"[all done] {time.time() - t0:.1f}s -> {results_dir/'summary.csv'}")


def rebuild_summary(results_dir: Path) -> Path:
    """Rewrite ``summary.csv`` from ``results/E*.json`` with fresh ratios.

    Ratios depend on E0, which may be re-run (or run after the others), so the
    table is always regenerated from the per-experiment JSON rather than
    appended to.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for path in results_dir.glob("E*.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    order = {e: i for i, e in enumerate(EXPERIMENTS)}
    rows.sort(key=lambda r: order.get(str(r.get("experiment")), 999))

    e0 = next((r for r in rows if r.get("experiment") == "E0"), None)
    base_ppl = None
    if e0 and e0.get("val_ppl"):
        base_ppl = float(e0["val_ppl"])
    for r in rows:
        ppl = r.get("val_ppl")
        r["ppl_ratio_vs_E0"] = (float(ppl) / base_ppl
                                if (base_ppl and ppl) else None)
        try:
            r["param_ratio_vs_E0"] = (float(r["n_params_matrix"])
                                      / float(e0["n_params_matrix"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            r["param_ratio_vs_E0"] = None

    path = results_dir / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=RESULT_COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in RESULT_COLS})
    return path


if __name__ == "__main__":
    main()
