"""Summarise the depth-recurrence ablation (E8r, E27-E30) as a Markdown table.

Usage::

    python scripts/ablation_table.py E8r E27 E28 E29 E30
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(run: str) -> dict:
    base = Path("runs") / run
    res = json.loads((base / "results.json").read_text(encoding="utf-8"))
    diag: dict = {}
    path = base / "diagnostics.json"
    if path.exists():
        diag = json.loads(path.read_text(encoding="utf-8")).get("final", {})
    cfg: dict = {}
    cfg_path = base / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
    return {"run": run, "res": res, "diag": diag, "cfg": cfg}


def main(argv: list[str]) -> int:
    runs = argv[1:] or ["E8r", "E27", "E28", "E29", "E30"]
    rows = [load(r) for r in runs]
    print("| run | sigma | K_win | params | bpb | delta | PPL | eff_rank(D) | eff_rank(P) |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    base = rows[0]["res"].get("val_bpb") if rows else None
    for r in rows:
        res, diag = r["res"], r["diag"]
        kc = (r["cfg"].get("model") or {}).get("kron") or {}
        bpb = res.get("val_bpb")
        delta = f"{100 * (bpb - base) / base:+.1f} %" if base and bpb else ""
        print(f"| {r['run']} | {kc.get('sigma_mode', 'learned')} | "
              f"{kc.get('K_win', 2)} | {res.get('n_params', 0):,} | "
              f"{bpb:.4f} | {delta} | {res.get('val_ppl', 0):.3f} | "
              f"{diag.get('spec_eff_rank_D', 0):.1f} | "
              f"{diag.get('spec_eff_rank_P', 0):.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
