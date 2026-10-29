"""Average neighbouring checkpoints and keep the pair that actually wins.

The rule (fixed 2026-10-29): average only *neighbouring* weights inside one
constant-LR phase.  Nothing is averaged across the anneal boundary -- the decay
pass fixes the final model, and mixing a coarse solution with a fine one
usually spoils both.

After a session there are two candidate pairs:

* ``best(n-1) + best(n)`` -- a whole session (~6-7k steps) apart, so the SGD
  noise in the two points is much more independent.  This is what SWA is meant
  to exploit, and it is the pair the owner proposed.
* ``best(n) + last(n)`` -- a few hundred steps apart.  This is the recipe that
  measured -2 % bpb on the 35,000 / 35,115 pair.

Both are scored on the same 1024 strided val blocks with the same alpha sweep,
and the pair with the lower bpb is written as the session's ``soup.pt``.  The
point is that the choice is measured, not assumed: at 115 steps apart the two
checkpoints are almost the same weights, so the second pair is the weaker
candidate in theory and has to win on numbers to be kept.

Usage::

    python scripts/session_soup.py --best data/kaggle_out/q10/best.pt \\
        --last data/kaggle_out/q10/last.pt \\
        --prev data/kaggle_out/archive/best_035000.pt \\
        --save-best data/kaggle_out/q10/soup.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_pair(a: Path, b: Path, label: str, blocks: int,
             workdir: Path) -> dict | None:
    """Score one pair with the full alpha sweep; return the best run."""
    out = workdir / f"{label}.json"
    ckpt = workdir / f"{label}.pt"
    cmd = [sys.executable, str(ROOT / "scripts" / "soup_q10.py"),
           "--a", str(a), "--b", str(b), "--blocks", str(blocks),
           "--out", str(out), "--save-best", str(ckpt)]
    print(f"[pair {label}] {a.name} + {b.name}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        print(f"[pair {label}] failed: {' | '.join(tail)}", flush=True)
        return None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("[alpha"):
            print(f"  {line}", flush=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    best = min(data["runs"], key=lambda r: r["bpb"])
    return {"label": label, "a": str(a), "b": str(b),
            "alpha": best["alpha"], "bpb": best["bpb"],
            "detail": best["detail"], "ckpt": str(ckpt),
            "rel_distance": data.get("rel_distance", {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="data/kaggle_out/q10/best.pt")
    ap.add_argument("--last", default="data/kaggle_out/q10/last.pt")
    ap.add_argument("--prev", default="",
                    help="best.pt of the previous session, if archived")
    ap.add_argument("--blocks", type=int, default=1024)
    ap.add_argument("--save-best", default="data/kaggle_out/q10/soup.pt")
    ap.add_argument("--out", default="data/session_soup.json")
    args = ap.parse_args()

    best, last = Path(args.best), Path(args.last)
    if not best.exists() or not last.exists():
        print(f"[skip] need {best} and {last}")
        return

    candidates = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        if args.prev and Path(args.prev).exists():
            got = run_pair(Path(args.prev), best, "best_prev+best", args.blocks,
                           workdir)
            if got:
                candidates.append(got)
        got = run_pair(best, last, "best+last", args.blocks, workdir)
        if got:
            candidates.append(got)
        if not candidates:
            print("[skip] no pair could be scored")
            return
        winner = min(candidates, key=lambda c: c["bpb"])
        Path(args.save_best).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(winner["ckpt"], args.save_best)

    report = {"winner": winner["label"], "alpha": winner["alpha"],
              "bpb": winner["bpb"], "candidates": candidates}
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\n{'pair':18s} {'alpha':>6s} {'bpb':>8s} {'en':>8s} {'ru':>8s}")
    for c in candidates:
        mark = "*" if c is winner else " "
        print(f"{mark}{c['label']:17s} {c['alpha']:6.2f} {c['bpb']:8.5f} "
              f"{c['detail']['en']['bpb']:8.5f} {c['detail']['ru']['bpb']:8.5f}")
    print(f"\n[winner] {winner['label']} alpha {winner['alpha']:.2f} "
          f"bpb {winner['bpb']:.5f} -> {args.save_best}")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
