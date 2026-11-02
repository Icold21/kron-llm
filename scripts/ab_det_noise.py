"""Seed-noise floor for the determinant A/B: the same base config, seed 43.

The arms differ by +0.003 to +0.011 bpb at 1,500 steps.  Whether that is an
effect or noise is decided by re-running one arm unchanged on a second seed and
looking at the spread -- there is no other way to know, and quoting the
differences without it would be dishonest.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Icold\Desktop\kron_llm")
run_dir = ROOT / "runs" / "AB_A_base_s43"
cmd = [sys.executable, "-u", "-m", "kronllm.cli", "train",
       "--exp", "L1", "--run-dir", str(run_dir), "--steps", "1500",
       "--override", "logging.tensorboard=false", "--override", "seed=43",
       "--override", "training.batch_size=32",
       "--override", "training.warmup_steps=100"]
t0 = time.time()
r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
(run_dir / "train.log").write_text((r.stdout or "") + (r.stderr or ""),
                                   encoding="utf-8")
res = run_dir / "results.json"
bpb = json.loads(res.read_text(encoding="utf-8"))["val_bpb"] if res.exists() else None
print(f"A_base seed 43  bpb {bpb}  ({time.time() - t0:.0f} s, exit {r.returncode})")
(ROOT / "runs" / "AB_noise.json").write_text(
    json.dumps({"arm": "A_base", "seed": 43, "val_bpb": bpb}, indent=2) + "\n",
    encoding="utf-8")
