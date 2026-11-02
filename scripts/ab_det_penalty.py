"""Fast A/B on the TinyStories bench (L1 shape: d=128, 12 layers, batch 64).

Arms, all at 11,000 steps (one bench epoch), two seeds each:

* ``A_base``   -- the L1 recipe, no penalty.  This is the reference the other
  arms are compared to; it reproduces the historical L1 number, batch 64.
* ``B_logdet`` -- ``relu(-log|det P|)``: the literal "penalise the determinant
  below 1" reading, coefficient 0.1.
* ``C_smin``   -- ``relu(1 - sigma_min(P))``: charge the direction that has
  actually collapsed, coefficient 0.1.
* ``D_smin1``  -- the same at coefficient 1.0, to see whether it is a knob or a
  wall.
* ``E_wd001``  -- no penalty, ``weight_decay`` 0.1 -> 0.01: the question "is the
  decay term hurting the run".

Two seeds because a 0.005 bpb difference between arms is noise on one seed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Icold\Desktop\kron_llm")
STEPS = 1_500
SEEDS = (42,)

ARMS = [
    ("A_base", []),
    ("B_logdet", ["model.det_penalty=0.1", "model.det_penalty_mode=logdet"]),
    ("C_smin", ["model.det_penalty=0.1", "model.det_penalty_mode=sigma_min"]),
    ("D_smin1", ["model.det_penalty=1.0", "model.det_penalty_mode=sigma_min"]),
    ("E_wd001", ["training.weight_decay=0.01"]),
]

rows = []
for seed in SEEDS:
    for name, over in ARMS:
        run_dir = ROOT / "runs" / f"AB_{name}_s{seed}"
        cmd = [sys.executable, "-u", "-m", "kronllm.cli", "train",
               "--exp", "L1", "--run-dir", str(run_dir),
               "--steps", str(STEPS),
               "--override", "logging.tensorboard=false",
               "--override", f"seed={seed}",
               # batch 32 and a 100-step warmup: at 1,500 steps a 500-step
               # warmup would spend a third of the run below full LR, which is
               # plenty of room for an arm to "win" on the schedule instead of
               # on the penalty.  Every arm shares both.
               "--override", "training.batch_size=32",
               "--override", "training.warmup_steps=100"]
        for o in over:
            cmd += ["--override", o]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        log = run_dir / "train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
        res = run_dir / "results.json"
        bpb = None
        if res.exists():
            bpb = json.loads(res.read_text(encoding="utf-8")).get("val_bpb")
        rows.append({"arm": name, "seed": seed, "val_bpb": bpb,
                     "exit": r.returncode, "sec": round(time.time() - t0, 1)})
        print(f"{name:10s} seed {seed}  bpb {bpb if bpb is None else round(bpb, 4)}"
              f"  ({rows[-1]['sec']:.0f} s, exit {r.returncode})", flush=True)
        (ROOT / "runs" / "AB_summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8")

print("\n=== summary (mean over seeds) ===")
for name, _ in ARMS:
    v = [r["val_bpb"] for r in rows if r["arm"] == name and r["val_bpb"]]
    if v:
        print(f"  {name:10s} {sum(v)/len(v):.4f}  (n={len(v)}, "
              f"spread {max(v)-min(v):.4f})")
