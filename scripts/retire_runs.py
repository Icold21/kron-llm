"""Record every run's numbers, then delete the weights.

`runs/` is git-ignored, so `results.json` inside it is the only copy of a run's
numbers -- and the E8 episode (runs/E8 deleted during an earlier cleanup, its
gap to E8r now unattributable) is exactly why this script exists: dump first,
delete second.

Writes:
    docs/retired_runs.json   every results.json, verbatim, plus the cleanup date
    docs/results.md          appends the "Retired runs" section

Usage::

    python scripts/retire_runs.py --dry-run
    python scripts/retire_runs.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

RUNS = Path("runs")
DOCS = Path("docs")
OUT_JSON = DOCS / "retired_runs.json"
RESULTS_MD = DOCS / "results.md"

# Kept on purpose: these two are the references behind the shipped artefacts.
KEEP_WEIGHTS = {"Q8g_long", "Q6_10k"}
# Plumbing checks: no measurement value, removed with their logs.
SMOKE = {"amp_test", "budget_test", "ckpt_test", "ckpt_test2", "ddp_test",
         "epoch_test", "q10_cpu_test"}

KAGGLE_DELETED = [
    "icold21/kron-llm-code", "icold21/kron-llm-code-v2",
    "icold21/kron-llm-code-v3", "icold21/kron-llm-code-v4",
    "icold21/kron-llm-code-v5", "icold21/kron-llm-code-v6",
    "icold21/kron-llm-code-v7", "icold21/kron-llm-code-v8",
    "icold21/kron-llm-code-v9", "icold21/kron-llm-code-v10",
    "icold21/kron-llm-code-v11", "icold21/kron-llm-code-v13",
    "icold21/kron-llm-code-v14", "icold21/kg-test-delete-me",
]


def collect() -> list[dict]:
    out = []
    for path in sorted(RUNS.glob("*/results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        data["_run"] = path.parent.name
        out.append(data)
    return out


def table(rows: list[dict]) -> str:
    lines = ["| run | params | steps | bpb | description |", "|---|---:|---:|---:|---|"]
    for d in rows:
        name = d["_run"]
        if name in SMOKE:
            continue
        params = d.get("params") or d.get("n_params") or d.get("parameters") or ""
        steps = d.get("steps", "")
        bpb = d.get("val_bpb")
        bpb_s = f"**{bpb:.4f}**" if isinstance(bpb, (int, float)) else "—"
        desc = str(d.get("desc") or d.get("experiment_desc") or "")
        if len(desc) > 64:
            desc = desc[:61] + "…"
        lines.append(f"| {name} | {params:,} | {steps:,} | {bpb_s} | {desc} |"
                     if isinstance(params, int) and isinstance(steps, int)
                     else f"| {name} | {params} | {steps} | {bpb_s} | {desc} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = collect()
    OUT_JSON.write_text(json.dumps({
        "cleanup_date": date.today().isoformat(),
        "note": "Verbatim results.json of every run in runs/, kept because "
                "runs/ is git-ignored and its weights were deleted.",
        "runs": rows,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[retire] {len(rows)} run records -> {OUT_JSON}")

    weights = sorted(RUNS.glob("**/*.pt"))
    doomed = [p for p in weights if p.parent.name not in KEEP_WEIGHTS]
    freed = sum(p.stat().st_size for p in doomed)
    print(f"[retire] {len(doomed)} checkpoints, {freed/1e6:.0f} MB")

    section = f"""
## 17. Retired runs and artefacts (cleanup, {date.today().isoformat()})

The weights of the development bench were deleted; the numbers were not. This
matters because it already went wrong once: `runs/E8` was removed in an earlier
cleanup and its `results.json` went with it, so the 3.3 % gap between E8 (1.4789)
and its re-run E8r (1.5282) can no longer be attributed to anything. Before
deleting a single checkpoint this time, every `results.json` was dumped verbatim
into [`retired_runs.json`](retired_runs.json).

What is deleted is the checkpoint file (`best.pt` / `last.pt`). Each run
directory keeps `log.csv`, `results.json` and `config.yaml`, so the curve and the
configuration survive; the weights are reproducible from those.

### Runs whose weights were deleted

{table(rows)}

Kept on purpose: `runs/Q8g_long/best.pt` (the fp32 reference behind the shipped
int8 artefact) and `runs/Q6_10k/best.pt` (the previous reference). Everything
else is regenerable, and the two shipped artefacts live in `artifacts/`.

### Smoke tests removed with their logs

`amp_test`, `budget_test`, `ckpt_test`, `ckpt_test2`, `ddp_test`, `epoch_test`,
`q10_cpu_test` — one to six steps each, plumbing checks (does AMP run, does the
optimizer state resume, does DDP start). No measurement value, and
`q10_cpu_test` alone held 160 MB.

### Kaggle datasets deleted

{chr(10).join('- `' + k + '`' for k in KAGGLE_DELETED)}

These were one-off code bundles: a new dataset id was cut on every code change
so a resumed session could never silently mount stale code. That made sense
during the run; keeping all fourteen afterwards does not.

Kept: `kron-corpus-2b` (the training bins), `kron-q10-ckpt` (the live resume
point, one version per session), `kron-q10a-anneal-bin` (the decay pass),
`kron-llm-code-v15` (current code) and `kron-llm-code-v12`, which stays only
until the session that has it mounted finishes — deleting a dataset a running
kernel has open is not worth the risk for 208 kB.
"""

    if args.dry_run:
        print(section[:1200])
        print("... (dry run, nothing written)")
        return

    text = RESULTS_MD.read_text(encoding="utf-8")
    marker = "## 17. Retired runs and artefacts"
    if marker in text:
        text = text[:text.index(marker)].rstrip() + "\n"
    RESULTS_MD.write_text(text + "\n" + section.strip() + "\n", encoding="utf-8")
    print(f"[retire] section appended to {RESULTS_MD}")

    for path in doomed:
        path.unlink()
    for name in sorted(SMOKE):
        shutil.rmtree(RUNS / name, ignore_errors=True)
    print(f"[retire] deleted {len(doomed)} checkpoints and {len(SMOKE)} smoke dirs")


if __name__ == "__main__":
    main()
