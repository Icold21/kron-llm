"""Write docs/reproducing.md from the retired-run dump.

Reproducibility of a number is a command, not a checkpoint: every claim in
``docs/results.md`` should be re-runnable from the code, the config, the seed
and the data recipe.  This script renders that promise into a table, using the
numbers captured in ``docs/retired_runs.json`` (so it survives the deletion of
the weights it describes).

Usage::

    python scripts/make_reproducing.py
"""
from __future__ import annotations

import json
from pathlib import Path

DUMP = Path("docs/retired_runs.json")
OUT = Path("docs/reproducing.md")

# Which preparation a family needs before the command runs.
DATA = {
    "byte": "TinyStories bytes: `python scripts/prepare_data.py`",
    "bpe": "TinyStories BPE: `python scripts/prepare_bpe.py --vocab 4096`",
    "wikitext": "WikiText-103 bytes: `python scripts/prepare_wikitext.py`",
}
TOKENIZER_URL = {
    "byte": "data/train.bin",
    "bpe": "data/train.bpe.bin",
    "wikitext": "data/wikitext/train.bin",
}


def family(run: dict) -> str:
    name = run["_run"]
    if name.endswith("w") or "wikitext" in str(run.get("desc", "")).lower():
        return "wikitext"
    if run.get("vocab_size") == 256:
        return "byte"
    return "bpe"


def main() -> None:
    data = json.loads(DUMP.read_text(encoding="utf-8"))
    runs = [r for r in data["runs"] if r["_run"] not in
            {"amp_test", "budget_test", "ckpt_test", "ckpt_test2", "ddp_test",
             "epoch_test", "q10_cpu_test", "E2"}]

    lines = [
        "# Reproducing the numbers",
        "",
        "Every claim in [results.md](results.md) is a command plus a seed, not a",
        "checkpoint. This page is the list of those commands, rendered from the",
        "verbatim `results.json` dump in [retired_runs.json](retired_runs.json),",
        "so it survives the deletion of the weights it describes.",
        "",
        "## Policy: what is kept, and why",
        "",
        "| artefact | kept? | reason |",
        "|---|---|---|",
        "| code, config, seed | yes, in git | the only things a re-run needs |",
        "| `results.json` of every run | yes, `docs/retired_runs.json` | the "
        "numbers are the result; `runs/` is git-ignored, so this dump is the copy |",
        "| `log.csv` per run | yes, local | curves are cheap and useful for diffing |",
        "| 190K–2.8M experiment weights | **no** | 4,000 steps ≈ 28 min on the "
        "reference GPU; storing them buys nothing |",
        "| shipped artefacts (`artifacts/q8g_int8.pt`, `q2_int8.pt`) | yes, in git "
        "| a reader can load them offline |",
        "| Q10 weights | yes, at the end of the epoch | 95 GPU-h is not "
        "reproducible on a laptop; publish weights-only + SHA256 |",
        "",
        "A 190K model that takes 28 minutes to retrain does not need a binary",
        "attached: the config is the artefact. The opposite is true for the 10M",
        "run, where nobody can afford the retrain — that one gets published in",
        "full.",
        "",
        "## What is *not* claimed",
        "",
        "* **Bit-exactness.** `deterministic=True` is set, but cuDNN and the",
        "  reduction order in `baddbmm` are not bit-stable across driver versions.",
        "  Expect the same bpb to ±0.01, not to the fourth digit.",
        "* **Seed-robust deltas below ~0.03 bpb.** One control run (hybrid",
        "  compositions) reversed a 1.3 % win on a second seed; that is the",
        "  measured noise floor and it applies to every single-seed number here.",
        "* **Anything on TinyStories as a statement about the target corpus.** The",
        "  bench is the bench.",
        "",
        "## Commands",
        "",
        "Family prerequisites: byte → " + DATA["byte"] + "; BPE → " + DATA["bpe"] +
        "; WikiText → " + DATA["wikitext"] + ".",
        "",
        "| run | params | steps | expected bpb | runtime | command |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for run in sorted(runs, key=lambda r: r["_run"]):
        name = run["_run"]
        bpb = run.get("val_bpb")
        bpb_s = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else "—"
        secs = run.get("time_total_sec")
        mins = f"{secs/60:.0f} min" if isinstance(secs, (int, float)) else "—"
        params = run.get("n_params")
        steps = run.get("steps")
        cmd = f"`python -m kronllm.cli train --exp {name} --steps {steps}`"
        lines.append(f"| {name} | {params:,} | {steps:,} | **{bpb_s}** | {mins} "
                     f"| {cmd} |" if isinstance(params, int)
                     else f"| {name} | {params} | {steps} | **{bpb_s}** | {mins} | {cmd} |")

    lines += [
        "",
        "The seed is `config.seed` (default 42), recorded in every",
        "`results.json`; nothing else about a run is implicit.",
        "",
        "## Reproducing the Q10 validation number",
        "",
        "The training-time `val_bpb` is measured on a fixed slice; the honest",
        "number is measured per bucket with real byte counts. Both are",
        "reproducible offline against a downloaded checkpoint:",
        "",
        "```bash",
        "# per-slice bpb with exact UTF-8 byte counts (one 10M forward pass)",
        "python scripts/eval_reference.py --source ckpt \\",
        "    --ckpt data/kaggle_ckpt/last.pt --blocks 1024 \\",
        "    --out data/baseline_q10.json",
        "",
        "# the GPT-2 reference on exactly the same blocks",
        "python scripts/eval_reference.py --source gpt2 --blocks 1024 \\",
        "    --out data/baseline_gpt2.json",
        "```",
        "",
        "Both write the byte count, the token count and every block they scored,",
        "so the comparison can be audited rather than trusted.",
    ]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
