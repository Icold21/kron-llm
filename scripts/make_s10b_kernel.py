"""Generate the S10b kernel: the second SFT pass, derived from S10's notebook.

S10b exists because 4,000 steps x 16,384 tokens is only 65.5M tokens -- 13 % of
the 502.7M-token SFT reservoir, 6.6 tokens per parameter.  That is a verdict
probe, not a recipe: it answers whether fine-tuning moves arithmetic at all, and
the second pass is what turns "it moved" into "it is trained".

Everything is inherited from ``kaggle/s10``; four things change:

* the experiment (``S10b``: 8,000 steps, lr 1e-5 -> 1e-6, new cosine);
* the start -- S10's own checkpoint out of ``kron-s10-ckpt``, with a step guard,
  because the *other* checkpoint dataset in the account (``kron-q10-ckpt``) also
  contains a ``last.pt`` and picking it would repeat the whole fine-tune;
* the budget -- 8,000 steps at ~2.7 s/step including evaluations is 6.3 h;
* the output directory.

Usage::

    python scripts/make_s10b_kernel.py --code-version 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC = Path("kaggle/s10")
DST = Path("kaggle/s10b")

PATCHES = [
    ("'--exp', 'S10'", "'--exp', 'S10b'"),
    ("'/kaggle/working/s10'", "'/kaggle/working/s10b'"),
    # S10 now runs 4,800 steps, so the inherited overrides and the guard move
    # with it; S10b keeps its own 8,000-step schedule from the experiment.
    ("'--override', 'training.max_steps=4800',\n"
     "       '--override', 'training.schedule_steps=4800',",
     "'--override', 'training.max_steps=8000',\n"
     "       '--override', 'training.schedule_steps=8000',"),
    ("if _step < 108507:", "if _step < 4700:"),
    ("f'109,507 (>=108,507 accepted): this is a stale dataset version -- '",
     "f'4,800 (>=4,700 accepted): this is not the S10 checkpoint -- '"),
    ("f're-push S10 once kron-q10-ckpt shows the new version')",
     "f'publish S10 first, then push S10b')"),
    ("'/kaggle/input has no last.pt: S10 starts from the '\n"
     "                            'decay pass, not from scratch'",
     "'/kaggle/input has no last.pt: S10b continues S10, '\n"
     "                            'it cannot start from scratch'"),
]

HEADER = """# KronLLM S10b — second supervised fine-tuning pass

Continues from S10's checkpoint (weights only, fresh AdamW) for 8,000 more steps
x 16,384 tokens = 131M tokens.  With S10's 65.5M that is 197M tokens of the
502.7M-token corpus: 39 %, 19.8 tokens per parameter.

Why a second pass exists: 4,000 steps is a verdict probe, not a recipe.  If S10
learns the drill format and exact-match arithmetic is still zero, the limit is
capacity, not data -- and this pass is not launched.  If the loss is still
falling at step 4,000 and arithmetic has moved, this is what makes it trained.

Learning rate 1e-5 -> 1e-6 over the 8,000 steps (an order of magnitude below
S10's: the format is already there), 50 warmup steps.  Checkpoints every 500.
""".splitlines(keepends=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-version", type=int, default=24)
    args = ap.parse_args()

    nb = json.loads((SRC / "train.ipynb").read_text(encoding="utf-8"))
    DST.mkdir(parents=True, exist_ok=True)

    applied = {old: 0 for old, _ in PATCHES}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for old, new in PATCHES:
            if old in src:
                src = src.replace(old, new)
                applied[old] += 1
        lines = src.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.strip().startswith("MINUTES ="):
                # 8,000 steps x ~2.7 s including the every-100-step evaluation.
                lines[i] = ("MINUTES = 380   # 8,000 steps ~ 6.3 h; the platform "
                            "cap is 720 min\n")
                applied["MINUTES"] = applied.get("MINUTES", 0) + 1
        cell["source"] = lines

    missing = [old[:60] for old, n in applied.items() if n == 0]
    if missing:
        raise SystemExit("these patches matched nothing:\n  " + "\n  ".join(missing))

    nb["cells"][0]["source"] = HEADER
    (DST / "train.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                                     encoding="utf-8")

    meta = {
        "id": "icold21/kron-s10b-sft-pass2",
        "title": "Kron S10b second SFT pass",
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        # kron-q10-ckpt is deliberately absent: both it and kron-s10-ckpt hold a
        # last.pt, and mounting both makes the init glob a coin toss.
        "dataset_sources": [
            f"icold21/kron-llm-code-v{args.code_version}",
            "icold21/kron-sft-corpus",
            "icold21/kron-s10-ckpt",
        ],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"{DST}/train.ipynb")
    print(f"{DST}/kernel-metadata.json")
    print("patches applied:", {old[:40]: n for old, n in applied.items()})


if __name__ == "__main__":
    main()
