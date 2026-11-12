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
    # The corpus is S10's with 10 % pretraining documents mixed in as plain text
    # (scripts/mix_sft_pretrain.py): S10 cost 0.139 bpb of general language, and a
    # second pass on the pure SFT corpus would deepen exactly that.
    ("find_dataset('kron-sft-corpus', 'train.bpe.bin')",
     "find_dataset('kron-sft-mix', 'train.bpe.bin')"),
    # One session is one complete anneal: 660 minutes is 14,558 steps at 2.72 s,
    # which is what the platform allows in one kernel (720 is the cap and the
    # output has to be published before it).
    ("'--override', 'training.max_steps=4800',\n"
     "       '--override', 'training.schedule_steps=4800',",
     "'--override', 'training.max_steps=14558',\n"
     "       '--override', 'training.schedule_steps=14558',"),
    # S10 stopped at step 4,521 of 4,800 -- the 205-minute budget ended the pass,
    # not the step count -- so the guard has to accept the run that exists, or
    # S10b refuses the only checkpoint it is allowed to continue from.
    ("if _step < 108507:", "if _step < 4500:"),
    ("f'109,507 (>=108,507 accepted): this is a stale dataset version -- '",
     "f'4,521 (>=4,500 accepted): this is not the S10 checkpoint -- '"),
    ("f're-push S10 once kron-q10-ckpt shows the new version')",
     "f'publish S10 first, then push S10b')"),
    ("'/kaggle/input has no last.pt: S10 starts from the '\n"
     "                            'decay pass, not from scratch'",
     "'/kaggle/input has no last.pt: S10b continues S10, '\n"
     "                            'it cannot start from scratch'"),
]

HEADER = """# KronLLM S10b — supervised fine-tuning on the mixed corpus

Continues from S10's checkpoint (weights only, fresh AdamW) for one anneal of
14,558 steps x 16,384 tokens = **238M tokens** of the mixed corpus: the SFT set
plus 10 % pretraining documents, copied in as plain text so that the loss on them
is the pretraining loss.

Why the mix: S10 (78.6M SFT tokens) bought 0.146 bpb on its own held-out bin and
cost **0.139 bpb on general text** (1.0842 -> 1.2236).  A second pass on the pure
SFT corpus would deepen exactly that, so a tenth of every batch is ordinary
prose -- no ``<|system|>``, no ``<|user|>``, no prompt for the mask to hide.

Learning rate 1e-5 -> 1e-6 over 14,558 steps (an order of magnitude below S10's:
the format is already there), 50 warmup steps.  Checkpoints every 500.
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
                lines[i] = ("MINUTES = 660   # 14,558 steps ~ 11 h; the platform "
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
            "icold21/kron-sft-mix",
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
