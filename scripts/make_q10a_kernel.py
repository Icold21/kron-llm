"""Generate the Q10A (decay pass) Kaggle kernel from the Q10 one.

The decay pass is the same training loop on a different corpus, an explicit
step count and no warmup, so cloning the pretraining notebook keeps the two
runs honest: any fix made to one is one diff away from the other, and there is
no second copy of the data-copy / dataset-mount logic to drift.

Usage::

    python scripts/make_q10a_kernel.py
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("kaggle/q10")
DST = Path("kaggle/q10a")

# (old, new) applied to every code cell, in order.
PATCHES = [
    ("find_dataset('kron-corpus-2b', 'train.bpe.bin')",
     "find_dataset('kron-q10a-anneal-bin', 'train.bpe.bin')"),
    # The decay pass ships with its own config (run_steps, step-size schedule),
    # so it must mount the code dataset that contains that config.
    ("find_dataset('kron-llm-code-v12', 'kronllm')",
     "find_dataset('kron-llm-code-v14', 'kronllm')"),
    ("'--exp', 'Q10'", "'--exp', 'Q10A'"),
    ("'/kaggle/working/q10'", "'/kaggle/working/q10a'"),
    ("open('/kaggle/working/q10/results.json')",
     "open('/kaggle/working/q10a/results.json')"),
]


def main() -> None:
    nb = json.loads((SRC / "train.ipynb").read_text(encoding="utf-8"))
    DST.mkdir(parents=True, exist_ok=True)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            for old, new in PATCHES:
                src = src.replace(old, new)
            cell["source"] = src.splitlines(keepends=True)

    # The markdown header describes one epoch over the full corpus; the decay
    # pass is a different stage and must not be advertised as pretraining.
    nb["cells"][0]["source"] = """# KronLLM Q10A — decay pass (WSD tail)

Resumes the end-of-epoch Q10 weights and anneals the LR 3e-4 → 1e-6 over
**29.6 % of an epoch** (40,649 steps × 16,384 tokens = 666M tokens) on the
curated mix: literature, encyclopedic, science, 3× more math than the epoch saw,
and the whole identity slice repeated to ~2.5 % of the tokens. The step count is
the bin divided by the step size, so the pass reads its data exactly once.

No warmup: the pass starts at full LR on the curve. Checkpoints every 250 steps
to `/kaggle/working/q10a`; the session stops at the 500-minute wall clock and
the next one resumes from `last.pt`.
""".splitlines(keepends=True)

    (DST / "train.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                                     encoding="utf-8")

    meta = {
        "id": "icold21/kron-q10a-decay-pass",
        "title": "Kron Q10A decay pass",
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            "icold21/kron-llm-code-v14",
            "icold21/kron-q10a-anneal-bin",
            "icold21/kron-q10-ckpt",
        ],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"{DST}/train.ipynb")
    print(f"{DST}/kernel-metadata.json")


if __name__ == "__main__":
    main()
