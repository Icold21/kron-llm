"""Generate the S10 (supervised fine-tuning) Kaggle kernel from the Q10 one.

SFT shares everything with pretraining except three things, and all three are
visible in the notebook:

* the corpus -- ``kron-sft-corpus`` instead of ``kron-corpus-2b``;
* the start -- ``training.init_from`` (weights only) instead of ``--resume``,
  because the pretraining optimiser state belongs to a loss landscape the SFT
  pass is deliberately leaving; ``--resume`` would also drag 51k steps of AdamW
  moments into a 40x smaller learning rate;
* the budget -- 4,000 steps is under three hours, so the session is sized to the
  step count rather than to the platform limit.

Usage::

    python scripts/make_s10_kernel.py --code-version 23
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC = Path("kaggle/q10")
DST = Path("kaggle/s10")

PATCHES = [
    ("find_dataset('kron-corpus-2b', 'train.bpe.bin')",
     "find_dataset('kron-sft-corpus', 'train.bpe.bin')"),
    ("'--exp', 'Q10'", "'--exp', 'S10'"),
    ("'/kaggle/working/q10'", "'/kaggle/working/s10'"),
    ("open('/kaggle/working/q10/results.json')",
     "open('/kaggle/working/s10/results.json')"),
    ("copying the 5.7 GB bin costs ~10 min, so 690 left no room",
     "copying the 1.1 GB SFT bin costs ~2 min, so 690 left no room"),
    # Weights only, fresh optimiser.
    ("resume = sorted(glob.glob('/kaggle/input/**/last.pt', recursive=True))\n"
     "if resume:\n"
     "    cmd += ['--resume', resume[0]]\n"
     "    print('resuming from', resume[0])",
     "init = sorted(glob.glob('/kaggle/input/**/last.pt', recursive=True))\n"
     "if init:\n"
     "    cmd += ['--override', 'training.init_from=' + init[0]]\n"
     "    print('initialising weights from', init[0])"),
]

HEADER = """# KronLLM S10 — supervised fine-tuning

Starts from the Q10 decay-pass weights (weights only, fresh AdamW) and trains
4,000 steps x 16,384 tokens = 65.5M tokens on the measured SFT corpus: 513M
tokens of math (with the column drills), code, open Q&A, forum, chat and
identity, 60/40 EN/RU by bytes.

The prompt is not data here.  `model.mask_user_turns` is on, so everything up to
and including `<|assistant|>` is masked and only the answer is a target --
`prompt_tokens` in `sft_index.npy` marks exactly that span.

Learning rate 5e-5 -> 5e-6 over the 4,000 steps, 100 warmup steps.  Checkpoints
every 250 steps to `/kaggle/working/s10`.
""".splitlines(keepends=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-version", type=int, default=23)
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
                # 4,000 steps x 2.55 s is 2.8 h: the budget is the step count.
                lines[i] = ("MINUTES = 180   # 4,000 steps ~ 2.8 h; the platform "
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
        "id": "icold21/kron-s10-sft",
        "title": "Kron S10 supervised fine-tuning",
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            f"icold21/kron-llm-code-v{args.code_version}",
            "icold21/kron-sft-corpus",
            "icold21/kron-q10-ckpt",
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
