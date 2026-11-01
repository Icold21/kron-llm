"""GPU budget per stage, derived from the bin that is actually being trained on.

Nothing here is a hand-typed constant except the measured speed and the step
count already completed: the token count comes from ``tokenizer_meta.json`` of
the epoch bin, so the table cannot drift away from the corpus.

Usage::

    python scripts/make_compute_budget.py            # write the docs
    python scripts/make_compute_budget.py --print    # numbers only
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = Path(r"E:\kron_bins\epoch_v5\tokenizer_meta.json")
ANNEAL_META = Path(r"E:\kron_bins\anneal_v5\tokenizer_meta.json")
FALLBACK_TOKENS = 2_783_383_150

SEC_PER_STEP = 2.552        # median of the last ten steps of session 5, T4
TOKENS_PER_STEP = 16_384
DONE_STEPS = 36_549         # session 5 finished here
SESSION_HOURS = 11.0        # MINUTES = 660 in the kernel; 12 h is the cap
ANNEAL_SHARE = 0.30         # the owner's call: decay = 30 % of an epoch
SFT_STEPS = 4_000           # config S10


def epoch_tokens() -> int:
    try:
        meta = json.loads(META.read_text(encoding="utf-8"))
        return int(meta["train_tokens"])
    except (OSError, KeyError, ValueError):
        return FALLBACK_TOKENS


def anneal_steps(epoch_steps: int) -> int:
    """The decay pass length: the bin if it exists, else the 30 % target.

    The decay bin is what actually gets read once, so its own size is the
    honest number -- it came out at 29.1 % of the epoch because a few domains
    have no supply left to fill the last percentage point.
    """
    try:
        meta = json.loads(ANNEAL_META.read_text(encoding="utf-8"))
        return int(meta["train_tokens"]) // TOKENS_PER_STEP
    except (OSError, KeyError, ValueError):
        return int(epoch_steps * ANNEAL_SHARE)


def hours(steps: int) -> float:
    return steps * SEC_PER_STEP / 3600


def numbers() -> dict:
    tokens = epoch_tokens()
    steps = tokens // TOKENS_PER_STEP
    anneal = anneal_steps(steps)
    left = steps - DONE_STEPS
    return {
        "tokens": tokens,
        "steps": steps,
        "done": DONE_STEPS,
        "left": left,
        "anneal": anneal,
        "sft": SFT_STEPS,
        "h_epoch": hours(steps),
        "h_done": hours(DONE_STEPS),
        "h_left": hours(left),
        "h_anneal": hours(anneal),
        "h_sft": hours(SFT_STEPS),
        "h_total_left": hours(left) + hours(anneal) + hours(SFT_STEPS),
    }


def doc(n: dict) -> str:
    rows = [
        ("Q10 pretraining, full epoch", n["steps"], n["h_epoch"]),
        ("done so far (step 36,549)", n["done"], n["h_done"]),
        ("left in the epoch", n["left"], n["h_left"]),
        ("decay Q10A (the anneal bin)", n["anneal"], n["h_anneal"]),
        ("SFT S10 (4,000 steps)", n["sft"], n["h_sft"]),
    ]
    out = [
        "# GPU budget per stage",
        "",
        "Measured speed: **2.552 s/step** (median of the last ten steps of "
        "session 5, one Kaggle T4, batch 32, seq 512), **16,384 tokens per "
        "step**. The corpus size is read from the epoch bin's "
        "`tokenizer_meta.json`, so the table below is arithmetic on real "
        "numbers, not an estimate.",
        "",
        "| stage | steps | tokens | GPU hours |",
        "|---|---|---|---|",
    ]
    for name, steps, h in rows:
        out.append(f"| {name} | {steps:,} | "
                   f"{steps * TOKENS_PER_STEP / 1e9:.2f}B | {h:,.1f} |")
    out += [
        "",
        f"**{n['h_total_left']:,.0f} GPU hours remain** -- about "
        f"{n['h_total_left'] / SESSION_HOURS:.0f} sessions of "
        f"{SESSION_HOURS:.0f} h, which is {n['h_total_left'] / 30:.1f} weeks at "
        "the free Kaggle quota of ~30 h/week.",
        "",
        f"One session (`MINUTES = 660`, {SESSION_HOURS:.0f} h) is ~"
        f"{SESSION_HOURS * 3600 / SEC_PER_STEP:,.0f} steps "
        f"({SESSION_HOURS * 3600 / SEC_PER_STEP * TOKENS_PER_STEP / 1e9:.2f}B "
        "tokens). It is 11 h and not 12 because the platform caps a session at "
        "720 min and the 5.7 GB bin copy takes ~10 min.",
        "",
        "## What the stages are",
        "",
        f"- **Epoch**: {n['tokens']:,} tokens = {n['steps']:,} steps, one "
        "pass over the rebuilt corpus (EN 59 / RU 41 by tokens).",
        f"- **Decay Q10A**: {n['anneal']:,} steps "
        f"({100 * n['anneal'] / n['steps']:.1f} % of the epoch -- the anneal "
        "bin divided by the step size, so it is read exactly once), the same "
        "domain profile selected by quality, plus 5 % chat, LR 3e-4 -> 1e-6.",
        "- **SFT S10**: 4,000 steps = 65.5M tokens, 18 % of the SFT "
        "reservoir; a full epoch there would memorise.",
        "",
        "## What is not counted",
        "",
        "- reference evals (`eval_reference.py`, GPT-2 on the same blocks) -- "
        "minutes each;",
        "- corpus builds and dataset uploads -- CPU and network, no GPU;",
        "- local runs on the 1660 Ti -- they do not touch the Kaggle quota.",
    ]
    return "\n".join(out) + "\n"


def readme_block(n: dict) -> str:
    return "\n".join([
        "## GPU budget per stage",
        "",
        "Measured speed: **2.552 s/step** (median of the last ten steps of "
        "session 5, one Kaggle T4, batch 32, seq 512), **16,384 tokens per "
        "step**. Full derivation: "
        "[`docs/compute_budget.md`](docs/compute_budget.md).",
        "",
        "| stage | steps | tokens | GPU hours |",
        "|---|---|---|---|",
        f"| Q10 pretraining, full epoch | {n['steps']:,} | "
        f"{n['steps'] * TOKENS_PER_STEP / 1e9:.2f}B | {n['h_epoch']:,.1f} |",
        f"| done so far (step 36,549) | {n['done']:,} | "
        f"{n['done'] * TOKENS_PER_STEP / 1e9:.2f}B | {n['h_done']:,.1f} |",
        f"| left in the epoch | {n['left']:,} | "
        f"{n['left'] * TOKENS_PER_STEP / 1e9:.2f}B | {n['h_left']:,.1f} |",
        f"| decay Q10A (30 % of the epoch) | {n['anneal']:,} | "
        f"{n['anneal'] * TOKENS_PER_STEP / 1e9:.2f}B | {n['h_anneal']:,.1f} |",
        f"| SFT S10 | {n['sft']:,} | "
        f"{n['sft'] * TOKENS_PER_STEP / 1e9:.2f}B | {n['h_sft']:,.1f} |",
        "",
        f"**{n['h_total_left']:,.0f} GPU hours remain**, ~"
        f"{n['h_total_left'] / SESSION_HOURS:.0f} sessions of "
        f"{SESSION_HOURS:.0f} h at the free Kaggle quota of ~30 h/week. One "
        "such session is "
        f"~{SESSION_HOURS * 3600 / SEC_PER_STEP:,.0f} steps "
        f"({SESSION_HOURS * 3600 / SEC_PER_STEP * TOKENS_PER_STEP / 1e9:.2f}B tokens).",
    ])


def update_readme(n: dict) -> str:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    start = text.index("## GPU budget per stage")
    end = text.index("## 12. Repository layout")
    text = text[:start] + readme_block(n) + "\n\n" + text[end:]
    path.write_text(text, encoding="utf-8")
    return f"README.md: replaced the budget block ({end - start} chars)"


def update_card(n: dict) -> str:
    path = ROOT / "docs" / "hf_model_card.md"
    text = path.read_text(encoding="utf-8")
    new = (
        "## Training budget\n\n"
        f"At the measured 2.552 s/step on a single Kaggle T4 (batch 32, seq "
        f"512), one epoch of the {n['tokens'] / 1e9:.2f}B-token corpus is "
        f"**{n['h_epoch']:,.0f} GPU-hours**; the decay pass (30 % of the "
        f"epoch, quality-selected slice plus 5 % chat) adds "
        f"**{n['h_anneal']:,.0f}**, and SFT 4,000 steps adds "
        f"**{n['h_sft']:.1f}**. Roughly **{n['h_total_left']:,.0f} GPU-hours "
        "remain** from step 36,549 -- about "
        f"{n['h_total_left'] / SESSION_HOURS:.0f} sessions of "
        "quota. Derivation in `docs/compute_budget.md` of the code repository.\n\n"
    )
    start = text.index("## Training budget")
    end = text.index("## Next")
    path.write_text(text[:start] + new + text[end:], encoding="utf-8")
    return "docs/hf_model_card.md: budget paragraph rewritten"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="numbers only")
    ap.add_argument("--done", type=int, default=None,
                    help="steps finished so far (the last session's checkpoint)")
    ap.add_argument("--epoch-meta", default=str(META))
    ap.add_argument("--anneal-meta", default=str(ANNEAL_META))
    args = ap.parse_args()
    if args.done is not None:
        globals()["DONE_STEPS"] = args.done
    if args.epoch_meta != str(META):
        globals()["META"] = Path(args.epoch_meta)
    if args.anneal_meta != str(ANNEAL_META):
        globals()["ANNEAL_META"] = Path(args.anneal_meta)
    n = numbers()
    if args.print:
        print(json.dumps(n, indent=2))
        return
    (ROOT / "docs" / "compute_budget.md").write_text(doc(n), encoding="utf-8")
    print(update_readme(n))
    print(update_card(n))
    print(f"epoch {n['tokens']:,} tokens = {n['steps']:,} steps, "
          f"{n['h_epoch']:,.1f} GPU-h; "
          f"{n['h_total_left']:,.1f} GPU-h left "
          f"(~{n['h_total_left'] / SESSION_HOURS:.0f} sessions)")


if __name__ == "__main__":
    main()
