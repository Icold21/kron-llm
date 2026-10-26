"""Build the end-of-session email body from the run artefacts.

Reads ``data/kaggle_out/q10/log.csv`` (and ``results.json`` if the run
finished) and writes ``data/session_email.md``.  The chain script then mails
that file, so the report always reflects the files that were actually
downloaded rather than a hand-written guess.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

RUN = Path("data/kaggle_out/q10")
OUT = Path("data/session_email.md")


def main() -> None:
    rows = list(csv.DictReader((RUN / "log.csv").open(encoding="utf-8")))
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    last = train[-1] if train else {}
    first = train[0] if train else {}

    lines = ["# Q10 session report", ""]
    if last:
        steps = int(last["step"])
        sec = float(last["sec_per_step"] or 0)
        lines += [
            f"- steps completed: **{steps:,}** of 137,377 "
            f"({100 * steps / 137377:.1f} % of one epoch)",
            f"- train loss {float(last['loss']):.3f}, "
            f"bpb {float(last['bpb']):.3f} "
            f"(first logged step {first['step']}: "
            f"loss {float(first['loss']):.3f}, bpb {float(first['bpb']):.3f})",
            f"- sec/step {sec:.2f} -> {16384 / sec:,.0f} tokens/s",
            f"- lr {float(last['lr']):.2e}, grad_norm {last['grad_norm']}",
            "",
        ]
    if val:
        lines += ["## Validation", "", "| step | val loss | val bpb |", "|---:|---:|---:|"]
        for r in val:
            lines.append(f"| {r['step']} | {float(r['val_loss']):.3f} | "
                         f"{float(r['val_bpb']):.4f} |")
        lines.append("")
    res = RUN / "results.json"
    if res.exists():
        try:
            data = json.loads(res.read_text(encoding="utf-8"))
            keys = ("steps", "best_val_bpb", "tokens_seen", "minutes", "params")
            got = {k: data[k] for k in keys if k in data}
            if got:
                lines += ["## Run summary", ""]
                lines += [f"- {k}: {v}" for k, v in got.items()]
                lines.append("")
        except (OSError, ValueError):
            pass
    lines += [
        "## Next",
        "",
        "The checkpoint is published as `icold21/kron-q10-ckpt` and the next "
        "session is already pushed on top of it, so training continues without "
        "a gap.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
