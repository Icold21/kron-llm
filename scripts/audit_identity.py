"""Audit the identity shards with the same gate that generated them.

Every document goes through :func:`gen_identity.acceptable`, so a clean audit
means every document obeys the rules the generator enforces: no digits, no
assistant boilerplate, no dialogue scaffolding, no stage directions, no
markup, first-person prose of a sane length.

The script also writes ``identity_samples.md`` with eight random documents per
slice, for a human read.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_identity import acceptable, clean

SHARDS = Path("data/bilingual/shards")
NAMES = ("identity_en", "identity_ru", "identity_rick_en", "identity_rick_ru")


def main() -> None:
    report = {}
    for name in NAMES:
        rows = [json.loads(l) for l in (SHARDS / f"{name}.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        lang = "ru" if name.endswith("_ru") else "en"
        reasons: Counter[str] = Counter()
        bad_examples: list[tuple[str, str]] = []
        for row in rows:
            ok, why = acceptable(clean(row["text"]), lang)
            if not ok:
                reasons[why] += 1
                if len(bad_examples) < 3:
                    bad_examples.append((why, row["text"][:110].replace("\n", " ")))
        report[name] = {"docs": len(rows), "defects": sum(reasons.values()),
                        "reasons": dict(reasons)}
        line = (f"{name}: {len(rows)} docs, "
                f"{sum(reasons.values())} defects")
        if reasons:
            line += f" {dict(reasons)}"
        print(line)
        for why, sample in bad_examples:
            print(f"    {why}: {sample}")

    Path("data/bilingual/identity_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    samples = ["# Identity samples", ""]
    for name in NAMES:
        rows = [json.loads(l) for l in (SHARDS / f"{name}.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        picks = random.Random(11).sample(rows, min(8, len(rows)))
        samples += [f"## {name} ({len(rows)} docs)", ""]
        for row in picks:
            samples += [f"> {row['text'].strip()}", ""]
    out = Path("data/bilingual/identity_samples.md")
    out.write_text("\n".join(samples), encoding="utf-8")
    print(f"[samples] {out}")


if __name__ == "__main__":
    main()
