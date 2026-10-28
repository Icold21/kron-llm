"""When does top_p actually bind, given a top_k cap?

The sweep showed that top_k=8 + top_p=0.9 is not the same as top_k=8 alone, so
the "p is decorative" claim needs a number rather than an argument.  The mass
walk in data/decoding_mass.json has mass@8 for every decoding step; if that mass
exceeds p, top_p keeps fewer than eight tokens and the two settings diverge.

Usage::

    python scripts/decoding_bind.py
"""
from __future__ import annotations

import json
from pathlib import Path

MASS = Path("data/decoding_mass.json")


def main() -> None:
    walk = json.loads(MASS.read_text(encoding="utf-8"))
    rows = [r for per_prompt in walk.values() for r in per_prompt]
    print(f"steps total: {len(rows)}")
    for p in (0.5, 0.8, 0.9, 0.95, 0.99):
        hits = sum(1 for r in rows if r["mass@8"] > p)
        print(f"  top_p={p}: binds on {hits}/{len(rows)} steps "
              f"({100 * hits / len(rows):.1f} %), i.e. keeps <8 tokens")
    print()
    for label, per_prompt in walk.items():
        peak = max(r["mass@8"] for r in per_prompt)
        hits = sum(1 for r in per_prompt if r["mass@8"] > 0.9)
        print(f"  {label:<11} mass@8>0.9 on {hits:>2}/{len(per_prompt)} steps, "
              f"peak mass@8 {peak:.3f}")


if __name__ == "__main__":
    main()
