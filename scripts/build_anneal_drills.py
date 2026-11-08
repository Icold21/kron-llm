"""Build the arithmetic drill pool for the decay pass (Q10A).

The decay pass is where the chat tokens stop being cold, and arithmetic is the
same kind of problem: the four special ids and the question -> worked-answer
format are equally absent from the epoch corpus.  Putting a slice of drills here
means SFT starts from a model that has already seen the *format*, so its 4,000
steps go into the skill rather than into learning what `<|user|>` means.

Records come from the same generator that feeds SFT
(`scripts/gen_arithmetic_sft.py`), rendered with the same `build_sft.render`, so
the tokens the decay pass sees are exactly the tokens SFT will see.

Usage::

    python scripts/build_anneal_drills.py --target-mb 20
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from build_anneal_chat import MAX_CHARS, MIN_CHARS, MIX  # noqa: E402
from build_sft import render  # noqa: E402
import gen_arithmetic_sft as arith  # noqa: E402

# Column drills are the point, so they get the larger share of the pool; the
# one-line families stay in so the model also meets the short form.
WEIGHTS = {"columns": 4, "other": 1}


def pool_for(lang: str, rng: random.Random) -> list:
    """Weighted list of generators for one language."""
    families, _ = arith.FAMILIES[lang]
    out = []
    for fn, weight in families:
        is_column = "_columns" in fn.__name__
        out.extend([fn] * (weight * WEIGHTS["columns"] if is_column
                           else weight * WEIGHTS["other"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=float, default=20.0,
                    help="per-language pool size in MB of rendered text")
    ap.add_argument("--out-dir", default=str(MIX))
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--bare-share", type=float, default=0.25,
                    help="fraction of the pool written as plain 'a op b = c' "
                         "lines instead of chat drills: the bare arithmetic "
                         "probe feeds a raw continuation, not a chat turn")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = int(args.target_mb * 1e6)

    for index, lang in enumerate(("en", "ru")):
        rng = random.Random(args.seed + index)
        pool = pool_for(lang, rng)
        used = 0
        kept = 0
        attempts = 0
        seen: set[str] = set()
        docs = []
        # A family with a small space (fixed prices, fixed speeds) runs out of
        # distinct texts; the loop must move on instead of retrying it forever.
        # The index advances on every attempt, not on every *accepted* record.
        stuck: Dict[int, int] = {}
        exhausted: set[int] = set()
        while used < target:
            slot = attempts % len(pool)
            attempts += 1
            if slot in exhausted:
                continue
            family = pool[slot]
            question, cot, answer = family(rng)
            tail = "Answer" if lang == "en" else "Ответ"
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant",
                 "content": (str(answer) if not cot.strip()
                             else f"{cot}\n\n{tail}: {answer}")},
            ]
            text, _ = render(messages)
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                stuck[slot] = stuck.get(slot, 0) + 1
                if stuck[slot] > 2000:
                    exhausted.add(slot)
                    print(f"  [{lang}] {family.__name__} exhausted, skipping")
                continue
            seen.add(digest)
            docs.append({
                "title": "",
                "text": text,
                "language": lang,
                "domain": "drills",
                "source": "synthetic:arith",
                "licence": "CC0-1.0",
                "url": "",
                "sha256": digest,
                "family": family.__name__,
            })
            stuck[slot] = 0
            used += len(text.encode("utf-8"))
            kept += 1
            if kept % 20_000 == 0:
                print(f"  [{lang}] {kept:,} records, {used/1e6:.1f} MB")

        # Plain equation lists.  The probe prompt is "67 + 75 = " with no frame,
        # so the continuation shape has to exist as text, not only inside a chat
        # turn -- that is the difference between 1/60 and the drill's 41/60.
        bare_target = int(target * args.bare_share)
        bare_used = 0
        while bare_used < bare_target:
            lines = []
            for _ in range(rng.randint(6, 20)):
                a, b = rng.randint(10, 999), rng.randint(10, 999)
                op = rng.choice(["+", "-"])
                if op == "-" and b > a:
                    a, b = b, a
                value = a + b if op == "+" else a - b
                lines.append(f"{a} {op} {b} = {value}")
            text = "\n".join(lines)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            docs.append({"title": "", "text": text, "language": lang,
                         "domain": "drills", "source": "synthetic:arith_bare",
                         "licence": "CC0-1.0", "url": "", "sha256": digest,
                         "family": "bare_equations"})
            bare_used += len(text.encode("utf-8"))
            used += len(text.encode("utf-8"))
            kept += 1

        path = out_dir / f"{lang}_drills.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
            for doc in docs:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        columns = sum(1 for d in docs if "_columns" in d["family"])
        print(f"[pool] {path} {len(docs):,} records, {used/1e6:.1f} MB, "
              f"{columns:,} column drills ({100*columns/max(1,len(docs)):.0f} %)")


if __name__ == "__main__":
    main()
