"""Build the curated mix for the WSD decay pass (Q10A).

The decay pass runs the tail of the epoch at LR 3e-4 -> 1e-6 on data chosen for
quality, not for coverage: the epoch has already taught the model the general
distribution, this pass decides what it sharpens.

Two things changed after the run started (2026-10-27, user decision):

* the pass is **30 % of an epoch** (41,213 steps, ~28 GPU-h) instead of 20 %,
  because at 10M parameters the epoch is far from compute-optimal and the tail
  is where extra cheap descent lives;
* the budget is spent more on **math** (14 % of tokens vs 6 %) and on **code,
  logic and ML**, and **the whole identity slice goes into the pass**, repeated
  to ~2.5 % of the tokens.  Identity is 0.11 % of the epoch *with* its 8x
  upsampling, so the decay pass is the only place the name and the facts can
  actually be memorised.

Budgeting is in **tokens per language**, not bytes: Russian costs ~5.0 bytes per
token and English ~2.9, so a byte budget would quietly hand Russian 20 % more
tokens than intended.  Selection is deterministic (documents sorted by sha256),
so the same command always produces the same mix.

Usage::

    python scripts/build_anneal_mix.py --dry-run
    python scripts/build_anneal_mix.py
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

MIX = Path("data/bilingual/mix")
OUT = Path("data/bilingual/anneal_mix")

EPOCH_STEPS = 137_377
TOKENS_PER_STEP = 32 * 512

# Measured on the epoch-1 held-out set (tokenizer_meta.json).
BYTES_PER_TOKEN = {"en": 2.917, "ru": 4.964}

# Shares of the decay budget, per language and domain, in tokens.  Literature
# still carries the most transferable signal per token; math is tripled against
# the first plan because arithmetic and step-by-step reasoning are the weak
# points a 10M model can still improve on.  A minority of general text stays in
# on purpose: a purely encyclopedic diet narrows a small model, and narrowing
# shows up as worse bpb on the epoch-1 held-out set.
SHARES = {
    "en/literature": 0.112,
    "ru/literature": 0.195,
    "en/encyclopedic": 0.130,
    "ru/encyclopedic": 0.090,
    "en/science": 0.120,
    "ru/science": 0.040,
    "en/math": 0.095,
    "ru/math": 0.050,
    "en/ml": 0.060,
    "ru/ml": 0.004,
    "en/code": 0.052,
    "ru/cs": 0.007,
    "en/logic": 0.015,
    "ru/logic": 0.005,
    "en/identity": 0.012,
    "ru/identity": 0.013,
}

# If a pool cannot fill its share, the shortfall goes here, in this order, so
# the pass still reaches the step count instead of silently shortening.
FILLERS = ["encyclopedic", "literature", "science", "math", "ml", "code",
           "logic", "cs"]


def domain_of(path: Path) -> tuple[str, str]:
    """``ru_encyclopedic.jsonl.gz`` -> ("ru", "encyclopedic")."""
    stem = path.stem.replace(".jsonl", "")
    lang, domain = stem.split("_")[:2]
    return lang, domain


def load_pool(path: Path, dedupe: bool = False) -> list[dict]:
    """Read one pool, optionally collapsing exact duplicates.

    The identity pools were upsampled in place for the epoch (8 copies of every
    document).  For the decay pass the copies are re-created deliberately from
    the unique set, so they are collapsed here first -- otherwise "all identity
    documents" would quietly mean "all 13,200 rows", eight times over.
    """
    docs: list[dict] = []
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            if dedupe:
                key = doc.get("sha256") or doc["text"][:200]
                if key in seen:
                    continue
                seen.add(key)
            docs.append(doc)
    docs.sort(key=lambda d: d.get("sha256", ""))
    return docs


def bytes_of(docs: list[dict]) -> int:
    return sum(len(d["text"].encode("utf-8")) for d in docs)


def take(docs: list[dict], want_tokens: int, bpt: float,
         repeats: bool = False) -> tuple[list[dict], int]:
    """Take documents until ``want_tokens`` worth of text, deterministically.

    ``repeats`` recycles the same documents when the pool is smaller than the
    budget (identity), otherwise it stops at the end of the pool and the
    shortfall is reported by the caller.
    """
    want_bytes = int(want_tokens * bpt)
    taken: list[dict] = []
    used = 0
    if not docs:
        return taken, used
    i = 0
    while used < want_bytes:
        if i >= len(docs):
            if not repeats:
                break
            i = 0
        doc = docs[i]
        taken.append(doc)
        used += len(doc["text"].encode("utf-8"))
        i += 1
    return taken, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=int(0.30 * EPOCH_STEPS),
                    help="decay-pass steps; default is 30 %% of the epoch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    budget_tokens = args.steps * TOKENS_PER_STEP
    print(f"[anneal] {args.steps:,} steps x {TOKENS_PER_STEP:,} tokens = "
          f"{budget_tokens/1e6:.0f}M tokens")

    pools = {domain_of(p): p for p in sorted(MIX.glob("*.jsonl.gz"))}
    missing = [k for k in SHARES if tuple(k.split("/")) not in pools]
    if missing:
        raise SystemExit(f"missing pools: {missing}")
    if not args.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_docs = 0
    lang_tokens = {"en": 0, "ru": 0}
    shortfall = {"en": 0.0, "ru": 0.0}
    written: dict[tuple[str, str], list[dict]] = {}

    for key, share in SHARES.items():
        lang, domain = key.split("/")
        bpt = BYTES_PER_TOKEN[lang]
        want = int(budget_tokens * share)
        docs = load_pool(pools[(lang, domain)], dedupe=(domain == "identity"))
        pool_tokens = bytes_of(docs) / bpt
        taken, used = take(docs, want, bpt, repeats=(domain == "identity"))
        got = used / bpt
        total_tokens += got
        total_docs += len(taken)
        lang_tokens[lang] += got
        written[(lang, domain)] = taken
        flag = ""
        if got < 0.97 * want:
            shortfall[lang] += want - got
            flag = "  <-- pool exhausted"
        note = (f", {len(taken)/max(1,len(docs)):.1f}x repeat"
                if domain == "identity" else "")
        print(f"[anneal] {lang}/{domain:<13} {len(taken):>7,} docs "
              f"{got/1e6:7.1f}M tok of {want/1e6:6.1f}M "
              f"({got/max(pool_tokens,1e-9):5.0%} of pool{note})" + flag)

    # Redistribute whatever the small pools could not supply, so the pass still
    # runs the planned number of steps.
    for lang, lost in shortfall.items():
        remaining = lost
        for filler in FILLERS:
            if remaining <= 0:
                break
            key = (lang, filler)
            if key not in pools or key not in written:
                continue
            bpt = BYTES_PER_TOKEN[lang]
            have = bytes_of(written[key]) / bpt
            # Re-read without the earlier cut: cheap for the filler pools.
            docs = load_pool(pools[key])
            pool_tokens = bytes_of(docs) / bpt
            room = max(0.0, pool_tokens - have)
            add_tokens = min(remaining, room)
            if add_tokens <= 0:
                continue
            extra, used = take(docs, int(have + add_tokens), bpt)
            extra = extra[len(written[key]):]
            written[key] = written[key] + extra
            added = used / bpt - have
            total_tokens += added
            total_docs += len(extra)
            lang_tokens[lang] += added
            remaining -= added
            print(f"[anneal] refill {lang}/{filler:<13} +{added/1e6:6.1f}M tok")
        if remaining > 1e6:
            print(f"[anneal] WARNING: {lang} still {remaining/1e6:.1f}M short")

    if not args.dry_run:
        for (lang, domain), docs in written.items():
            # Plain JSONL, not gzip: the mix is an intermediate consumed once by
            # prepare_bilingual_bin.py, and level-9 gzip of 2.6 GB of text costs
            # hours for no benefit (the tokenised bin is what gets uploaded).
            out = OUT / f"{lang}_{domain}.jsonl"
            with out.open("w", encoding="utf-8") as fh:
                for doc in docs:
                    fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    share_en = 100 * lang_tokens["en"] / max(1, total_tokens)
    print(f"[anneal] total {total_docs:,} docs, {total_tokens/1e6:.0f}M tokens "
          f"({total_tokens/args.steps/TOKENS_PER_STEP:.2f}x the step budget)")
    print(f"[anneal] language by tokens: en {share_en:.1f}% / "
          f"ru {100-share_en:.1f}%")
    if not args.dry_run:
        print(f"[anneal] written to {OUT}")


if __name__ == "__main__":
    main()
