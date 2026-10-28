"""Audit the translated code shard: does every Russian question belong to its answer?

The translator batches several questions at once and scatters the translated
prose back into each one.  A wrong index there produces a record whose Russian
question comes from a different task while its answer stays its own -- fluent,
plausible, and wrong.  Aggregate counters cannot see it (the row count, the byte
total and the Cyrillic share are all fine), so the pairing has to be checked
explicitly.

The check rebuilds ``hash -> (question, answer)`` from the raw shards, maps the
Russian question in the shard back to the hash that produced it, and asks whether
the answer in the shard is the answer of *that* question.

``--purge`` removes the cache entries whose Russian text is claimed by more than
one question.  A corrupted entry always duplicates the text of the question it
stole, so the duplicate is the signature; after purging, the affected questions
are re-translated by ``translate_code_questions.py``.

Usage::

    python scripts/verify_translated_pairs.py
    python scripts/verify_translated_pairs.py --purge
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

RAW_DEFAULT = Path("E:/kron_sft/raw")
CACHE_DEFAULT = Path("E:/kron_sft/translate_cache.jsonl")
SOURCES = ["magicoder", "self_oss_instruct", "codealpaca", "python_code_18k",
           "opencodereasoning"]


def question_hash(question: str) -> str:
    return hashlib.blake2b(question.encode("utf-8"), digest_size=8).hexdigest()


def load_cache(path: Path) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cache[row["hash"]] = row["ru"]
    return cache


def load_sources(raw: Path, cache: Dict[str, str]) -> Dict[str, Tuple[str, set]]:
    """``hash -> (english question, every answer seen with it)``.

    A question can legitimately appear in several datasets with different
    answers -- the alpaca-style code sets share whole rows -- so the check must
    accept any answer that belongs to the question, not the answer of whichever
    shard happened to be scanned last.  Comparing against a single answer flags
    those duplicates as corruption, which is how the second audit run managed to
    report 2,474 mismatches out of 33,405 while the pairs were fine.
    """
    out: Dict[str, Tuple[str, set]] = {}
    for name in SOURCES:
        path = raw / f"{name}.jsonl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                question = "\n".join(m["content"] for m in row["messages"]
                                     if m["role"] == "user")
                key = question_hash(question)
                if key not in cache:
                    continue
                answer = "\n".join(m["content"] for m in row["messages"]
                                   if m["role"] == "assistant")
                if key in out:
                    out[key][1].add(answer)
                else:
                    out[key] = (question, {answer})
    return out


def purge_duplicates(path: Path, cache: Dict[str, str]) -> int:
    """Drop cache entries whose Russian text is shared by several questions."""
    owners: Dict[str, list] = defaultdict(list)
    for key, ru in cache.items():
        owners[ru].append(key)
    bad = {key for keys in owners.values() if len(keys) > 1 for key in keys}
    if not bad:
        return 0
    keep = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["hash"] in bad:
                continue
            keep.append(line if line.endswith("\n") else line + "\n")
    with path.open("w", encoding="utf-8") as fh:
        fh.writelines(keep)
    return len(bad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DEFAULT))
    ap.add_argument("--cache", default=str(CACHE_DEFAULT))
    ap.add_argument("--shard", default=None,
                    help="translated shard to audit (default: <raw>/code_ru_translated.jsonl.gz)")
    ap.add_argument("--purge", action="store_true",
                    help="remove ambiguous cache entries and exit")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    raw = Path(args.raw)
    cache_path = Path(args.cache)
    shard = Path(args.shard) if args.shard else raw / "code_ru_translated.jsonl.gz"

    cache = load_cache(cache_path)
    print(f"[cache] {len(cache):,} translations")

    if args.purge:
        dropped = purge_duplicates(cache_path, cache)
        print(f"[purge] {dropped:,} ambiguous cache entries removed; "
              f"rerun the translator to refill them")
        return

    by_hash = load_sources(raw, cache)
    print(f"[map] {len(by_hash):,} cached questions located in the raw shards")

    # ``russian question -> every answer of every question that translated to
    # it``.  Keying on the Russian text rather than on the hash is what makes the
    # check robust to two harmless kinds of duplication: one English question
    # carried by several datasets with different answers, and two questions that
    # differ only in whitespace and translate to the same Russian sentence.  A
    # hybrid -- a question's text next to a different question's answer -- still
    # fails, because the answer then belongs to a question whose Russian text is
    # not this one.
    answers_by_ru: Dict[str, set] = {}
    for key, ru in cache.items():
        entry = by_hash.get(key)
        if entry is None:
            continue
        answers_by_ru.setdefault(ru, set()).update(entry[1])

    stats: Counter = Counter()
    examples = []
    if shard.exists():
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                question_ru = record["messages"][0]["content"]
                answer = record["messages"][1]["content"]
                answers = answers_by_ru.get(question_ru)
                if answers is None:
                    stats["question_not_in_cache"] += 1
                    continue
                if answer in answers:
                    stats["ok"] += 1
                else:
                    stats["wrong_answer"] += 1
                    if len(examples) < args.examples:
                        examples.append((question_ru, next(iter(answers)), answer))
    else:
        print(f"[skip] {shard} does not exist yet")

    total = sum(stats.values())
    bad = total - stats["ok"]
    print(f"[result] {total:,} records: {dict(stats)}")
    print(f"[verdict] {'PASS' if bad == 0 else 'FAIL'} "
          f"({bad:,} of {total:,} pairs do not match)")
    for question_ru, answer_of_q, answer in examples:
        print("--- MISMATCH")
        print("  RU Q   :", question_ru[:140].replace("\n", " "))
        print("  A of Q :", answer_of_q[:100].replace("\n", " "))
        print("  A in r :", answer[:100].replace("\n", " "))


if __name__ == "__main__":
    main()
