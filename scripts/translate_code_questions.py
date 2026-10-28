"""Translate English code-task questions into Russian, leaving the code English.

Code is written in English everywhere, so a bilingual code slice does not need a
Russian *solution* -- it needs a Russian *question*.  Translating the question
and keeping the answer verbatim gives the model exactly the pairing it will meet
in practice: a Russian instruction, a Python/C++ answer.

The translator is a small Marian model (``Helsinki-NLP/opus-mt-en-ru``), not an
LLM: it is two orders of magnitude faster on the 1660 Ti, and the text is short
technical prose, the one regime where a dedicated translation model is good.

Code never enters the translator.  The question is split into prose and
non-prose segments (fenced blocks, inline spans, URLs, formulas); only the prose
goes through Marian and the code segments are put back verbatim.  The first
version masked code with placeholders and 80 % of the questions failed the
round-trip check because Marian drops or re-spaces a bracket -- splitting costs
nothing and cannot corrupt the code.

Every translation is cached by the hash of the English question, and the output
shard is always rebuilt from the cache in candidate order.  That way a rerun
never re-translates anything it has already seen, and the shard is a pure
function of (raw shards, cache) rather than of which run happened to write it.

Usage::

    python scripts/translate_code_questions.py --per-source 100 --limit 40
    python scripts/translate_code_questions.py --per-source 40000
    python scripts/translate_code_questions.py --rebuild-only
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "Helsinki-NLP/opus-mt-en-ru"

# Short algorithmic instructions translate best.  Long competitive-programming
# statements are still fine, but they are slower to generate and noisier.
SOURCES = ["magicoder", "self_oss_instruct", "codealpaca", "python_code_18k",
           "opencodereasoning"]

SEGMENT = re.compile(r"(```.*?```|`[^`\n]+`|https?://\S+|\$[^$]+\$)", re.S)
_CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")
_LATIN = re.compile(r"[a-zA-Z]")

MIN_CYRILLIC = 0.45      # share of letters that must be Russian in the prose


def split_segments(text: str) -> List[Tuple[bool, str]]:
    """Split into ``(is_code, text)`` pieces; only prose is translated."""
    parts: List[Tuple[bool, str]] = []
    pos = 0
    for match in SEGMENT.finditer(text):
        if match.start() > pos:
            parts.append((False, text[pos:match.start()]))
        parts.append((True, match.group(0)))
        pos = match.end()
    if pos < len(text):
        parts.append((False, text[pos:]))
    return parts


def cyrillic_share(text: str) -> float:
    cyr = len(_CYRILLIC.findall(text))
    lat = len(_LATIN.findall(text))
    return cyr / max(1, cyr + lat)


def load_translator(device: str):
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).to(device).eval()
    if device == "cuda":
        model = model.half()
    return tok, model, torch


def translate_batch(texts: List[str], tok, model, torch, device: str,
                    max_new: int = 512) -> List[str]:
    if not texts:
        return []
    batch = tok(texts, return_tensors="pt", padding=True, truncation=True,
                max_length=400)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new, num_beams=1,
                             do_sample=False)
    return tok.batch_decode(out, skip_special_tokens=True)


def cache_path() -> Path:
    return Path("E:/kron_sft/translate_cache.jsonl")


def load_cache() -> Dict[str, str]:
    path = cache_path()
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["hash"]] = row["ru"]
    return out


def append_cache(rows: List[dict]) -> None:
    with cache_path().open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def question_hash(question: str) -> str:
    return hashlib.blake2b(question.encode("utf-8"), digest_size=8).hexdigest()


def collect(raw: Path, sources: List[str], per_source: int,
            cache: Dict[str, str], max_chars: int = 700,
            skip_cached: bool = True) -> List[Tuple[str, str, dict]]:
    """Gather ``(hash, question, row)`` candidates in a deterministic order."""
    out: List[Tuple[str, str, dict]] = []
    for name in sources:
        path = raw / f"{name}.jsonl.gz"
        if not path.exists():
            print(f"  [skip] {name}: no shard", flush=True)
            continue
        taken = 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if taken >= per_source:
                    break
                row = json.loads(line)
                question = "\n".join(m["content"] for m in row["messages"]
                                     if m["role"] == "user")
                if not 60 <= len(question) <= max_chars:
                    continue
                taken += 1
                key = question_hash(question)
                if skip_cached and key in cache:
                    continue
                out.append((key, question, row))
        print(f"  [{name}] {taken:,} scanned, {len(out):,} to translate so far",
              flush=True)
    return out


def build_record(question_ru: str, answer: str) -> dict:
    record = {
        "messages": [{"role": "user", "content": question_ru},
                     {"role": "assistant", "content": answer}],
        "lang": "ru", "domain": "code", "task": "code_problem",
        "source": "code_ru_translated", "licence": "Apache-2.0",
    }
    record["bytes"] = sum(len(m["content"].encode("utf-8"))
                          for m in record["messages"])
    return record


def write_shard(out_path: Path, candidates: List[Tuple[str, str, dict]],
                cache: Dict[str, str]) -> Tuple[int, int]:
    """Rebuild the shard from the cache, in candidate order.

    The shard is a function of the cache, not of the run that produced it, so a
    rerun with a larger ``--per-source`` extends the file instead of replacing
    it with whatever this invocation happened to translate.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = kept = 0
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=3) as out:
        for key, _question, row in candidates:
            seen += 1
            ru = cache.get(key)
            if ru is None:
                continue
            answer = "\n".join(m["content"] for m in row["messages"]
                               if m["role"] == "assistant")
            out.write(json.dumps(build_record(ru, answer),
                                 ensure_ascii=False) + "\n")
            kept += 1
    return seen, kept


def write_marker(out_path: Path, seen: int, kept: int, translated: int) -> None:
    marker = out_path.parent / out_path.name.replace(".jsonl.gz", ".done.json")
    marker.write_text(json.dumps({
        "source": "code_ru_translated", "hf": f"translated:{MODEL}",
        "licence": "Apache-2.0", "lang": "ru", "domain": "code",
        "task": "code_problem", "rows_seen": seen, "rows_kept": kept,
        "rows_translated": translated,
        "note": "questions translated to RU by MarianMT, code kept verbatim",
    }, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="E:/kron_sft/raw")
    ap.add_argument("--per-source", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--max-chars", type=int, default=700,
                    help="skip longer questions: slow to translate, noisy")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--rebuild-only", action="store_true",
                    help="write the shard from the cache without touching the GPU")
    ap.add_argument("--out", default="E:/kron_sft/raw/code_ru_translated.jsonl.gz")
    args = ap.parse_args()

    out_path = Path(args.out)
    raw = Path(args.raw)
    cache = load_cache()
    print(f"[cache] {len(cache):,} questions already translated", flush=True)

    if args.rebuild_only:
        candidates = collect(raw, SOURCES, args.per_source, cache,
                             args.max_chars, skip_cached=False)
        seen, kept = write_shard(out_path, candidates, cache)
        write_marker(out_path, seen, kept, 0)
        print(f"[rebuild] {kept:,} of {seen:,} candidates written from cache")
        return

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    tok, model, torch = load_translator(device)
    print(f"[translator] {MODEL} on {device}", flush=True)

    todo = collect(raw, SOURCES, args.per_source, cache, args.max_chars)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[translate] {len(todo):,} questions", flush=True)

    written = 0
    t0 = time.time()
    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        # Flatten every prose segment of the batch into one translation call.
        # ``plan[i]`` is the *absolute* index of segment i in ``flat`` when the
        # segment is prose, and ``None`` when it is code or blank.  The index has
        # to be absolute: rebuilding with a per-question cursor that restarts at
        # zero hands every question the first question's prose while keeping its
        # own answer -- 874 such pairs in the first build, all of them invisible
        # in the aggregate counts and found only by reading them.
        flat: List[str] = []
        plans: List[List[Optional[int]]] = []
        for _, question, _ in chunk:
            plan: List[Optional[int]] = []
            for is_code, text in split_segments(question):
                if is_code or not text.strip():
                    plan.append(None)
                else:
                    plan.append(len(flat))
                    flat.append(text)
            plans.append(plan)

        translated = translate_batch(flat, tok, model, torch, device,
                                     max_new=args.max_new)
        cache_rows = []
        for (key, question, _row), plan in zip(chunk, plans):
            rebuilt = [text if idx is None else translated[idx]
                       for (_is_code, text), idx in zip(split_segments(question),
                                                        plan)]
            ru = "".join(rebuilt).strip()
            written += 1
            prose = "".join(t for c, t in split_segments(ru) if not c)
            if cyrillic_share(prose) < MIN_CYRILLIC or len(ru) < 40:
                continue
            cache_rows.append({"hash": key, "ru": ru})
        append_cache(cache_rows)
        cache.update({row["hash"]: row["ru"] for row in cache_rows})
        if (start // args.batch) % 20 == 0:
            rate = written / max(1e-9, time.time() - t0)
            print(f"  [progress] {written:,}/{len(todo):,} kept "
                  f"{len(cache_rows):,} ({rate:.1f} q/s)", flush=True)

    print(f"[done] translated {written:,} in {time.time() - t0:.0f}s",
          flush=True)

    candidates = collect(raw, SOURCES, args.per_source, cache, args.max_chars,
                         skip_cached=False)
    seen, kept = write_shard(out_path, candidates, cache)
    write_marker(out_path, seen, kept, written)
    print(f"[shard] {kept:,} of {seen:,} candidates written")


if __name__ == "__main__":
    main()
