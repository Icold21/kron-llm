"""Measure how far the student tokenizer is from a math-teacher tokenizer.

Token-level distillation needs both models to score the same units.  Our model
uses an 8K byte-level BPE over raw bytes; Qwen2.5-Math uses a ~150K
sentencepiece/BPE vocabulary with its own normalisation.  This script measures
the mismatch on real SFT text so ``docs/distillation.md`` can argue with numbers:

* bytes/token and characters/token for each tokenizer, per language;
* the share of *our* tokens that are exactly one teacher token (easy to align)
  and the share that straddle a teacher boundary (need segment scoring);
* the number of teacher tokens behind one of our tokens, on average.

Usage::

    python scripts/tokenizer_alignment.py --sample 2000
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TEACHERS = {
    "Qwen2.5-Math-1.5B": "Qwen/Qwen2.5-Math-1.5B",
    "DeepSeek-Math-7B": "deepseek-ai/deepseek-math-7b-instruct",
}


def load_sft_texts(path: Path, limit: int):
    """Yield (lang, text) pairs from a raw SFT shard or the arithmetic file."""
    by_lang = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            lang = row.get("lang")
            text = "\n".join(m["content"] for m in row["messages"])
            by_lang.setdefault(lang, []).append(text)
            if sum(len(v) for v in by_lang.values()) >= limit:
                break
    return by_lang


def teacher_offsets(tokens):
    """Byte offsets of teacher tokens when the pieces are byte-decodable."""
    offsets, pos = [], 0
    for piece in tokens:
        raw = piece.encode("utf-8", errors="replace")
        offsets.append((pos, pos + len(raw)))
        pos += len(raw)
    return offsets


def compare(student: Tokenizer, teacher, texts, lang: str) -> dict:
    """Count how our tokens line up with the teacher's on the same text."""
    s_enc = [e.ids for e in student.encode_batch(texts)]
    t_enc = teacher(texts, add_special_tokens=False)["input_ids"]
    student_tokens = teacher_tokens = 0
    exact = straddle = multi = 0
    ratio = []
    s_bytes = t_bytes = 0
    for ids, tids in zip(s_enc, t_enc):
        s_pieces = [student.id_to_token(i) or "" for i in ids]
        t_pieces = teacher.convert_ids_to_tokens(tids)
        # Byte strings behind the tokens; the student is a byte-level BPE, so
        # its pieces map to bytes through the GPT-2 byte table.
        s_bounds = _student_bounds(s_pieces)
        t_bounds = teacher_offsets(t_pieces)
        s_bytes += s_bounds[-1][1] if s_bounds else 0
        t_bytes += t_bounds[-1][1] if t_bounds else 0
        student_tokens += len(s_bounds)
        teacher_tokens += len(t_bounds)
        j = 0
        for start, end in s_bounds:
            while j < len(t_bounds) and t_bounds[j][1] <= start:
                j += 1
            # How many teacher tokens cover this student token?
            cover = 0
            k = j
            while k < len(t_bounds) and t_bounds[k][0] < end:
                cover += 1
                k += 1
            if cover == 0:
                straddle += 1
            elif cover == 1 and _same_span((start, end), t_bounds[j]):
                exact += 1
            elif cover == 1:
                straddle += 1
            else:
                multi += 1
        if t_bounds:
            ratio.append(len(s_bounds) / len(t_bounds))
    total = max(1, student_tokens)
    return {
        "lang": lang,
        "texts": len(texts),
        "student_tokens": student_tokens,
        "teacher_tokens": teacher_tokens,
        "student_bytes_per_token": s_bytes / total,
        "teacher_bytes_per_token": t_bytes / max(1, teacher_tokens),
        "exact_share": exact / total,
        "straddle_share": straddle / total,
        "multi_share": multi / total,
        "tokens_per_teacher_token": statistics.fmean(ratio) if ratio else 0.0,
    }


_BYTE_DECODER = None


def _student_bounds(pieces):
    """Byte spans of the student's tokens, using the GPT-2 byte alphabet."""
    global _BYTE_DECODER
    if _BYTE_DECODER is None:
        bs = list(range(ord("!"), ord("~") + 1)) + list(
            range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        _BYTE_DECODER = {chr(c): b for b, c in zip(bs, cs)}
    offsets, pos = [], 0
    for piece in pieces:
        if piece.startswith("<|") and piece.endswith("|>"):
            continue
        raw = bytes(_BYTE_DECODER[ch] for ch in piece if ch in _BYTE_DECODER)
        offsets.append((pos, pos + len(raw)))
        pos += len(raw)
    return offsets


def _same_span(a, b) -> bool:
    return a[0] == b[0] and a[1] == b[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="E:/kron_sft/raw/arith_en.jsonl.gz")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--teacher", default="Qwen2.5-Math-1.5B")
    args = ap.parse_args()

    student = Tokenizer.from_file(str(Path("data/bilingual/tokenizer.json")))
    teacher = AutoTokenizer.from_pretrained(TEACHERS[args.teacher])
    texts = load_sft_texts(Path(args.shard), args.sample)
    print(f"# student 8K byte-BPE vs teacher {args.teacher}\n")
    print("| lang | texts | student b/t | teacher b/t | exact | straddle "
          "| multi | S tokens / T token |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for lang, items in sorted(texts.items()):
        row = compare(student, teacher, items, lang)
        print(f"| {row['lang']} | {row['texts']} | "
              f"{row['student_bytes_per_token']:.2f} | "
              f"{row['teacher_bytes_per_token']:.2f} | "
              f"{row['exact_share']:.1%} | {row['straddle_share']:.1%} | "
              f"{row['multi_share']:.1%} | "
              f"{row['tokens_per_teacher_token']:.2f} |")


if __name__ == "__main__":
    main()
