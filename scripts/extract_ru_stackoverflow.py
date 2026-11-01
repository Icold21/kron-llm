"""Turn the Russian Stack Overflow dump into a pretraining shard.

The file is a plain zstd-compressed JSONL (`IlyaGusev/ru_stackoverflow`), so no
`trust_remote_code` is involved: we read `title`, `text_markdown`, `score` and
`answers[]` and write documents in the corpus format.

One document = the question plus its three best answers, which keeps the thread
readable instead of cutting a question away from its answer.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zstandard as zstd

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
SRC = __import__("pathlib").Path(r"E:\kron_sources\ru_stackoverflow.jsonl.zst")
OUT = ROOT / "data" / "bilingual" / "shards" / "hf_ru_so_cs.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=float, default=1600)
    ap.add_argument("--per-answer-chars", type=int, default=4000)
    args = ap.parse_args()

    target = args.target_mb * 1e6
    written = 0
    docs = 0
    kept_ans = 0
    with open(SRC, "rb") as fh, open(OUT, "w", encoding="utf-8") as out:
        raw = zstd.ZstdDecompressor().stream_reader(fh, read_across_frames=True)
        reader = io.TextIOWrapper(raw, encoding="utf-8")
        for line in reader:
            if written >= target:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            answers = rec.get("answers") or []
            answers = [a for a in answers if (a.get("text_markdown") or "").strip()]
            answers.sort(key=lambda a: int(a.get("score") or 0), reverse=True)
            answers = answers[:3]
            body = (rec.get("text_markdown") or "").strip()
            if len(body) < 200 or not answers:
                continue
            parts = [f"# {rec.get('title') or ''}".strip(), body]
            for a in answers:
                text = (a.get("text_markdown") or "").strip()[: args.per_answer_chars]
                parts.append(text)
                kept_ans += 1
            text = "\n\n".join(p for p in parts if p)
            if len(text) < 400:
                continue
            out.write(json.dumps({
                "title": str(rec.get("title") or "")[:200],
                "text": text, "language": "ru", "domain": "cs",
                "source": "RuStackOverflow (HF IlyaGusev/ru_stackoverflow)",
                "licence": "Stack Exchange CC-BY-SA-4.0",
                "url": str(rec.get("url") or "")}, ensure_ascii=False) + "\n")
            written += len(text.encode("utf-8"))
            docs += 1
            if docs % 20000 == 0:
                print(f"  {docs:,} docs, {written/1e6:.0f} MB", flush=True)

    print(f"[done] {docs:,} docs, {written/1e6:.0f} MB, {kept_ans:,} answers "
          f"-> {OUT}")


if __name__ == "__main__":
    main()
