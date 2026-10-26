"""Translate the English identity pastiche into Russian, with strict filters.

Russian generation with a 4B model produces the right attitude but erratic
prose, and a plain translation still picks up stage directions ("*the scientist
leans forward*") and mixed-language artefacts.  This script translates the
clean English slice and keeps only texts that read as first-person Russian.

Usage::

    python scripts/translate_identity.py --samples 850 --max-attempts 1800
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kronllm.bilingual import append_jsonl, normalize, quality, record_shard
from gen_identity import FRAMING, MISSPELL, SPECS, ollama_generate

ROOT = Path("data/bilingual")
SHARDS = ROOT / "shards"
MANIFEST = ROOT / "manifest.jsonl"

PROMPT = (
    "Переведи текст ниже на русский язык. Сохрани смысл, тон и грубоватый "
    "сарказм. Верни только перевод: без пояснений, без заголовков, без "
    "ремарок в звёздочках, без пометок вроде «Тур 1» и без кавычек вокруг "
    "всего текста.\n\n{text}"
)

STAGE = re.compile(
    r"(наклон\w*|улыбн\w*|вздохн\w*|приподня\w*|посмотрел\w*|кивнул\w*|"
    r"произнёс\w*|произнес\w*|пожал\w* плеч\w*|почесал\w*)", re.I)
ARTIFACT = re.compile(r"(\bwait\b|\bokay\b|\btranslation\b|\*\*|Тур\s*\d|Перевод:)", re.I)


def clean_translation(text: str) -> str:
    """Drop the scaffolding the model likes to add around a translation."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            out.append("")
            continue
        if re.match(r"^\*+.*\*+$", s):
            continue
        if re.match(r"^(\*\*)?(тур|turn|перевод|translation)\s*\d*", s.lower()):
            continue
        if s.lower().startswith("перевод:"):
            s = s.split(":", 1)[1].strip()
        out.append(s)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > 1 and text[0] in "«\"“" and text[-1] in "»\"”":
        text = text[1:-1].strip()
    return text


def accept(text: str, min_chars: int) -> bool:
    if len(text) < min_chars:
        return False
    if "Крон" not in text and "Kron" not in text:
        return False
    if SPECS.search(text) or FRAMING.search(text) or MISSPELL.search(text):
        return False
    if STAGE.search(text) or ARTIFACT.search(text):
        return False
    if len(re.findall(r"[.!?…]", text)) < 2:
        return False
    return quality(text, "ru", min_chars=min_chars).ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=850)
    ap.add_argument("--max-attempts", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=501)
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()

    src_path = SHARDS / "identity_rick_en.jsonl"
    src = [json.loads(l) for l in src_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[translate] {len(src)} English documents, target {args.samples}")

    docs: list[dict] = []
    attempts = 0
    i = 0
    while len(docs) < args.samples and attempts < args.max_attempts:
        row = src[i % len(src)]
        i += 1
        attempts += 1
        try:
            raw = ollama_generate(PROMPT.format(text=row["text"]),
                                  seed=args.seed + attempts, num_predict=320)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [warn] {attempts}: {exc}")
            time.sleep(2)
            continue
        text = clean_translation(normalize(raw))
        if not accept(text, args.min_chars):
            continue
        docs.append({"title": f"identity/rick/ru/{len(docs):03d}", "text": text,
                     "language": "ru", "domain": "identity",
                     "source": "synthetic: qwen3.5:4b (translated from EN)",
                     "licence": "CC0-1.0",
                     "url": "https://github.com/Icold21/kron-llm",
                     "note": "translated pastiche, no numeric facts"})
        if len(docs) % 25 == 0:
            print(f"  {len(docs)} kept from {attempts} attempts", flush=True)

    if not docs:
        print("[warn] nothing passed")
        return
    out = SHARDS / "identity_rick_ru.jsonl"
    if out.exists():
        out.unlink()
    n = append_jsonl(out, docs)
    shard = record_shard(out, MANIFEST,
                         source="synthetic: qwen3.5:4b (translated from EN)",
                         licence="CC0-1.0",
                         url="https://github.com/Icold21/kron-llm",
                         language="ru", domain="identity", docs=n,
                         note="rick-style pastiche, translated from the EN slice")
    print(f"[done] {n} docs from {attempts} attempts, {shard.bytes/1e3:.0f} KB")


if __name__ == "__main__":
    main()
