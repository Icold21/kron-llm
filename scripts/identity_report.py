"""Write the identity-slice completion report as Markdown.

Used by ``scripts/night_chain.ps1`` to email the result without a human in the
loop: counts, sizes, filter statistics and two samples.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("data/bilingual")
SHARDS = ROOT / "shards"
LOG = ROOT / "identity_v5.log"
OUT = ROOT / "identity_report.md"

FILES = [("identity_en", "EN", "обычный"),
         ("identity_ru", "RU", "обычный"),
         ("identity_rick_en", "EN", "пастиш"),
         ("identity_rick_ru", "RU", "пастиш")]


def main() -> None:
    rows = []
    total_docs = 0
    total_bytes = 0
    samples: dict[str, str] = {}
    for stem, lang, style in FILES:
        path = SHARDS / f"{stem}.jsonl"
        if not path.exists():
            rows.append((style, lang, 0, 0.0, "нет файла"))
            continue
        docs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        nbytes = sum(len(d["text"].encode("utf-8")) for d in docs)
        rows.append((style, lang, len(docs), nbytes / 1e3, "ok"))
        total_docs += len(docs)
        total_bytes += nbytes
        if lang == "EN" and style not in samples and docs:
            samples[style] = docs[0]["text"]

    skips: Counter[str] = Counter()
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"\[skip\]\s+([^:]+):", line)
            if m:
                skips[m.group(1).strip()] += 1

    lines = ["# Identity-слайс готов", ""]
    lines.append(f"Всего **{total_docs}** документов, **{total_bytes/1e6:.2f} MB** "
                 f"(≈{total_bytes/3.4/1e3:.0f}K токенов при 3.4 байт/токен).")
    lines += ["", "| слайс | язык | документов | объём |", "|---|---|---:|---:|"]
    for style, lang, n, kb, status in rows:
        lines.append(f"| {style} | {lang} | {n} | {kb:.0f} KB |")
    lines.append("")
    lines.append("Факты в текстах: «небольшая языковая модель, созданная в проекте "
                 "KronLLM; компактная и эффективная; веса — кронекеровы "
                 "произведения, отсюда имя». Без чисел, без двуязычности, без "
                 "«не чат-бот».")
    if skips:
        lines += ["", "**Отбраковка по фильтрам:** " + ", ".join(
            f"{k} — {v}" for k, v in skips.most_common())]
    if samples.get("обычный"):
        lines += ["", "> " + samples["обычный"][:260].replace("\n", " ")]
    if samples.get("пастиш"):
        lines += ["", "> " + samples["пастиш"][:260].replace("\n", " ")]
    lines += ["", "Дальше по цепочке: сборка корпуса (дедуп, квоты, held-out), "
                  "токенизация 8K, бины в Kaggle, прогон 10M на двух T4."]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {OUT} ({total_docs} docs, {total_bytes/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
