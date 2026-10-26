"""Audit the identity shards before tokenization.

Checks each document for the failure modes seen in generation: refusals
("I cannot write in the first person as ..."), stage directions, dialogue
format where it does not belong, numbers, framing phrases and name typos.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SHARDS = Path("data/bilingual/shards")
NAMES = ("identity_en", "identity_ru", "identity_rick_en", "identity_rick_ru")

REFUSAL = re.compile(
    r"(I cannot|I can't|I can not|I am unable|As an AI|I am an AI|I'm an AI"
    r"|I'm sorry|I am sorry|I must clarify|I should clarify"
    r"|я не могу|как ИИ|не имею возможности|я не в состоянии"
    r"|не могу писать|не могу представить)", re.I)
DIALOGUE = re.compile(r"(^\s*User:|^\s*Kron:|^\s*—|^\s*Пользователь:|^\s*Крон:)",
                      re.M)
STAGE = re.compile(
    r"(наклон\w*|улыбн\w*|вздохн\w*|приподня\w*|посмотрел\w*|кивнул\w*"
    r"|\*[^*\n]{2,60}\*|\*ик\*|\([^)\n]{20,90}\))", re.I)
NUMBER = re.compile(r"\b\d[\d,.]*\b")
FRAMING = re.compile(
    r"(research model|research prototype|experimental model|not a product"
    r"|not a chatbot|bilingual|исследовательск|не чат-бот|двуязычн)", re.I)
TYPO = re.compile(r"(Кронон|Кронтон|KronLLLM|Kronnecker|CronyLLM|Kronos"
                  r"|Kronite|Kronzic)")


def audit(name: str) -> dict:
    path = SHARDS / f"{name}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    hits = {"refusal": [], "dialogue": [], "stage": [], "number": [],
            "framing": [], "typo": []}
    for i, row in enumerate(rows):
        text = row["text"]
        for key, pat in (("refusal", REFUSAL), ("dialogue", DIALOGUE),
                         ("stage", STAGE), ("number", NUMBER),
                         ("framing", FRAMING), ("typo", TYPO)):
            if pat.search(text):
                hits[key].append((i, text[:120].replace("\n", " ")))
    return {"docs": len(rows), "hits": hits}


def main() -> None:
    report = {}
    for name in NAMES:
        report[name] = audit(name)
        r = report[name]
        line = f"{name}: {r['docs']} docs"
        for key, found in r["hits"].items():
            if found:
                line += f" | {key} {len(found)}"
        print(line)
        for key, found in r["hits"].items():
            if key in ("refusal", "dialogue", "framing", "typo"):
                for idx, sample in found[:3]:
                    print(f"    {key} #{idx}: {sample[:110]}")
    Path("data/bilingual/identity_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
