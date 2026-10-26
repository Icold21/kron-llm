"""Compose the final corpus report (identity + corpus + tokenizer + Kaggle).

Usage::

    python scripts/final_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("data/bilingual")
OUT = ROOT / "final_report.md"
KAGGLE = "https://www.kaggle.com/datasets/icold21/kron-corpus-2b"


def gb(n: float) -> float:
    return round(n / 1e9, 2)


def mb(n: float) -> float:
    return round(n / 1e6, 1)


def main() -> None:
    report = json.loads((ROOT / "corpus_report.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "tokenizer_meta.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "identity_audit.json").read_text(encoding="utf-8"))

    lines = ["# KronCorpus 2B: сборка, identity и токенизация", ""]

    # ---------------------------------------------------------------- identity
    lines += ["## Identity", "",
              "| слайс | документов | дефектов |", "|---|---:|---:|"]
    total_docs = 0
    total_bad = 0
    for name, r in audit.items():
        bad = r["defects"]
        total_docs += r["docs"]
        total_bad += bad
        lines.append(f"| {name} | {r['docs']} | {bad} |")
    lines += ["", f"Всего **{total_docs}** документов, дефектов по проверке: "
                  f"**{total_bad}** (цифры, assistant-фразы, диалоговая "
                  "разметка, ремарки, опечатки в имени).", ""]

    # ------------------------------------------------------------------ corpus
    d = report["dedup"]
    lines += ["## Корпус", "",
              f"- текста до дедупа: **{gb(report['total_bytes'])} ГБ**, "
              f"{report['total_docs']:,} документов",
              f"- после MinHash 0.8: **{d['kept']:,}** документов "
              f"({gb(d['kept_bytes'])} ГБ), выброшено {d['dropped']:,}",
              f"- микс: **{gb(report['mixed_total_bytes'])} ГБ**, "
              f"доля русского **{round(report['ru_share'] * 100, 1)} %**", "",
              "| домен | доля |", "|---|---:|"]
    for name, share in sorted(report["domain_share"].items(),
                              key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {round(share * 100, 1)} % |")
    lines.append("")
    lines += ["Недостача (не добита искусственно):", ""]
    for item in report["shortfalls"]:
        lines.append(f"- {item}")
    lines.append("")

    # --------------------------------------------------------------- tokenizer
    lines += ["## Токенизация 8K", "",
              f"- словарь **{meta['vocab_size']}**, chat-ids: "
              "`<|system|>` 8187, `<|user|>` 8188, `<|assistant|>` 8189, "
              "`<|eot|>` 8190, `<|endoftext|>` 8191",
              f"- train: **{meta['train_tokens'] / 1e9:.3f} B** токенов "
              f"({gb(meta['train_bytes'])} ГБ)",
              f"- val: {meta['val_tokens'] / 1e6:.1f} M токенов "
              f"({mb(meta['val_bytes'])} МБ)",
              f"- байт на токен: **{meta['bytes_per_token']:.2f}**", "",
              "| язык | байт/токен | доля токенов |", "|---|---:|---:|"]
    lang = meta.get("language", {})
    for lg in ("en", "ru"):
        if lg in lang:
            share = meta.get("token_share", {}).get(lg, 0)
            lines.append(f"| {lg} | {lang[lg]['bytes_per_token']:.2f} | "
                         f"{round(share * 100, 1)} % |")
    lines += ["", "Кириллица стоит два байта на символ, поэтому байтовая доля "
                  "русского выше токеновой: квоты считаются по байтам, а "
                  "обучается модель на токенах.", ""]

    # ------------------------------------------------------------------ kaggle
    train = ROOT / "train.bpe.bin"
    val = ROOT / "val.bpe.bin"
    lines += ["## Kaggle", "",
              f"- датасет: {KAGGLE}",
              f"- `train.bpe.bin`: {gb(train.stat().st_size)} ГБ" if train.exists()
              else "- `train.bpe.bin`: ещё нет",
              f"- `val.bpe.bin`: {mb(val.stat().st_size)} МБ" if val.exists()
              else "- `val.bpe.bin`: ещё нет", ""]

    # -------------------------------------------------------------------- next
    steps = meta["train_tokens"] / 24576
    lines += ["## Дальше", "",
              f"Q10 (9 938 886 параметров) на двух T4: 24 576 токенов на шаг, "
              f"то есть **{steps:,.0f} шагов** на одну эпоху. Чекпоинт каждые "
              "500 шагов, сессия останавливается по бюджету времени и "
              "продолжается с `last.pt`.", ""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
