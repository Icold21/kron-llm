"""Compose the final corpus report from the artifacts on disk.

Everything here is measured after the fact: the mix is scanned document by
document, the tokenizer meta comes from the actual bin build, and the identity
numbers come from the audit that uses the generator's own gate.

Usage::

    python scripts/final_report.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path("data/bilingual")
MIX = ROOT / "mix"
OUT = ROOT / "final_report.md"
KAGGLE = "https://www.kaggle.com/datasets/icold21/kron-corpus-2b"


def gb(n: float) -> float:
    return round(n / 1e9, 2)


def mb(n: float) -> float:
    return round(n / 1e6, 1)


def scan_mix() -> dict:
    stats: dict[str, dict] = {"en": {}, "ru": {}}
    total_bytes = 0
    total_docs = 0
    for path in sorted(MIX.glob("*.jsonl.gz")):
        lang, domain = path.name.split(".")[0].split("_")[:2]
        n = b = 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)
                n += 1
                b += len(doc["text"].encode("utf-8"))
        stats.setdefault(lang, {})[domain] = {"docs": n, "bytes": b}
        total_bytes += b
        total_docs += n
    return {"stats": stats, "bytes": total_bytes, "docs": total_docs}


def main() -> None:
    mix = scan_mix()
    meta = json.loads((ROOT / "tokenizer_meta.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "identity_audit.json").read_text(encoding="utf-8"))
    build = json.loads((ROOT / "corpus_report.json").read_text(encoding="utf-8"))

    lines = ["# KronCorpus: сборка, identity и токенизация", ""]

    # ---------------------------------------------------------------- identity
    lines += ["## Identity", "",
              "| слайс | документов | дефектов |", "|---|---:|---:|"]
    total_docs = total_bad = 0
    for name, r in audit.items():
        total_docs += r["docs"]
        total_bad += r["defects"]
        lines.append(f"| {name} | {r['docs']} | {r['defects']} |")
    lines += ["", f"Всего **{total_docs}** документов, дефектов по гейту "
                  f"генератора: **{total_bad}** (цифры, assistant-фразы, "
                  "диалоговая разметка, ремарки, опечатки, дубли).", ""]

    # ------------------------------------------------------------------ corpus
    ru_bytes = sum(d["bytes"] for d in mix["stats"].get("ru", {}).values())
    lines += ["## Корпус", "",
              f"- микс: **{gb(mix['bytes'])} ГБ**, {mix['docs']:,} документов",
              f"- языки по байтам: EN **{100 * (1 - ru_bytes / mix['bytes']):.1f} %**"
              f" / RU **{100 * ru_bytes / mix['bytes']:.1f} %**",
              f"- собрано из {build['shards']} шардов; MinHash выбросил "
              f"{build['dedup']['dropped']:,} дублей на исходной сборке",
              "",
              "| домен | доля |", "|---|---:|"]
    agg: dict[str, int] = {}
    for lang, domains in mix["stats"].items():
        for domain, v in domains.items():
            agg[domain] = agg.get(domain, 0) + v["bytes"]
    for name, byts in sorted(agg.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {100 * byts / max(1, mix['bytes']):.1f} % |")
    lines += ["", "Русский добран полной Википедией и Викитекой после сборки "
                  "по квотам: категорийные шарды не могли дать нужный объём. "
                  "Заголовки, уже вошедшие в категорийные шарды, пропущены.",
              "",
              "Недоборы (честно, без искусственного добивания):", ""]
    for item in build.get("shortfalls", []):
        lines.append(f"- {item}")
    lines.append("")

    # --------------------------------------------------------------- tokenizer
    lines += ["## Токенизация 8K", "",
              f"- словарь **{meta['vocab_size']}**, chat-ids: "
              "`<|system|>` 8187, `<|user|>` 8188, `<|assistant|>` 8189, "
              "`<|eot|>` 8190, `<|endoftext|>` 8191",
              f"- train: **{meta['train_tokens']:,}** токенов "
              f"({gb(meta['train_bytes'])} ГБ)",
              f"- val: {meta['val_tokens'] / 1e6:.1f}M токенов "
              f"({mb(meta['val_bytes'])} МБ)",
              f"- байт на токен: **{meta['bytes_per_token']:.2f}**", "",
              "| язык | байт/токен | доля токенов |", "|---|---:|---:|"]
    for lg in ("en", "ru"):
        if lg in meta.get("language", {}):
            lines.append(f"| {lg} | {meta['language'][lg]['bytes_per_token']:.2f} "
                         f"| {100 * meta['token_share'].get(lg, 0):.1f} % |")
    lines += ["", "Русский стоит ~5 байт на токен против ~2.7 у английского, "
                  "поэтому доля в токенах ниже доли в байтах; квоты считаются "
                  "в байтах, и это записано явно.", ""]

    # ------------------------------------------------------------------ kaggle
    train = ROOT / "train.bpe.bin"
    val = ROOT / "val.bpe.bin"
    lines += ["## Kaggle", "", f"- датасет: {KAGGLE}"]
    if train.exists():
        lines.append(f"- `train.bpe.bin`: {gb(train.stat().st_size)} ГБ")
    if val.exists():
        lines.append(f"- `val.bpe.bin`: {mb(val.stat().st_size)} МБ")
    lines += ["", "## Дальше", "",
              f"Q10 (9 938 886 параметров) на двух T4. Шагов на эпоху: "
              f"**{meta['train_tokens'] // 24576:,}** (24 576 токенов/шаг). "
              "Чекпоинт каждые 500 шагов, сессия останавливается по бюджету "
              "времени и продолжается с `last.pt`.", ""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
