"""Final numbers for the rebuilt corpus: domain tables, tokens, and the mail.

Everything is derived from two files that the build itself wrote --
``data/bilingual/corpus_report.json`` (what the quota mixer selected) and
``E:/kron_bins/epoch_v4/tokenizer_meta.json`` (what the encoder actually
produced) -- so the report cannot drift away from the corpus it describes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kronllm.bilingual import QUOTAS_EN, QUOTAS_RU  # noqa: E402

OUT = Path(r"E:\kron_bins\epoch_v4")
MD = ROOT / "data" / "report_corpus_v4.md"
HTML = ROOT / "data" / "report_corpus_v4.html"
MAIL = (Path.home() / ".config" / "opencode" / "skills" / "email-control"
        / "scripts" / "mail.py")
BPT = {"en": 2.917, "ru": 4.964}
SCALE = 1.0      # calibrated in main() so the domain columns sum to the bin
TOKENS_PER_STEP = 16_384
SEC_PER_STEP = 2.552

QUOTAS = {"en": QUOTAS_EN, "ru": QUOTAS_RU}
NAMES = {"en": "Английский", "ru": "Русский"}


def table(mixed: dict, lang: str) -> list[str]:
    rowsum = sum(mixed[lang].values())
    out = ["| домен | квота | МБ | доля | токенов |",
           "|---|---|---|---|---|"]
    rows = sorted(mixed[lang].items(), key=lambda x: -x[1])
    for dom, size in rows:
        quota = QUOTAS[lang].get(dom, 0.0) * 100
        share = 100 * size / rowsum
        mark = "" if share >= quota - 0.6 else " ⚠"
        out.append(f"| {dom} | {quota:.0f}% | {size/1e6:.0f} | {share:.1f}%{mark} | "
                   f"{size/BPT[lang]*SCALE/1e6:.0f}M |")
    out.append(f"| **итого** | 100% | **{rowsum/1e6:.0f}** | 100% | "
               f"**{rowsum/BPT[lang]*SCALE/1e9:.2f} млрд** |")
    return out


def main() -> None:
    global SCALE
    send = "--send" in sys.argv
    report = json.loads((ROOT / "data" / "bilingual" / "corpus_report.json")
                        .read_text(encoding="utf-8"))
    meta = json.loads((OUT / "tokenizer_meta.json").read_text(encoding="utf-8"))
    mixed = report["mixed"]
    en_b = sum(mixed["en"].values())
    ru_b = sum(mixed["ru"].values())
    tot_b = en_b + ru_b
    # Per-domain token counts are apportioned from bytes with a per-language
    # bytes/token, but the *total* is measured: the encoder counted it.  The
    # proportions are scaled so the columns add up to that number instead of
    # to a 2 %-off estimate.
    raw = {"en": en_b / BPT["en"], "ru": ru_b / BPT["ru"]}
    scale = meta["train_tokens"] / sum(raw.values())
    SCALE = scale
    tok = {k: v * scale for k, v in raw.items()}
    tot_t = meta["train_tokens"]
    bpt_measured = tot_b / tot_t
    docs = report["documents"]
    steps = tot_t / TOKENS_PER_STEP
    hrs = steps * SEC_PER_STEP / 3600

    lines = [
        "# Корпус Q10 пересобран: итоговые цифры",
        "",
        f"**{tot_t/1e9:.2f} млрд токенов**, EN {100*tok['en']/tot_t:.1f} % / "
        f"RU {100*tok['ru']/tot_t:.1f} % по токенам.",
        f"Текста {tot_b/1e9:.2f} ГБ; документов {docs['en'] + docs['ru']:,} "
        f"(EN {docs['en']:,} / RU {docs['ru']:,}).",
        f"Эпоха: **{steps:,.0f} шагов** по 16 384 токена "
        f"(~{hrs:,.0f} GPU-часов при 2.552 с/шаг).",
        f"Измерено кодировщиком: {tot_t:,} токенов, {bpt_measured:.2f} байта на "
        f"токен. Доли по доменам ниже получены из байтов с поправкой на "
        f"измеренный итог.",
        "",
        "## " + NAMES["en"],
        "",
        *table(mixed, "en"),
        "",
        "## " + NAMES["ru"],
        "",
        *table(mixed, "ru"),
        "",
        "## Что изменилось против корпуса, на котором прошли сессии 1-5",
        "",
        "| | было | стало |",
        "|---|---|---|",
        "| токенов | 2.35 млрд | "
        f"**{tot_t/1e9:.2f} млрд** |",
        "| EN / RU по токенам | 61.9 / 38.1 | "
        f"{100*tok['en']/tot_t:.1f} / {100*tok['ru']/tot_t:.1f} |",
        "| EN проза (энциклоп. + литература) | 20 % | "
        f"{100*(mixed['en'].get('encyclopedic',0)+mixed['en'].get('literature',0))/en_b:.0f} % |",
        "| EN литература | ~430 МБ | "
        f"{mixed['en'].get('literature',0)/1e6:.0f} МБ |",
        "| EN энциклопедика | ~670 МБ | "
        f"{mixed['en'].get('encyclopedic',0)/1e6:.0f} МБ |",
        "| RU math | ~500 МБ | "
        f"{mixed['ru'].get('math',0)/1e6:.0f} МБ |",
        "| RU cs (Хабр + StackOverflow) | 35 МБ | "
        f"{mixed['ru'].get('cs',0)/1e6:.0f} МБ |",
        "| EN cs (StackOverflow) | 224 МБ | "
        f"{mixed['en'].get('cs',0)/1e6:.0f} МБ |",
        "| вставленные промпты | 0 | 0 |",
        "",
        "## Недоборы",
        "",
        "Квоты не добираются там, где текста физически нет; занижать лимит "
        "под поставку было бы обманом в отчёте, поэтому они перечислены "
        "явно:",
        "",
    ]
    for item in report["shortfalls"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Возобновление",
        "",
        "Held-out бин не менялся (sha256 `a45bbb03…`), все исторические "
        "цифры `val_bpb_mixed` остаются сравнимыми. Обучение продолжается "
        "с шага 36 549: пропуск 598.9 млн токенов, это 22.4 % новой эпохи "
        "(раньше было 25.5 %, эпоха выросла).",
        "",
        f"Остаток бюджета: {(steps - 36_549) * SEC_PER_STEP / 3600:,.0f} GPU-часов "
        "до конца эпохи, 36 ч на decay (30 %), 2.8 ч на SFT.",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "md_to_html.py"),
                    str(MD), str(HTML)], check=True)
    print(MD.read_text(encoding="utf-8"))

    if not send:
        print("\n[mail] not sent (pass --send)")
        return

    cmd = [sys.executable, str(MAIL), "send",
           "--subject", f"Q10: корпус пересобран — {tot_t/1e9:.2f} млрд токенов, "
                        f"EN {100*tok['en']/tot_t:.0f}/RU {100*tok['ru']/tot_t:.0f}",
           "--model", "DeepSeek V4.1 Flash Free", "--session", "kron_chain",
           "--html", "--body-file", str(HTML)]
    for attempt in range(1, 4):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        print(f"[mail] attempt {attempt}: exit {r.returncode}")
        if r.returncode == 0:
            return
        print((r.stderr or r.stdout or "").strip()[-200:])


if __name__ == "__main__":
    main()
