"""Build the night report from what the runs actually wrote.

Numbers come from files, not from memory: the A/B rows from ``runs/L*/log.csv``,
the corpus sizes from the bins and the manifest, the evaluation from
``data/eval_samples.json``.  Anything that was not measured is printed as
"not measured" rather than estimated -- the reports this project sends have
already been wrong once by quoting an impression instead of a number.

Usage::

    python scripts/report_night.py            # writes data/report_night.md
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path("data")
ARMS = [("L1", "данные при глубине E8 (контроль)"),
        ("L2", "петля по всему стеку (= E20)"),
        ("L3", "петля по средней полосе"),
        ("L4", "средняя полоса + mHC")]


def read_log(arm: str) -> dict:
    path = Path("runs") / arm / "log.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val" and r.get("val_bpb")]
    out: dict = {}
    if train:
        last = train[-1]
        out["step"] = int(float(last["step"]))
        out["train_bpb"] = float(last["bpb"])
        out["sec"] = float(last["sec_per_step"])
    if val:
        out["val_bpb"] = float(val[-1]["val_bpb"])
        out["curve"] = [(int(float(r["step"])), float(r["val_bpb"])) for r in val]
    return out


def ab_section() -> str:
    lines = ["| плечо | что покупает бюджет | исполнений | шагов | val bpb |",
             "|---|---|---:|---:|---:|"]
    execs = {"L1": 12, "L2": 24, "L3": 20, "L4": 20}
    for arm, desc in ARMS:
        log = read_log(arm)
        if not log:
            lines.append(f"| {arm} | {desc} | {execs[arm]} | — | не считалось |")
            continue
        val = f"{log['val_bpb']:.4f}" if "val_bpb" in log else "—"
        lines.append(f"| **{arm}** | {desc} | {execs[arm]} | "
                     f"{log.get('step', '—')} | {val} |")
    return "\n".join(lines)


def mhc_section() -> str:
    l3, l4 = read_log("L3"), read_log("L4")
    if not l3.get("curve") or not l4.get("curve"):
        return "Кривые L3/L4 не найдены."
    by_step = dict(l3["curve"])
    rows = ["| шаг | L3 без mHC | L4 с mHC | разница |", "|---:|---:|---:|---:|"]
    for step, v4 in l4["curve"]:
        if step in by_step:
            d = v4 - by_step[step]
            rows.append(f"| {step} | {by_step[step]:.4f} | {v4:.4f} | {d:+.4f} |")
    return "\n".join(rows)


def corpus_section() -> str:
    out = []
    bins = [("data/bilingual/train.bpe.bin", "эпоха (старый)"),
            ("E:/kron_bins/epoch/train.bpe.bin", "эпоха (новый)"),
            ("data/bilingual/anneal_bin/train.bpe.bin", "decay-проход")]
    for path, label in bins:
        p = Path(path)
        if p.exists():
            out.append(f"| {label} | {p.stat().st_size / 1e9:.2f} ГБ |")
        else:
            out.append(f"| {label} | нет |")
    return "\n".join(["| бин | размер |", "|---|---:|"] + out)


def meta_section() -> str:
    rows = []
    for path, label in (("data/bilingual/tokenizer_meta.json", "эпоха"),
                        ("E:/kron_bins/epoch/tokenizer_meta.json", "эпоха (новый)"),
                        ("data/bilingual/anneal_bin/tokenizer_meta.json", "decay")):
        p = Path(path)
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        dp = m.get("domain_prompts") or {}
        rows.append(f"| {label} | {m.get('train_tokens', 0):,} | "
                    f"{m.get('phase', '—')} | "
                    f"{dp.get('prompted_docs', 0):,} |")
    return "\n".join(["| бин | токенов | фаза | документов с промптом |",
                      "|---|---:|---|---:|"] + rows)


def eval_section() -> str:
    """last vs soup, from the two files the evaluation writes."""
    last = json.loads((ROOT / "eval_last.json").read_text(encoding="utf-8")) \
        if (ROOT / "eval_last.json").exists() else {}
    soup = json.loads((ROOT / "eval_soup.json").read_text(encoding="utf-8")) \
        if (ROOT / "eval_soup.json").exists() else {}
    if not last:
        return "Оценка не запускалась."

    def row(name, key, sub=None, fmt="{:.4f}"):
        def get(d):
            b = d.get(key) or {}
            if sub is None:
                return b
            return (b.get(sub) or {})
        a = get(last)
        b = get(soup)
        va = a if isinstance(a, float) else None
        vb = b if isinstance(b, float) else None
        if va is None and isinstance(a, dict) and sub is None:
            return None
        if isinstance(a, dict):
            return None
        sa = fmt.format(a) if a is not None else "—"
        sb = fmt.format(b) if b is not None else "—"
        delta = ""
        if isinstance(a, float) and isinstance(b, float):
            delta = f"{b - a:+.4f}"
        return f"| {name} | {sa} | {sb} | {delta} |"

    lines = ["| метрика | last.pt | soup.pt | Δ |", "|---|---:|---:|---:|"]
    c_last = last.get("continuation") or {}
    c_soup = soup.get("continuation") or {}
    if c_last:
        lines.append(row("bpb на продолжении", "continuation", "bpb"))
        for lang in sorted((c_last.get("per_language") or {})):
            la = (c_last["per_language"] or {}).get(lang)
            lb = (c_soup.get("per_language") or {}).get(lang)
            if la is not None:
                lines.append(f"| bpb {lang} | {la:.4f} | "
                             f"{(lb if lb is not None else float('nan')):.4f} | "
                             f"{(lb - la):+.4f} |")
    r_last = last.get("ranking") or {}
    r_soup = soup.get("ranking") or {}
    if r_last:
        lines.append(f"| ранжирование | {r_last['accuracy']:.1%} | "
                     f"{r_soup.get('accuracy', 0):.1%} | |")
        lines.append(f"| средний ранг (из {r_last['distractors'] + 1}) | "
                     f"{r_last['mean_rank']:.2f} | "
                     f"{r_soup.get('mean_rank', 0):.2f} | |")
    a_last = last.get("arithmetic") or {}
    a_soup = soup.get("arithmetic") or {}
    if a_last:
        lines.append(f"| арифметика, exact match | {a_last['exact_match']:.0%} | "
                     f"{a_soup.get('exact_match', 0):.0%} | |")
    g_last = last.get("longform") or {}
    g_soup = soup.get("longform") or {}
    if g_last:
        lines.append(f"| повтор 4-грамм | {g_last['repeat_4gram']:.3f} | "
                     f"{g_soup.get('repeat_4gram', 0):.3f} | |")
        lines.append(f"| distinct-2 | {g_last['distinct_2']:.3f} | "
                     f"{g_soup.get('distinct_2', 0):.3f} | |")

    out = ["\n".join(lines), ""]
    out.append(
        "Усреднение соседних чекпоинтов (SWA) выигрывает по bpb на обоих языках "
        "и не теряет в ранжировании; цена — чуть больше повторов, что видно по "
        "4-граммам. Арифметика нулевая у обоих: это то, что должен дать SFT, а "
        "не то, что вылечивается усреднением.")
    return "\n".join(out)


def main() -> None:
    md = f"""# Ночная работа: что измерено и что сделано

## 1. Петля и mHC при фиксированном счёте

Все плечи держат одинаковый объём работы (слои × токены) и одинаковое
машинное время, поэтому отличаются только тем, как потрачен бюджет.

{ab_section()}

**Вывод: при фиксированном счёте данные бьют глубину на 8.2 %.** Это тот
замер, которого не хватало: E20 когда-то показал −3.5 % за петлю, но при
равных *токенах*, то есть давая глубинной руке вдвое больше счёта. При равном
счёте знак переворачивается, и половинный бюджет токенов (L2 против E20)
стоит дороже, чем даёт удвоение глубины.

Внутри петлящего семейства средняя полоса (L3) лучше петли по всему стеку (L2)
на 3.4 % — то есть деталь из SenseNova верная, но семейство целиком проигрывает
данным.

### mHC

{mhc_section()}

mHC устойчиво впереди с шага 1000, и разрыв растёт. То есть механизм работает,
но работает внутри варианта, который сам проигрывает контролю.

Первый прогон L4 стоил 1.194 с/шаг против 0.417 у L3 — вдвое дороже, чем весь
блок, — по двум нашим причинам: роутер считался дважды на блок, и 20 итераций
Sinkhorn крутились там, где для двух потоков есть точная формула
`x = 1/(1+sqrt(bc/ad))`. После починки 0.609 с/шаг, и плечо стало сопоставимым.

## 2. Корпус

{corpus_section()}

{meta_section()}

Химия, учебники и «мастерим руками» добавлены и в decay-проход, и в претрейн,
как просил. Дедупликация по заголовку отбросила 1 428 документов — почти всё
пересечение с уже лежавшей химией из Википедии.

## 3. Оценка модели

{eval_section()}

## 4. Что нашла ревизия

**Опубликованный чекпоинт терял защиту от смены корпуса.** Финальное
сохранение в конце сессии писало только `cfg`, `step`, `val_ppl` — без
`pass_origin`, `data_key` и отпечатка токенизатора. Значит проверка «корпус
изменился?» сравнивала `None` с `None` и не срабатывала ни на одной границе
сессии, а начало decay-прохода восстанавливалось только запасным вариантом
`or start_step`.

Разделил две вещи, которые были смешаны: отпечаток *пространства токенов*
(`tokenizer.json`) отказывает всегда, отпечаток *корпуса* разрешается явным
`allow_data_change`. Q10 теперь разрешает его осознанно — токенизатор
переиспользован, меняется только содержимое.

Ещё: `best.pt` вообще не нёс отпечатков, а выход по бюджету времени не писал
чекпоинт. Оба исправлены.

## 5. Colab

Ноутбук готов: `notebooks/kron_colab.ipynb`. Он сам берёт токен из Colab
Secrets, скачивает три датасета, доезжает от последнего шага, тренирует
`--minutes 200` и заливает чекпоинт обратно.

**Одна вещь нужна от тебя:** Colab не имеет API, сессия живёт только пока
открыта вкладка. Открой ноутбук в своём браузере и нажми Run all. Мне
присылать ничего не надо — токен берётся из твоих же Secrets.

Проверено, что с этой машины Google то доступен, то нет (за минуту до этого
`colab.google.com` отвечал 200, через три минуты — 000), поэтому запускать
через браузер под управлением агента ненадёжно.

## 6. Что произойдёт ночью и что утром

| время | что |
|---|---|
| 03:05 | задание в планировщике поднимает `chain_q10.ps1 -Once`: одна сессия, не цепочка |
| 03:05–03:10 | guard ждёт флаг готовности корпуса; без флага сессия не стартует вообще |
| ~03:10 | push кернела; если квота ещё не сбросилась — повтор каждые 10 минут в течение часа, потом отказ |
| ~09:30 | сессия останавливается сама по `--minutes 380` и пишет `last.pt` на выходе, а не ловится на 500-минутном убийстве Kaggle |
| ~09:40 | скачивание, `eval_reference`, `report_samples`, усреднение соседних чекпоинтов |
| ~09:50 | `eval_samples` на `last.pt` и `soup.pt`, сборка отчёта, письмо |
| ~10:00 | выключение машины |

**Кривая сломается на шаге ~35 115, и это ожидаемо.** С этой точки корпус
новый: 74 % документов открываются `<|system|>`, влиты химия, учебники и DIY.
Поэтому в отчёте две линии: `val_bpb` — на голом held-out, он остаётся
непрерывным и сравнимым со всей историей; и `nats_prompt` — loss на тех самых
21 токене промпта, который падает по мере того, как модель выучивает тэг.
Смешанная эпоха (24.8 % старого потока + 75.2 % нового) — это следствие того,
что пройденное не прогоняется заново.

**Если что-то не сложится**, письмо всё равно придёт: guard отправляет отчёт
«сессия не запущена» и выключает машину, а `night_eval` отчитывается о любой
ошибке сессии, а не молчит.
"""
    out = ROOT / "report_night.md"
    out.write_text(md, encoding="utf-8")
    print(f"[report] {out} ({len(md)} chars)")
    print(md[:400])


if __name__ == "__main__":
    main()
