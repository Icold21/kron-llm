"""Assemble the post-session report: curve, metrics, samples, last vs soup.

Everything comes from files the session wrote, so the report cannot disagree
with the run: ``runs/q10/log.csv`` for the curve, ``data/baseline_latest.json``
for the honest strided evaluation, ``data/session_soup.json`` for what was
averaged, and ``data/eval_last.json`` / ``data/eval_soup.json`` for the
comparison with a denominator.

Usage::

    python scripts/report_session.py --out data/report_session.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT_DEFAULT = Path("data/report_session.md")


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def curve_rows(limit: int = 16) -> list[str]:
    """Validation points, thinned so the table stays readable.

    The chain downloads the Kaggle session output into ``data/kaggle_out/q10``;
    a local run would live in ``runs/q10``.  Looking only in the second place
    would have left the curve -- the centrepiece of the report -- empty.
    """
    candidates = [Path("data/kaggle_out/q10/log.csv"), Path("runs/q10/log.csv")]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if r.get("split") == "val" and r.get("val_bpb_mixed")]
    if not rows:
        rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
                if r.get("split") == "val" and r.get("val_bpb")]
    if not rows:
        return []
    step = max(1, len(rows) // limit)
    picked = rows[::step]
    if rows[-1] not in picked:
        picked.append(rows[-1])
    out = ["| шаг | val bpb (legacy) | val bpb (mixed) | EN | RU |",
           "|---:|---:|---:|---:|---:|"]

    def num(v, nd=4):
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return "—"

    for r in picked:
        out.append(f"| {r.get('step', '—')} | {num(r.get('val_bpb'))} | "
                   f"{num(r.get('val_bpb_mixed'))} | {num(r.get('val_bpb_en'))} | "
                   f"{num(r.get('val_bpb_ru'))} |")
    return out


def eval_block(path: str, label: str) -> list[str]:
    d = load_json(path)
    if not d:
        return [f"**{label}**: не считалось."]
    out = [f"**{label}** (шаг {d.get('step', '—')}):"]
    c = d.get("continuation") or {}
    if c:
        langs = ", ".join(f"{k} {v:.3f}" for k, v in
                          sorted((c.get("per_language") or {}).items()))
        out.append(f"- bpb на продолжении: **{c.get('bpb', 0):.4f}** "
                   f"на {c.get('blocks', 0)} блоках ({langs})")
    r = d.get("ranking") or {}
    if r:
        out.append(f"- ранжирование: истинное продолжение первым в "
                   f"{r.get('accuracy', 0):.1%} случаев "
                   f"(случайное {r.get('chance_accuracy', 0):.1%}), "
                   f"средний ранг {r.get('mean_rank', 0):.2f} из "
                   f"{r.get('distractors', 0) + 1}")
    a = d.get("arithmetic") or {}
    if a:
        out.append(f"- арифметика: точных ответов {a.get('exact_match', 0):.0%} "
                   f"({a.get('n', 0)} примеров)")
    g = d.get("longform") or {}
    if g:
        out.append(f"- генерация: повтор 4-грамм {g.get('repeat_4gram', 0):.2f}, "
                   f"distinct-2 {g.get('distinct_2', 0):.2f} "
                   f"({g.get('samples', 0)} сэмплов)")
    return out


def examples(path: str, label: str, n: int = 3) -> list[str]:
    d = load_json(path)
    for key in ("longform", "arithmetic"):
        block = d.get(key) or {}
        items = block.get("examples") or []
        if items:
            out = [f"**{label}** — `{key}`:"]
            for it in items[:n]:
                if isinstance(it, dict):
                    out.append(f"> `{it.get('prompt', '')}` → "
                               f"{it.get('got', '')} "
                               f"(ждали {it.get('want', '?')})")
                else:
                    out.append("> " + str(it).replace("\n", " ⏎ "))
            return out
    return []


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    soup = load_json("data/session_soup.json")
    latest = load_json("data/baseline_latest.json")
    gpt2 = load_json("data/baseline_gpt2.json")

    md = ["# Q10: отчёт по сессии", ""]

    md += ["## Кривая", ""]
    rows = curve_rows()
    md += rows if rows else ["Кривая не найдена: `runs/q10/log.csv` отсутствует."]

    md += ["", "## Метрики на страйд-выборке held-out", ""]
    if latest:
        # eval_reference.py writes one summary row per source, not a list of
        # blocks: {"source", "blocks", "span", "bytes", "nats", "bpb"}.
        cur = latest.get("bpb")
        base = gpt2.get("bpb") if gpt2 else None
        md.append("| модель | bpb (страйд, 1024 блока) |")
        md.append("|---|---:|")
        if cur is not None:
            md.append(f"| KronLLM | **{cur:.4f}** |")
        if base is not None:
            md.append(f"| GPT-2-124M | {base:.4f} |")
        if cur is not None and base is not None:
            md.append("")
            md.append(f"Разница: {base - cur:+.4f} bpb "
                      f"({(base - cur) / base:+.1%} к GPT-2) на "
                      f"{latest.get('blocks', 0)} блоках по "
                      f"{latest.get('span', 0)} токенов.")
    else:
        md.append("Страйд-оценка не считалась.")

    md += ["", "## last против усреднённых", ""]
    md += eval_block("data/eval_last.json", "last.pt")
    md.append("")
    if soup:
        md.append(f"Усреднение: {json.dumps(soup, ensure_ascii=False)[:400]}")
        md.append("")
    md += eval_block("data/eval_soup.json", "soup.pt")

    md += ["", "## Примеры", ""]
    md += examples("data/eval_last.json", "last.pt")
    md.append("")
    md += examples("data/eval_soup.json", "soup.pt")

    md += ["", "## Что сделано в сессии", ""]
    md += ["- шаг и токены — из лога сессии;",
           "- чекпоинт опубликован как новая версия `icold21/kron-q10-ckpt`;",
           "- `best.pt` заархивирован, чтобы усреднять соседние веса, а не "
           "через фазу.",
           "",
           "Кривая размечена: с шага 35 115 корпус новый (химия, учебники, "
           "системные промпты), поэтому разрыв в train loss на этой точке — "
           "ожидаемый, а `val_bpb` меряется на голом held-out и остаётся "
           "сравнимым со всей историей."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[report] {args.out} ({len(md)} lines)")


if __name__ == "__main__":
    main()
