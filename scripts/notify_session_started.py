"""Mail "the session is queued" with the numbers that decided it.

The chain normally mails at the end of a session; this is the other end -- the
message that says a session exists at all, what it is resuming from, and why
that resume point is not the latest checkpoint.  It is a separate script because
the launch is now driven by a watcher that waits for Kaggle to come back, so
nobody is watching the terminal when it happens.

Usage::

    python scripts/notify_session_started.py --minutes 250
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIL = Path.home() / ".config" / "opencode" / "skills" / "email-control" / "scripts" / "mail.py"
MD = ROOT / "data" / "session_started.md"
HTML = ROOT / "data" / "session_started.html"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta_path = Path(r"E:\kron_bins\epoch\tokenizer_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dp = meta["domain_prompts"]
    tokens = meta["train_tokens"]

    body = f"""# Сессия Q10 поставлена

Ядро `icold21/kron-q10-10m-bilingual-pretraining` запущено на коде **v19** с
бюджетом **{args.minutes} минут** — так, чтобы оно само записалось и вышло до
полуночи.

## С чего стартует

| параметр | значение |
|---|---|
| точка возобновления | `soup_035115.pt`, шаг **35 115** |
| сессия 4 (до 43 895) | **отброшена** |
| корпус | epoch-бин, **{tokens:,}** токенов |
| промпт в документах | {dp["prompted_docs"]:,} добавлено + {dp["already_prompted_docs"]:,} уже чат = **{dp["docs"]:,} / {dp["docs"]:,}** |

## Почему откат

Сессия 4 училась, считая лосс и на наших же вставленных промптах: `<|system|>` плюс
инструкция плюс `<|eot|>` — это 23 токена из 2 303 на документ, около 1 % целей.
На инференсе этот префикс пишет харнесс, модель его никогда не порождает, так что
градиент на нём был потрачен зря, а отчётный лосс из-за него был занижен.

Теперь такие позиции вырезаны из лосса (`kronllm/loss_mask.py`), причём в обеих
ветках — включая chunked-путь `loss_chunk: 1024`, которым Q10 и учится. `<|system|>`
остаётся обучаемой целью: это граница документа, и модель должна знать, где он
начинается.

## Что изменится в цифрах

- тренировочный лосс **подскочит вверх примерно на 0.012 нат** на первом же шаге:
  убирается дешёвый 1 % целей. Это честное направление, а не регресс;
- `val_bpb_mixed` не изменится: held-out бин голый, маска на нём пустая, серия
  остаётся сравнимой со всеми прошлыми сессиями;
- оптимизатора в точке возобновления нет (это усреднённый чекпоинт), поэтому
  первые ~100 шагов пойдут «горячими» — AdamW стартует с нуля.

Отчёт по завершении придёт отдельным письмом.
"""
    MD.write_text(body, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "md_to_html.py"),
                    str(MD), str(HTML)], check=True)

    cmd = [sys.executable, str(MAIL), "send",
           "--subject", f"Q10: сессия поставлена ({args.minutes} мин, откат на шаг 35 115)",
           "--model", "DeepSeek V4.1 Flash Free", "--session", "kron_chain",
           "--html", "--body-file", str(HTML)]
    if args.dry_run:
        cmd.append("--dry-run")
    for attempt in range(1, 4):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0:
            print(f"[mail] sent on attempt {attempt}")
            return
        print(f"[mail] attempt {attempt} failed: {(r.stderr or r.stdout).strip()[-300:]}")
    print("[mail] all attempts failed; the body is in data/session_started.html")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
