"""Wait for the Kaggle kernel, download its output, mail the result.

The session was pushed at 22:42 with MINUTES = 65, so it ends on its own around
23:48.  This waits for it, downloads /kaggle/working, and reports the numbers
that are actually in the files:

* the checkpoint step -- proof that it resumed at 35,115 instead of starting
  from zero (a scratch run would be near 1,500);
* ``val_bpb_mixed`` from log.csv, which is the number worth quoting, not the
  legacy ``val_ppl`` in the checkpoint;
* sec/step, so the next session's budget can be planned.

Usage::

    python -u scripts/wait_kernel_then_report.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = "icold21/kron-q10-10m-bilingual-pretraining"
OUT = ROOT / "data" / "kaggle_out" / "session5"
LOG = ROOT / "data" / "wait_kernel.log"
MAIL = (Path.home() / ".config" / "opencode" / "skills" / "email-control"
        / "scripts" / "mail.py")


def say(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def kaggle(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "kaggle", *argv],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def status() -> str:
    r = kaggle("kernels", "status", KERNEL)
    tail = (r.stdout or r.stderr or "").strip().splitlines()
    return tail[-1].strip() if tail else "?"


def read_log() -> tuple[list[dict], list[dict]]:
    path = OUT / "q10" / "log.csv"
    if not path.exists():
        cand = sorted(OUT.rglob("log.csv"))
        if not cand:
            return [], []
        path = cand[0]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    return train, val


def checkpoint_step() -> int | None:
    import torch
    cand = sorted(OUT.rglob("last.pt"))
    if not cand:
        return None
    try:
        return int(torch.load(cand[0], map_location="cpu",
                              weights_only=False).get("step"))
    except Exception as exc:  # noqa: BLE001 - report, never crash the mailer
        say(f"  could not read the checkpoint: {exc}")
        return None


def body(kind: str, train: list[dict], val: list[dict], step: int | None) -> str:
    if kind != "complete":
        return f"""# Q10: сессия на Kaggle завершилась со статусом {kind}

Ядро `{KERNEL}` остановилось не штатно. Выгрузка: `{OUT}`.

Что известно из локальных данных: шаг чекпоинта {step if step is not None else 'не определён'},
строк в логе {len(train)}.

Следующий шаг — посмотреть логи ядра на Kaggle и перезапустить.
"""
    last = train[-1] if train else {}
    first = train[0] if train else {}
    v = val[-1] if val else {}
    steps = int(last.get("step", 0) or 0)
    done = steps - 35115
    sec = float(last.get("sec_per_step") or 0)
    mixed = v.get("val_bpb_mixed") or v.get("bpb") or "—"
    lines = [
        "# Q10: сессия на Kaggle завершена",
        "",
        f"Ядро `{KERNEL}`, бюджет 65 минут. Старт был с шага **35 115** "
        f"(`soup_035115`, сессия 4 отброшена), корпус — **без вставленных "
        f"промптов**: 2 347 352 486 токенов, `<|system|>` только в 63 744 "
        f"чат-документах.",
        "",
        "| параметр | значение |",
        "|---|---|",
        f"| шаг чекпоинта | **{step if step is not None else '—'}** |",
        f"| пройдено за сессию | {done:,} шагов |",
        f"| скорость | {sec:.2f} с/шаг |",
        f"| loss (последняя запись) | {last.get('loss', '—')} |",
        f"| **val_bpb_mixed** | **{mixed}** |",
        "",
    ]
    if steps and step is not None and abs(step - steps) > 5:
        lines += [f"Внимание: чекпоинт на шаге {step}, а последняя строка лога "
                  f"на {steps} — расхождение больше пяти шагов, проверьте выгрузку.", ""]
    if first and last:
        lines += [
            f"Первая запись лога: шаг {first.get('step')}, "
            f"loss {first.get('loss')}. Последняя: шаг {last.get('step')}, "
            f"loss {last.get('loss')}. Лосс на первом шаге после отката "
            f"ожидаемо выше — стартовал свежий AdamW.",
            "",
        ]
    lines += [
        "## Что дальше",
        "",
        "- скачать выгрузку целиком и собрать `best + last` этой фазы в soup;",
        "- следующая сессия продолжит с нового шага; корпус и код уже на Kaggle, "
        "заливка не нужна;",
        "- отдельно: Colab сегодня стартовал, но тренер убивается сигналом −9 "
        "внутри `train()` при любом размере батча — разбираемся отдельно, "
        "на данные и на чекпоинт это не влияет.",
        "",
        f"Выгрузка: `{OUT}`",
    ]
    return "\n".join(lines) + "\n"


def send(subject: str, md_text: str) -> None:
    md = ROOT / "data" / "report_session5.md"
    html = ROOT / "data" / "report_session5.html"
    md.write_text(md_text, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "md_to_html.py"),
                    str(md), str(html)], check=True)
    cmd = [sys.executable, str(MAIL), "send", "--subject", subject,
           "--model", "DeepSeek V4.1 Flash Free", "--session", "kron_chain",
           "--html", "--body-file", str(html)]
    for attempt in range(1, 4):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0:
            say(f"[mail] sent on attempt {attempt}")
            return
        say(f"[mail] attempt {attempt} failed: {(r.stderr or r.stdout).strip()[-200:]}")
    say("[mail] could not send; body is in data/report_session5.html")


def main() -> None:
    say("waiting for the kaggle kernel")
    deadline = time.time() + 3 * 3600
    kind = "?"
    while time.time() < deadline:
        s = status()
        if s != kind:
            say(f"  status: {s}")
            kind = s
        low = s.lower()
        if "complete" in low or "error" in low or "cancel" in low or "fail" in low:
            break
        time.sleep(120)

    label = "complete" if "complete" in kind.lower() else kind
    say(f"kernel finished as {label}; downloading the output")
    OUT.mkdir(parents=True, exist_ok=True)
    r = kaggle("kernels", "output", KERNEL, "-p", str(OUT))
    say(f"  download exit {r.returncode}")
    if r.returncode != 0:
        say("  " + (r.stderr or r.stdout or "").strip()[-300:])

    train, val = read_log()
    step = checkpoint_step()
    say(f"  log rows: {len(train)} train, {len(val)} val | ckpt step {step}")

    md_text = body(label, train, val, step)
    if label == "complete":
        v = val[-1] if val else {}
        subject = (f"Q10: сессия завершена, val_bpb_mixed {v.get('val_bpb_mixed', '—')} "
                   f"(шаг {step})")
    else:
        subject = f"Q10: сессия на Kaggle завершилась со статусом {label}"
    send(subject, md_text)
    say("done")


if __name__ == "__main__":
    main()
