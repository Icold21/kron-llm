"""Finish a Kaggle session without a human at the keyboard.

Sequence, each step gated on the previous one:

1. poll the kernel until it is terminal (default deadline: 14 h);
2. download the output to ``E:\\kron_out\\session<N>`` -- ``C:`` has run out of
   space before, and a half-downloaded ``last.pt`` is worse than none;
3. read ``log.csv``: the step, ``val_bpb_mixed`` and sec/step, which are the
   numbers worth quoting (``val_ppl`` in the checkpoint is the legacy metric);
4. publish the checkpoint as a new version of ``kron-q10-ckpt`` and wait until
   the dataset reports ready -- pushing the next session before that would
   resume from the *previous* session's weights;
5. mail the report as HTML;
6. push the next session, unless ``--no-push`` (or the epoch is finished, in
   which case the decay pass is what comes next and is left to a human).

Usage::

    python -u scripts/finish_session.py --start-step 36549 --session 6
    python -u scripts/finish_session.py --start-step 36549 --session 6 --no-push
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = "icold21/kron-q10-10m-bilingual-pretraining"
CKPT_SLUG = "icold21/kron-q10-ckpt"
STAGE = Path(r"E:\kron_ckpt_stage")
MAIL = (Path.home() / ".config" / "opencode" / "skills" / "email-control"
        / "scripts" / "mail.py")
TOKENS_PER_STEP = 16_384
EPOCH_STEPS = 169_884          # the rebuilt corpus, 2,783,383,150 tokens


def log_path(session: int) -> Path:
    return ROOT / "data" / f"finish_session{session}.log"


def say(msg: str, session: int) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with log_path(session).open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def kaggle(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "kaggle", *argv],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def status() -> str:
    r = kaggle("kernels", "status", KERNEL)
    tail = (r.stdout or r.stderr or "").strip().splitlines()
    return tail[-1].strip() if tail else "?"


def read_logs(out: Path) -> tuple[list[dict], list[dict]]:
    cand = sorted(out.rglob("log.csv"))
    if not cand:
        return [], []
    rows = list(csv.DictReader(cand[0].open(encoding="utf-8")))
    return ([r for r in rows if r.get("split") == "train"],
            [r for r in rows if r.get("split") == "val"])


def ckpt_info(out: Path) -> dict:
    import torch
    info: dict = {}
    for name in ("last.pt", "best.pt"):
        cand = sorted(out.rglob(name))
        if not cand:
            continue
        try:
            ck = torch.load(cand[0], map_location="cpu", weights_only=False)
            info[name] = {"path": cand[0], "step": int(ck.get("step", 0)),
                          "val_ppl": ck.get("val_ppl"),
                          "data_key": ck.get("data_key"),
                          "pass_origin": ck.get("pass_origin")}
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline
            info[name] = {"path": cand[0], "error": str(exc)[:120]}
    return info


def publish_ckpt(info: dict, session: int) -> bool:
    """Upload last.pt + best.pt as a new version and wait until it is ready."""
    if "last.pt" not in info or "error" in info["last.pt"]:
        say("no usable last.pt: leaving the dataset alone", session)
        return False
    STAGE.mkdir(parents=True, exist_ok=True)
    for stale in STAGE.glob("*.pt"):
        stale.unlink()
    for name in ("last.pt", "best.pt"):
        if name in info and "error" not in info[name]:
            shutil.copyfile(info[name]["path"], STAGE / name)
    (STAGE / "dataset-metadata.json").write_text(json.dumps({
        "title": "Kron Q10 checkpoint", "id": CKPT_SLUG,
        "licenses": [{"name": "other"}]}), encoding="utf-8")
    step = info["last.pt"]["step"]
    say(f"uploading step {step:,} as a new version of {CKPT_SLUG}", session)
    r = kaggle("datasets", "version", "-p", str(STAGE), "--dir-mode", "zip",
               "-m", f"session {session}: step {step:,}")
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    say(f"  upload exit {r.returncode}: {tail[-1][:120] if tail else '-'}", session)
    if r.returncode != 0:
        return False
    for _ in range(40):
        f = kaggle("datasets", "files", CKPT_SLUG)
        if "last.pt" in (f.stdout or ""):
            say("  dataset reports ready", session)
            return True
        time.sleep(20)
    say("  dataset never reported ready", session)
    return False


def mail_report(session: int, out: Path, train: list[dict], val: list[dict],
                info: dict, start_step: int) -> None:
    last = train[-1] if train else {}
    v = val[-1] if val else {}
    step = info.get("last.pt", {}).get("step")
    sec = float(last.get("sec_per_step") or 0)
    done = (step - start_step) if step else 0
    md = [
        f"# Q10: сессия {session} завершена",
        "",
        f"Старт был с шага **{start_step:,}**, бюджет 660 минут.",
        "",
        "| параметр | значение |",
        "|---|---|",
        f"| шаг чекпоинта | **{step if step else '—'}** |",
        f"| пройдено за сессию | {done:,} шагов ({done * TOKENS_PER_STEP / 1e9:.2f} млрд токенов) |",
        f"| скорость | {sec:.2f} с/шаг |",
        f"| loss (последняя запись) | {last.get('loss', '—')} |",
        f"| **val_bpb_mixed** | **{v.get('val_bpb_mixed') or v.get('bpb') or '—'}** |",
        f"| val_bpb_en / ru | {v.get('val_bpb_en', '—')} / {v.get('val_bpb_ru', '—')} |",
        f"| до конца эпохи | {max(0, EPOCH_STEPS - (step or start_step)):,} шагов |",
        "",
        f"Выгрузка: `{out}`. Чекпоинт опубликован версией `{CKPT_SLUG}`.",
    ]
    md_path = ROOT / "data" / f"report_session{session}.md"
    html_path = ROOT / "data" / f"report_session{session}.html"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "md_to_html.py"),
                    str(md_path), str(html_path)], check=True)
    cmd = [sys.executable, str(MAIL), "send",
           "--subject", f"Q10: сессия {session} завершена, "
                        f"val_bpb_mixed {v.get('val_bpb_mixed', '—')} "
                        f"(шаг {step})",
           "--model", "DeepSeek V4.1 Flash Free", "--session", "kron_chain",
           "--html", "--body-file", str(html_path)]
    for attempt in range(1, 4):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0:
            say(f"  mail sent (attempt {attempt})", session)
            return
        say(f"  mail attempt {attempt} failed", session)
    say("  mail failed; body is in data/report_session*.html", session)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-step", type=int, required=True)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--hours", type=float, default=14.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    s = args.session
    out = Path(args.out) if args.out else Path(rf"E:\kron_out\session{s}")
    say(f"waiting for {KERNEL} (session {s}, started at step {args.start_step:,})", s)
    deadline = time.time() + args.hours * 3600
    kind = "?"
    while time.time() < deadline:
        st = status()
        if st != kind:
            say(f"  status: {st}", s)
            kind = st
        low = st.lower()
        if any(k in low for k in ("complete", "error", "cancel", "fail")):
            break
        time.sleep(180)
    if "complete" not in kind.lower():
        say(f"kernel ended as {kind}", s)

    out.mkdir(parents=True, exist_ok=True)
    r = kaggle("kernels", "output", KERNEL, "-p", str(out))
    say(f"download exit {r.returncode}", s)
    if r.returncode != 0:
        say((r.stderr or r.stdout or "").strip()[-300:], s)

    train, val = read_logs(out)
    info = ckpt_info(out)
    say(f"log rows: {len(train)} train / {len(val)} val | "
        f"last.pt step {info.get('last.pt', {}).get('step')}", s)

    published = publish_ckpt(info, s)
    mail_report(s, out, train, val, info, args.start_step)

    step = info.get("last.pt", {}).get("step", 0)
    if args.no_push:
        say("--no-push: stopping here", s)
        return
    if not published:
        say("checkpoint not published: refusing to start the next session", s)
        return
    if step >= EPOCH_STEPS - 1_000:
        say("the epoch is finished: the next stage is Q10A, left to a human", s)
        return
    if "complete" not in kind.lower():
        say(f"the session did not complete ({kind}): not chaining", s)
        return
    say("pushing the next epoch session", s)
    r = kaggle("kernels", "push", "-p", str(ROOT / "kaggle" / "q10"))
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    say(f"  push exit {r.returncode}: {tail[-1][:140] if tail else '-'}", s)


if __name__ == "__main__":
    main()
