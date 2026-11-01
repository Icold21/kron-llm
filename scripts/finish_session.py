"""Finish a Kaggle session without a human at the keyboard.

Sequence, each step gated on the previous one:

1. poll the kernel until it is terminal (default deadline: 14 h);
2. download the output to ``E:\\kron_out\\session<N>`` -- ``C:`` has run out of
   space before, and a half-downloaded ``last.pt`` is worse than none -- with a
   timeout and a retry, because this link stalls;
3. read ``log.csv``: the step, ``val_bpb_mixed`` and sec/step, which are the
   numbers worth quoting (``val_ppl`` in the checkpoint is the legacy metric);
4. publish the checkpoint as a new version of ``kron-q10-ckpt`` and wait until
   the dataset reports ready -- pushing the next session before that would
   resume from the *previous* session's weights;
5. generate samples from the checkpoint just downloaded and redraw the curve;
6. mail the report as HTML with the curve attached;
7. push the next session, patching the kernel's wall-clock budget first, unless
   ``--no-push`` (or the epoch is finished, in which case the decay pass is what
   comes next and is left to a human).

``--chain N`` repeats steps 1-7 N times back to back, so one command covers the
night; ``--minutes-plan 330,660`` sets the kernel budget of each session it
pushes, which is how a session is made to end at a chosen hour instead of after
a fixed eleven.

Usage::

    python -u scripts/finish_session.py --start-step 36549 --session 6
    python -u scripts/finish_session.py --session 6 --chain 2 --minutes-plan 330,660
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
EPOCH_STEPS = 172_273          # corpus v5: 2,822,521,746 tokens (drills in)
MODEL_NAME = "DeepSeek V4.1 Flash"


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


def run_generation(session: int, out: Path, minutes: int = 25) -> Path | None:
    """Write ``data/samples_session<N>.md`` from the checkpoint just downloaded."""
    ckpt = out / "last.pt"
    if not ckpt.exists():
        say("  no last.pt: skipping the generation", session)
        return None
    dst = ROOT / "data" / f"samples_session{session}.md"
    step = ckpt_info(out).get("last.pt", {}).get("step", "?")
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "generate_samples.py"),
           "--ckpt", str(ckpt), "--out", str(dst),
           "--label", f"сессия {session}, шаг {step}", "--tokens", "60"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=minutes * 60)
        tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        say(f"  generation exit {r.returncode}: "
            f"{tail[-1][:110] if tail else '-'}", session)
        return dst if dst.exists() else None
    except subprocess.TimeoutExpired:
        say(f"  generation timed out after {minutes} min", session)
        return None


def run_plot(session: int) -> Path | None:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_curve.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    say(f"  plot exit {r.returncode}: {tail[-1][:110] if tail else '-'}", session)
    png = ROOT / "data" / "curve_q10.png"
    return png if png.exists() else None


def curve_table(val: list[dict], limit: int = 10) -> list[str]:
    rows = [r for r in val if (r.get("val_bpb_mixed") or "").strip()][-limit:]
    if not rows:
        return ["_в этом логе нет строк валидации_"]
    out = ["| шаг | val_bpb_mixed | EN | RU |", "|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {int(r['step']):,} | {r['val_bpb_mixed']} | "
                   f"{r.get('val_bpb_en', '—')} | {r.get('val_bpb_ru', '—')} |")
    return out


def mail_report(session: int, out: Path, train: list[dict], val: list[dict],
                info: dict, start_step: int, samples: Path | None,
                png: Path | None, minutes: int | None) -> None:
    last = train[-1] if train else {}
    v = val[-1] if val else {}
    step = info.get("last.pt", {}).get("step")
    sec = float(last.get("sec_per_step") or 0)
    done = (step - start_step) if step else 0
    md = [
        f"# Q10: сессия {session} завершена",
        "",
        f"Старт был с шага **{start_step:,}**, бюджет "
        f"{minutes if minutes else '—'} минут.",
        "",
        "| параметр | значение |",
        "|---|---|",
        f"| шаг чекпоинта | **{step if step else '—'}** |",
        f"| пройдено за сессию | {done:,} шагов "
        f"({done * TOKENS_PER_STEP / 1e9:.2f} млрд токенов) |",
        f"| скорость | {sec:.2f} с/шаг |",
        f"| loss (последняя запись) | {last.get('loss', '—')} |",
        f"| **val_bpb_mixed** | **{v.get('val_bpb_mixed') or v.get('bpb') or '—'}** |",
        f"| val_bpb_en / ru | {v.get('val_bpb_en', '—')} / "
        f"{v.get('val_bpb_ru', '—')} |",
        f"| до конца эпохи | {max(0, EPOCH_STEPS - (step or start_step)):,} шагов "
        f"({max(0, EPOCH_STEPS - (step or start_step)) / EPOCH_STEPS:.1%}) |",
        "",
        "## Кривые",
        "",
        f"Валидация за сессию (`val_bpb_mixed` — единственная честная метрика: "
        f"held-out бин не меняется с первого шага, поэтому серия сопоставима "
        f"несмотря на пересборки корпуса):",
        "",
    ]
    md += curve_table(val)
    md += [
        "",
        "График приложен к письму (`curve_q10.png`): сверху train loss, снизу "
        "`val_bpb_mixed` с языками и референсами GPT-2-124M (1.554) и soup на "
        "шаге 35k (1.133). Вертикальные линии — смены корпуса; ступени на "
        "train loss в этих местах ожидаемы, на валидации их нет.",
        "",
    ]
    if samples and samples.exists():
        md += [samples.read_text(encoding="utf-8"), ""]
    md += [f"Выгрузка: `{out}`. Чекпоинт опубликован версией `{CKPT_SLUG}`."]

    md_path = ROOT / "data" / f"report_session{session}.md"
    html_path = ROOT / "data" / f"report_session{session}.html"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "md_to_html.py"),
                    str(md_path), str(html_path)], check=True)
    cmd = [sys.executable, str(MAIL), "send",
           "--subject", f"Q10: сессия {session} завершена, "
                        f"val_bpb_mixed {v.get('val_bpb_mixed', '—')} "
                        f"(шаг {step})",
           "--model", MODEL_NAME, "--session", "kron_chain",
           "--html", "--body-file", str(html_path)]
    if png and png.exists():
        cmd += ["--attach", str(png)]
    for attempt in range(1, 4):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0:
            say(f"  mail sent (attempt {attempt})", session)
            return
        say(f"  mail attempt {attempt} failed", session)
    say("  mail failed; body is in data/report_session*.html", session)


def download_output(out: Path, s: int, attempts: int = 4,
                    timeout: float = 900.0) -> bool:
    """``kaggle kernels output`` with a timeout, a retry and a size check."""
    ckpt = out / "last.pt"
    for i in range(1, attempts + 1):
        for stale in out.glob("*.pt"):
            stale.unlink(missing_ok=True)
        try:
            r = subprocess.run(
                [sys.executable, "-m", "kaggle", "kernels", "output", KERNEL,
                 "-p", str(out)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout)
            tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
            say(f"  download {i}/{attempts}: exit {r.returncode} "
                f"{(tail[-1][:120] if tail else '-')}", s)
            if r.returncode == 0 and ckpt.exists() and ckpt.stat().st_size > 1_000_000:
                return True
        except subprocess.TimeoutExpired:
            say(f"  download {i}/{attempts}: stalled after {timeout:.0f}s", s)
        time.sleep(30)
    return False


def notebook_minutes() -> int | None:
    """The kernel budget this session was pushed with, for the report."""
    nb_path = ROOT / "kaggle" / "q10" / "train.ipynb"
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except OSError:
        return None
    for cell in nb["cells"]:
        for line in cell["source"]:
            if line.strip().startswith("MINUTES ="):
                try:
                    return int(line.split("=", 1)[1].strip().split()[0])
                except (IndexError, ValueError):
                    return None
    return None


def patch_minutes(minutes: int, session: int) -> bool:
    """Set ``MINUTES`` in the kernel notebook before pushing it.

    The session is meant to end at a wall-clock hour, not after a fixed eleven;
    the notebook is the only place the budget lives, and the watcher is what
    pushes it.
    """
    nb_path = ROOT / "kaggle" / "q10" / "train.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    hits = 0
    for cell in nb["cells"]:
        for i, line in enumerate(cell["source"]):
            if line.strip().startswith("MINUTES ="):
                cell["source"][i] = f"MINUTES = {minutes}\n"
                hits += 1
    if hits:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    say(f"  kernel MINUTES -> {minutes} ({hits} cell)", session)
    return bool(hits)


def finish_one(session: int, start_step: int | None, hours: float, out: Path,
               push: bool, next_minutes: int | None) -> int | None:
    """Wait, download, publish, report; return the step the run reached."""
    say(f"waiting for {KERNEL} (session {session}, "
        f"started at {('step ' + format(start_step, ',')) if start_step else 'an unknown step'})",
        session)
    deadline = time.time() + hours * 3600
    kind = "?"
    while time.time() < deadline:
        st = status()
        if st != kind:
            say(f"  status: {st}", session)
            kind = st
        low = st.lower()
        if any(k in low for k in ("complete", "error", "cancel", "fail")):
            break
        time.sleep(180)
    if "complete" not in kind.lower():
        say(f"kernel ended as {kind}", session)

    out.mkdir(parents=True, exist_ok=True)
    if not download_output(out, session):
        say("the output could not be downloaded; the checkpoint is still in the "
            "kernel output on Kaggle and the next session was NOT pushed", session)
        return None

    train, val = read_logs(out)
    info = ckpt_info(out)
    step = info.get("last.pt", {}).get("step", 0)
    if start_step is None:
        # Infer from the log: the first row is where this session began.
        start_step = int(train[0]["step"]) if train else step
        say(f"  start step inferred from log.csv: {start_step:,}", session)
    say(f"log rows: {len(train)} train / {len(val)} val | last.pt step {step}", session)

    published = publish_ckpt(info, session)
    samples = run_generation(session, out)
    png = run_plot(session)
    mail_report(session, out, train, val, info, start_step, samples,
                png, notebook_minutes())

    if not push:
        say("--no-push: stopping here", session)
        return None
    if not published:
        say("checkpoint not published: refusing to start the next session", session)
        return None
    if step >= EPOCH_STEPS - 1_000:
        say("the epoch is finished: the next stage is Q10A, left to a human", session)
        return None
    if "complete" not in kind.lower():
        say(f"the session did not complete ({kind}): not chaining", session)
        return None
    if next_minutes:
        patch_minutes(next_minutes, session)
    say("pushing the next epoch session", session)
    r = kaggle("kernels", "push", "-p", str(ROOT / "kaggle" / "q10"))
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    say(f"  push exit {r.returncode}: {tail[-1][:140] if tail else '-'}", session)
    if r.returncode != 0:
        return None
    wait_for_restart(session)
    return step


def wait_for_restart(session: int, minutes: float = 25.0) -> bool:
    """Poll until the kernel stops reporting the previous run as complete."""
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        st = status().lower()
        if not any(k in st for k in ("complete", "error", "cancel", "fail")):
            say(f"  kernel restarted: {st}", session)
            return True
        time.sleep(30)
    say("  kernel never left the previous terminal state: not chaining further",
        session)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-step", type=int, default=None,
                    help="step the session started from; inferred from log.csv "
                         "when omitted, which is what chaining needs")
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--hours", type=float, default=14.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--chain", type=int, default=1,
                    help="how many sessions to finish back to back")
    ap.add_argument("--minutes-plan", default="",
                    help="comma list of kernel budgets, one per pushed session")
    args = ap.parse_args()

    plan = [int(x) for x in args.minutes_plan.split(",") if x.strip()]
    session = args.session
    start = args.start_step
    for turn in range(args.chain):
        out = Path(args.out) if args.out else Path(rf"E:\kron_out\session{session}")
        step = finish_one(session, start, args.hours, out,
                          push=not args.no_push,
                          next_minutes=plan[turn] if turn < len(plan) else None)
        if step is None:
            break
        start = step
        session += 1
    say("done", session)


if __name__ == "__main__":
    main()
