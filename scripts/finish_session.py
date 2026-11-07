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
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
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
PUSH_DIR = Path("kaggle") / "q10"
MINUTES_RE = re.compile(r"^(\s*MINUTES\s*=\s*)(\d+)(.*)$")
THEN_PUSH_DIR = None            # None means "the next stage is left to a human"
THEN_MINUTES = 0


def log_path(session: int) -> Path:
    return ROOT / "data" / f"finish_session{session}.log"


def say(msg: str, session: int) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    try:
        with log_path(session).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        # The log file is a convenience; stdout is the record the harness keeps.
        # A held-open file (Tee-Object does exactly that) must not end the run.
        pass


def kaggle(*argv: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
    """Run the Kaggle CLI, and never wait on it forever.

    This link stalls: one upload sat 2 h 20 min, and a `kernels status` call
    blocked the whole watcher for over an hour on 2026-11-02.  A timeout turns
    that into a failed call the loop can report and retry.
    """
    try:
        return subprocess.run([sys.executable, "-m", "kaggle", *argv],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=list(argv), returncode=1, stdout="",
            stderr=f"timeout after {timeout:.0f}s")


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
    local_bytes = (STAGE / "last.pt").stat().st_size
    upload_started = datetime.utcnow()
    say(f"uploading step {step:,} as a new version of {CKPT_SLUG}", session)
    r = kaggle("datasets", "version", "-p", str(STAGE), "--dir-mode", "zip",
               "-m", f"session {session}: step {step:,}")
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    say(f"  upload exit {r.returncode}: {tail[-1][:120] if tail else '-'}", session)
    if r.returncode != 0:
        return False
    # "last.pt is listed" was already true before this upload and the listing
    # lags the version, so the old check passed on the previous file.  Wait for
    # the new row instead: same byte count, creation date after the upload began.
    deadline = time.time() + 40 * 20
    while time.time() < deadline:
        f = kaggle("datasets", "files", CKPT_SLUG)
        for line in (f.stdout or "").splitlines():
            if "last.pt" not in line:
                continue
            cols = line.split()
            try:
                size = int(cols[1])
                stamp = datetime.strptime(f"{cols[2]} {cols[3]}",
                                          "%Y-%m-%d %H:%M:%S.%f")
            except (IndexError, ValueError):
                continue
            if size == local_bytes and stamp >= upload_started - timedelta(minutes=2):
                say(f"  the new version is live ({size:,} bytes, {stamp} UTC)",
                    session)
                # The file listing can lead the version pointer by a moment, and
                # the next stage mounts this dataset when it is pushed.
                time.sleep(60)
                return True
        time.sleep(20)
    say("  the new version never appeared in the file listing", session)
    return False


def run_generation(session: int, out: Path, minutes: int = 25) -> Path | None:
    """Write ``data/samples_session<N>.md`` from the checkpoint just downloaded."""
    found = sorted(out.rglob("last.pt"), key=lambda q: q.stat().st_mtime)
    if not found:
        say("  no last.pt: skipping the generation", session)
        return None
    ckpt = found[-1]
    dst = ROOT / "data" / f"samples_session{session}.md"
    step = ckpt_info(out).get("last.pt", {}).get("step", "?")
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "generate_samples.py"),
           "--ckpt", str(ckpt), "--out", str(dst),
           "--label", f"сессия {session}, шаг {step}"]
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


OUT_ROOT = Path("E:/kron_out")

SLICES = ["val_bpb_mixed", "val_bpb_en", "val_bpb_ru", "val_bpb_code",
          "val_bpb_cs", "val_bpb_encyclopedic", "val_bpb_humour",
          "val_bpb_identity", "val_bpb_literature", "val_bpb_logic",
          "val_bpb_math", "val_bpb_ml", "val_bpb_science"]


def previous_results(step: int):
    """Newest results.json from an earlier step, as (step, dict)."""
    best = None
    for q in OUT_ROOT.glob("session*/**/results.json"):
        try:
            d = json.loads(q.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        st = int(d.get("steps") or 0)
        if 0 < st < step and (best is None or st > best[0]):
            best = (st, d)
    return best


def slice_table(step: int, current: dict) -> list[str]:
    """Where the checkpoint moved against the previous one, per slice.

    A mixed bpb can rise while several slices fall: that is a shift in where the
    loss sits, not a uniform regression, and the table is the only place the
    difference is visible.  Negative delta = better.
    """
    prev = previous_results(step)
    if not prev:
        return []
    prev_step, prev_d = prev
    rows = [f"| срез | шаг {prev_step:,} | шаг {step:,} | Δ |", "|---|---|---|---|"]
    for key in SLICES:
        a, b = prev_d.get(key), current.get(key)
        if a is None or b is None:
            continue
        name = key.replace("val_bpb_", "") or "mixed"
        rows.append(f"| {name} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    if len(rows) == 2:
        return []
    return (["## Что сдвинулось с прошлого чекпойнта", "",
             "Δ < 0 — лучше. Разные знаки в одном столбце значат смену "
             "распределения, а не общий регресс.", ""] + rows + [""])


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
        f"| до конца прогона | {max(0, EPOCH_STEPS - (step or start_step)):,} шагов "
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
    md += slice_table(int(step or 0), v)
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
    def complete() -> bool:
        # The trainer writes into /kaggle/working/q10, so the files come back
        # under <out>/q10/ -- a top-level check misses them entirely.
        return any(p.stat().st_size > 1_000_000 for p in out.rglob("last.pt"))

    for i in range(1, attempts + 1):
        for stale in out.rglob("*.pt"):
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
            if r.returncode == 0 and complete():
                say("  checkpoint on disk", s)
                return True
        except subprocess.TimeoutExpired:
            say(f"  download {i}/{attempts}: stalled after {timeout:.0f}s", s)
        time.sleep(30)
    return False


def notebook_minutes() -> int | None:
    """The kernel budget this session was pushed with, for the report."""
    nb_path = ROOT / PUSH_DIR / "train.ipynb"
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


def minutes_until(end_by: str, session: int, reserve: int = 35) -> int | None:
    """Training budget that still ends the *kernel* by ``end_by`` (HH:MM).

    A fixed ``--minutes-plan`` cannot honour a wall-clock deadline: the push is
    followed by a Kaggle queue (25 min the first Saturday after a quota reset),
    the notebook copies the bin before the trainer starts, and the checkpoint has
    to be written before the kernel ends.  ``reserve`` covers those three, so the
    budget is what is left of the clock after them.
    """
    try:
        hh, mm = (int(x) for x in end_by.split(":"))
    except ValueError:
        say(f"  --end-by {end_by!r} is not HH:MM: ignoring it", session)
        return None
    now = datetime.now()
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    budget = int((target - now).total_seconds() // 60) - reserve
    if budget > 700:
        # The deadline is further away than one session can run (or it already
        # passed and rolled to tomorrow): the platform cap is what applies.
        say(f"  {end_by} is more than a session away: capping the budget at 700",
            session)
        budget = 700
    say(f"  the next session must end by {end_by}: {budget} min of training "
        f"after {reserve} min of queue, copy and teardown", session)
    return budget


def patch_minutes(minutes: int, session: int, push_dir: Path | None = None) -> bool:
    """Set ``MINUTES`` in the kernel notebook before pushing it.

    The session is meant to end at a wall-clock hour, not after a fixed eleven;
    the notebook is the only place the budget lives, and the watcher is what
    pushes it.  ``push_dir`` is the directory actually being pushed -- the decay
    chain pushes ``kaggle/q10a``, and patching ``kaggle/q10`` while pushing the
    other one silently keeps the old budget.
    """
    nb_path = ROOT / (push_dir or PUSH_DIR) / "train.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    hits = 0
    for cell in nb["cells"]:
        for i, line in enumerate(cell["source"]):
            m = MINUTES_RE.match(line.rstrip("\n"))
            if m:
                # Only the number changes: the explanation next to it stays.
                cell["source"][i] = f"{m.group(1)}{minutes}{m.group(3)}\n"
                hits += 1
    if hits:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    say(f"  {nb_path.relative_to(ROOT)} MINUTES -> {minutes} ({hits} cell)",
        session)
    return bool(hits)


def finish_one(session: int, start_step: int | None, hours: float, out: Path,
               push: bool, next_minutes: int | None,
               end_by: str | None = None) -> int | None:
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
        if not THEN_PUSH_DIR:
            say("this stage is finished: the next one is left to a human", session)
            return None
        say(f"this stage is finished at step {step:,}: pushing the next stage "
            f"from {THEN_PUSH_DIR}", session)
        if THEN_MINUTES:
            patch_minutes(THEN_MINUTES, session, THEN_PUSH_DIR)
        r = kaggle("kernels", "push", "-p", str(ROOT / THEN_PUSH_DIR))
        text = (r.stdout or "") + (r.stderr or "")
        tail = text.strip().splitlines()
        say(f"  push exit {r.returncode}: {tail[-1][:140] if tail else '-'}",
            session)
        low = text.lower()
        if r.returncode != 0 or "error" in low or "quota" in low:
            say("  the next stage was NOT pushed: stopping here", session)
            return None
        say(f"  {THEN_PUSH_DIR} pushed; it is a different kernel, so this watch "
            f"ends here -- start a watcher for it", session)
        return None
    if "complete" not in kind.lower():
        say(f"the session did not complete ({kind}): not chaining", session)
        return None
    if end_by:
        budget = minutes_until(end_by, session)
        if budget is not None and budget < 25:
            say(f"  under 25 min before {end_by}: not pushing this session, it "
                f"would start and be cut almost at once", session)
            return None
        if budget is not None:
            next_minutes = budget
    if next_minutes:
        patch_minutes(next_minutes, session)
    say("pushing the next session", session)
    r = kaggle("kernels", "push", "-p", str(ROOT / PUSH_DIR))
    text = (r.stdout or "") + (r.stderr or "")
    tail = text.strip().splitlines()
    say(f"  push exit {r.returncode}: {tail[-1][:140] if tail else '-'}", session)
    low = text.lower()
    if r.returncode != 0 or "error" in low or "quota" in low:
        # A quota refusal comes back as exit 0 with "Kernel push error: Maximum
        # weekly GPU quota ..." on stdout, so the return code alone says nothing.
        if "quota" in low:
            say("  the weekly GPU quota is spent: this stage waits for the reset",
                session)
        say("  the next session was NOT pushed: stopping the chain", session)
        return None
    if not wait_for_restart(session):
        return None
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
    ap.add_argument("--kernel", default=None,
                    help="kernel slug to watch (default: the epoch kernel)")
    ap.add_argument("--push-dir", default="kaggle/q10",
                    help="directory whose kernel-metadata.json gets pushed")
    ap.add_argument("--then-push-dir", default=None,
                    help="push this directory once the current stage reaches its "
                         "final step (the decay pass hands off to S10)")
    ap.add_argument("--then-minutes", type=int, default=0,
                    help="kernel budget for that next stage, in minutes")
    ap.add_argument("--total-steps", type=int, default=None,
                    help="step at which this stage is done; the epoch by "
                         "default, the decay pass's own end otherwise")
    ap.add_argument("--minutes-plan", default="",
                    help="comma list of kernel budgets, one per pushed session")
    ap.add_argument("--end-by", default="",
                    help="comma list of HH:MM deadlines, one per pushed session: "
                         "the budget is then computed from the clock so the "
                         "session cannot run past that hour (empty = use the "
                         "minutes plan for that slot)")
    args = ap.parse_args()

    if args.kernel:
        globals()["KERNEL"] = args.kernel
    globals()["PUSH_DIR"] = Path(args.push_dir)
    if args.total_steps:
        globals()["EPOCH_STEPS"] = args.total_steps
    if args.then_push_dir:
        globals()["THEN_PUSH_DIR"] = Path(args.then_push_dir)
    if args.then_minutes:
        globals()["THEN_MINUTES"] = int(args.then_minutes)
    plan = [int(x) for x in args.minutes_plan.split(",") if x.strip()]
    end_plan = ([x.strip() for x in args.end_by.split(",")]
                if args.end_by else [])
    session = args.session
    start = args.start_step
    for turn in range(args.chain):
        out = Path(args.out) if args.out else Path(rf"E:\kron_out\session{session}")
        step = finish_one(session, start, args.hours, out,
                          push=not args.no_push,
                          next_minutes=plan[turn] if turn < len(plan) else None,
                          end_by=(end_plan[turn]
                                  if turn < len(end_plan) and end_plan[turn]
                                  else None))
        if step is None:
            break
        start = step
        session += 1
    say("done", session)


if __name__ == "__main__":
    main()
