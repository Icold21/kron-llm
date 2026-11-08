"""From S10's checkpoint to a pushed S10b, in one command.

S10b continues S10, so it needs S10's weights as a dataset -- and Kaggle kernels
read datasets, not kernel outputs.  Doing that by hand is four steps in the right
order (download, publish, wait for the version to go live, push), and the third
one is the step that silently produces a stale mount.

Usage::

    python scripts/after_s10.py                    # normal path
    python scripts/after_s10.py --no-push          # publish only
    python scripts/after_s10.py --out E:/kron_out/s10
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path(r"E:\kron_s10_stage")          # C: has run out of space before
CKPT_SLUG = "icold21/kron-s10-ckpt"
KERNEL = "icold21/kron-s10-sft"
PUSH_DIR = Path("kaggle") / "s10b"


def kaggle(*args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "kaggle", *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def download(out: Path) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        r = kaggle("kernels", "output", KERNEL, "-p", str(out), timeout=1800)
        if r.returncode == 0:
            say(f"  downloaded on attempt {attempt}")
            return True
        say(f"  download attempt {attempt} failed: "
            f"{(r.stderr or r.stdout or '').strip().splitlines()[-1:]}")
        time.sleep(30)
    return False


def ckpt_step(path: Path) -> int:
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state.get("step") or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"E:\kron_out\s10")
    ap.add_argument("--min-step", type=int, default=3900,
                    help="refuse a checkpoint below this step (S10 runs 4,000)")
    ap.add_argument("--push-dir", default=str(PUSH_DIR))
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if not download(out):
        raise SystemExit("could not download the S10 output; nothing published")

    found = sorted(out.rglob("last.pt"), key=lambda q: q.stat().st_mtime)
    if not found:
        raise SystemExit(f"no last.pt under {out}")
    src = found[-1]
    step = ckpt_step(src)
    say(f"S10 checkpoint: {src} at step {step:,}")
    if step < args.min_step:
        raise SystemExit(f"step {step:,} < {args.min_step:,}: the run did not "
                         f"get far enough to continue from")

    STAGE.mkdir(parents=True, exist_ok=True)
    for stale in STAGE.glob("*.pt"):
        stale.unlink()
    for name in ("last.pt", "best.pt"):
        cand = src.with_name(name)
        if cand.exists():
            shutil.copyfile(cand, STAGE / name)
    (STAGE / "dataset-metadata.json").write_text(json.dumps({
        "title": "Kron S10 SFT checkpoint", "id": CKPT_SLUG,
        "licenses": [{"name": "other"}]}), encoding="utf-8")

    local_bytes = (STAGE / "last.pt").stat().st_size
    started = datetime.utcnow()
    r = kaggle("datasets", "create", "-p", str(STAGE), "--dir-mode", "zip")
    text = (r.stdout or "") + (r.stderr or "")
    if "already exists" in text.lower() or r.returncode != 0:
        say("  dataset exists (or create failed): uploading a new version")
        r = kaggle("datasets", "version", "-p", str(STAGE), "--dir-mode", "zip",
                   "-m", f"S10 at step {step:,}")
        text = (r.stdout or "") + (r.stderr or "")
    say(f"  upload: {text.strip().splitlines()[-1][:110] if text.strip() else '-'}")
    if r.returncode != 0:
        raise SystemExit("the checkpoint could not be published")

    deadline = time.time() + 20 * 60
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
            if size == local_bytes and stamp >= started - timedelta(minutes=2):
                say(f"  version live ({size:,} bytes, {stamp} UTC)")
                time.sleep(60)     # let the pointer settle before mounting it
                deadline = 0
                break
        if deadline == 0:
            break
        time.sleep(20)
    else:
        raise SystemExit("the new dataset version never appeared")

    if args.no_push:
        say("--no-push: published only")
        return

    r = kaggle("kernels", "push", "-p", str(ROOT / args.push_dir), timeout=900)
    text = (r.stdout or "") + (r.stderr or "")
    low = text.lower()
    say(f"  push: {text.strip().splitlines()[-1][:110] if text.strip() else '-'}")
    if r.returncode != 0 or "error" in low or "quota" in low:
        raise SystemExit("S10b was NOT pushed (see the message above)")


if __name__ == "__main__":
    main()
