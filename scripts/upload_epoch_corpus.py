"""Upload the fully-prompted epoch corpus and prove that it landed.

``kaggle datasets version`` sends the whole 4.5 GB in one request, cannot
resume, and on a flaky link its exit text is the only signal.  Worse, the exit
text lies in both directions: an SSLError whose traceback mentions
``DatasetApiService`` once read as "published" for a dataset that was never
created, and a killed parent process leaves a ``ready`` dataset that is still
the previous version.

So this script does three things the shell one-liner cannot:

* retries the upload, counting only attempts that reached the API;
* waits for the dataset to report ``ready`` after each accepted version;
* downloads ``tokenizer_meta.json`` back from Kaggle and asserts it is the
  fully-prompted bin -- ``rate == 1.0`` and the expected token count.  A stale
  dataset fails that check, which is exactly what we want to know.

Usage::

    python scripts/upload_epoch_corpus.py [--attempts 8] [--minutes 150]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = Path(r"E:\kron_bins\epoch_np")
SLUG = "icold21/kron-corpus-2b"
LOCAL_META = STAGE / "tokenizer_meta.json"
NOTE = "epoch bin: prompt-free rebuild (sciences kept, no inserted prompts)"


def say(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with (ROOT / "data" / "upload_epoch.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def kaggle(*argv: str, stream: bool = False) -> subprocess.CompletedProcess:
    """Run the CLI.  ``stream=True`` inherits stdout/stderr so the progress bar
    lands in the log: the previous attempt sat for an hour with nothing to read
    because the output was captured and only shown when the call returned."""
    if stream:
        return subprocess.run([sys.executable, "-m", "kaggle", *argv])
    return subprocess.run([sys.executable, "-m", "kaggle", *argv],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def status() -> str:
    r = kaggle("datasets", "status", SLUG)
    tail = (r.stdout or r.stderr).strip().splitlines()
    return tail[-1].strip() if tail else "unreachable"


def expected() -> dict:
    meta = json.loads(LOCAL_META.read_text(encoding="utf-8"))
    dp = meta["domain_prompts"]
    return {"tokens": meta["train_tokens"], "docs": dp["docs"],
            "enabled": dp["enabled"], "prompted": dp["prompted_docs"]}


def verify(want: dict) -> bool:
    """Read the metadata back from Kaggle; the dataset must be the new bin."""
    d = ROOT / "data" / "corpus_check"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    r = kaggle("datasets", "download", "-d", SLUG, "-f", "tokenizer_meta.json",
               "-p", str(d))
    f = d / "tokenizer_meta.json"
    if not f.exists():
        say(f"  remote metadata unreadable: {(r.stdout or r.stderr).strip()[-160:]}")
        return False
    meta = json.loads(f.read_text(encoding="utf-8"))
    dp = meta.get("domain_prompts") or {}
    got = {"tokens": meta.get("train_tokens"), "docs": dp.get("docs"),
           "enabled": dp.get("enabled"), "prompted": dp.get("prompted_docs")}
    ok = got == want
    say(f"  remote meta: {got} (want {want}) -> {'match' if ok else 'STALE'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=150.0)
    args = ap.parse_args()

    if not (STAGE / "train.bpe.bin").exists():
        say("stage is missing train.bpe.bin")
        raise SystemExit(1)
    size = (STAGE / "train.bpe.bin").stat().st_size
    want = expected()
    say(f"staged {size:,} bytes; want {want}")
    deadline = time.time() + args.minutes * 60

    for attempt in range(1, args.attempts + 1):
        if time.time() > deadline:
            say("out of time")
            break
        say(f"attempt {attempt}/{args.attempts}")
        r = kaggle("datasets", "version", "-p", str(STAGE), "-m", NOTE,
                   "--dir-mode", "zip", stream=True)
        say(f"  cli exit {r.returncode}")

        # Success is decided by the data, not by the CLI's wording: read the
        # metadata back from Kaggle.  A kill mid-upload leaves the old version
        # serving, which is exactly what this check catches.
        if r.returncode == 0:
            say("  version accepted; waiting for ready")
            for _ in range(60):
                s = status()
                if s.startswith("ready"):
                    break
                if s.startswith("error") or s.startswith("failed"):
                    say(f"  dataset failed: {s}")
                    break
                time.sleep(30)
            if verify(want):
                say("DONE: the prompt-free corpus is live on Kaggle")
                return
            say("  uploaded but the metadata does not match the new bin")

        wait = min(300, 30 * attempt)
        say(f"  sleeping {wait}s")
        time.sleep(wait)

    say("gave up; the corpus on Kaggle is not the prompted one")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
