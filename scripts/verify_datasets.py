"""Verify both datasets are ready, remembering each success across polls.

The first version required one poll where both answered "ready".  With an API
that fails about half the calls (SSL EOF, 00:40-01:45) that is a coin flip per
poll squared: each dataset reported ready many times, never in the same poll,
and the flag was never written even though both had finished.  What matters is
that *each* has been seen ready, not that they were seen ready together.
"""
import subprocess
import sys
import time

SLUGS = {"corpus": "icold21/kron-corpus-2b",
         "anneal": "icold21/kron-q10a-anneal-bin"}


def status(slug: str, tries: int = 3) -> str:
    for i in range(tries):
        p = subprocess.run([sys.executable, "-m", "kaggle", "datasets",
                            "status", slug], capture_output=True, text=True,
                           timeout=180)
        out = (p.stdout or "") + (p.stderr or "")
        if "SSLError" not in out and "Max retries" not in out:
            return out.strip().splitlines()[-1] if out.strip() else "unknown"
        time.sleep(6 * (i + 1))
    return "unreachable"


ready: dict[str, bool] = {k: False for k in SLUGS}
for attempt in range(30):
    for key, slug in SLUGS.items():
        if ready[key]:
            continue
        line = status(slug)
        if "ready" in line.lower():
            ready[key] = True
            print(f"[{attempt}] {key}: ready")
        else:
            print(f"[{attempt}] {key}: {line[:80]}")
    if all(ready.values()):
        break
    time.sleep(20)

print("ready state:", ready)
if all(ready.values()):
    from datetime import datetime
    from pathlib import Path

    flag = Path("data/corpus_ready.flag")
    flag.write_text(
        f"epoch corpus and anneal bin verified ready at "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"corpus: {SLUGS['corpus']}\nanneal: {SLUGS['anneal']}\n"
        f"verified by scripts/verify_datasets.py (each seen ready at least once)\n",
        encoding="utf-8")
    print(f"[flag] {flag} written")
    sys.exit(0)

print("[stop] not both ready")
sys.exit(1)
