"""Fetch SFT sources one subprocess at a time, with a per-source time limit.

``datasets`` streaming blocks forever when a Hub CDN read stalls (MetaMathQA
did: a ``ReadTimeoutError`` from ``us.aws.cdn.hf.co`` that the library retries
without surfacing).  One stuck source must not hold up twenty others, so every
source is fetched by its own ``build_sft.py fetch --sources <name>`` process
under a timeout, and a failure is recorded instead of retried forever.

Usage::

    python scripts/fetch_sft_sources.py --raw E:/kron_sft/raw --timeout 900
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sft_registry import SOURCES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="E:/kron_sft/raw")
    ap.add_argument("--target-tokens", type=int, default=500_000_000)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    names = args.sources or [s.name for s in SOURCES if not s.local]
    env = dict(os.environ)
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    failed = {}
    for name in names:
        done = raw / f"{name}.done.json"
        fail = raw / f"{name}.failed.json"
        if done.exists() and not args.force:
            print(f"[skip] {name}: already fetched", flush=True)
            continue
        ok = False
        for attempt in range(1, args.attempts + 1):
            cmd = [sys.executable, str(ROOT / "scripts" / "build_sft.py"),
                   "fetch", "--raw", str(raw),
                   "--target-tokens", str(args.target_tokens),
                   "--sources", name]
            if args.force:
                cmd.append("--force")
            t0 = time.time()
            print(f"[run] {name} attempt {attempt}/{args.attempts}", flush=True)
            try:
                proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                      timeout=args.timeout,
                                      capture_output=True, text=True,
                                      encoding="utf-8", errors="replace")
                tail = [line for line in (proc.stdout or "").splitlines()
                        if line.strip()][-4:]
                for line in tail:
                    print(f"    {line}", flush=True)
                if proc.returncode != 0 and proc.stderr:
                    print(f"    [stderr] {proc.stderr.strip().splitlines()[-1][:160]}",
                          flush=True)
                ok = done.exists()
            except subprocess.TimeoutExpired:
                print(f"    [timeout] {name} exceeded {args.timeout}s, killed",
                      flush=True)
            if ok:
                break
            time.sleep(5)
        if not ok:
            failed[name] = {"attempts": args.attempts,
                            "seconds": round(time.time() - t0, 1)}
            fail.write_text(json.dumps(failed[name], indent=1), encoding="utf-8")
            print(f"[fail] {name}", flush=True)
        else:
            fail.unlink(missing_ok=True)

    print(f"\n[fetch] failures: {len(failed)}", flush=True)
    for name, info in failed.items():
        print(f"  - {name} ({info['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
