"""How far did the session actually get?

The Kaggle API happily hands back the *previous* version's output when a run
fails before writing anything, so "the download succeeded" does not mean "the
session progressed".  The chain compares these two numbers before it publishes
a checkpoint or pushes a resume.

    python scripts/session_progress.py --new       # last step in this download
    python scripts/session_progress.py --history   # last step seen before
"""
import argparse
import csv
from pathlib import Path

NEW = Path("data/kaggle_out/q10/log.csv")
HIST = Path("data/q10_history.csv")


def max_step(path: Path) -> int:
    if not path.exists():
        return -1
    best = -1
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                best = max(best, int(row["step"]))
            except (KeyError, ValueError):
                continue
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()
    print(max_step(NEW if args.new else HIST))


if __name__ == "__main__":
    main()
