"""Archive a downloaded checkpoint under the step it reached.

The chain clears ``data/kaggle_out/q10`` before every download (a failed run can
hand back the previous version's files, and stale weights are worse than no
weights), so the previous session's weights vanish.  Averaging needs the
*neighbouring* checkpoint -- the rule is ``best`` of the previous session plus
``best`` of this one, both taken inside the same constant-LR phase -- so the
best weights are copied aside before the directory is cleared.

Only ``best.pt`` is archived: it is weights-only (38 MB), while ``last.pt``
carries the optimizer state (115 MB) and is only needed for the same-session
``best + last`` soup, which is computed from the freshly downloaded pair.

Usage::

    python scripts/archive_ckpt.py --ckpt data/kaggle_out/q10/best.pt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/kaggle_out/q10/best.pt")
    ap.add_argument("--out-dir", default="data/kaggle_out/archive")
    args = ap.parse_args()

    src = Path(args.ckpt)
    if not src.exists():
        print(f"[skip] {src} not found")
        return

    import torch

    ck = torch.load(src, map_location="cpu", weights_only=False)
    step = int(ck.get("step", 0) or 0)
    dst = Path(args.out_dir) / f"{src.stem}_{step:06d}.pt"
    if dst.exists():
        print(f"[skip] {dst.name} already archived")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[archive] {src.name} step {step} -> {dst} "
          f"({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
