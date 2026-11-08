"""Make a decay-sized mix from the full one by keeping every k-th document.

The decay pass trains on ~30 % of the corpus, not on all of it: the full mix is
9.5 GB and its bin would be 5.4 GB, three times what 51,294 steps need, and the
"curated" tail would stop being curated.  The full mix is already the right
composition (EN+5 %, per-domain quotas), so the decay mix is the same thing
sampled down -- one document in ``--keep-1-in``, deterministically by position,
which preserves every domain's share instead of truncating alphabetically.

Side pools (chat, drills) are copied whole: they are sized for the pass, not for
the corpus, and the bin builder interleaves them itself.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIX = ROOT / "data" / "bilingual" / "mix"
OUT = ROOT / "data" / "bilingual" / "mix_decay"
SIDE = ("chat", "drills")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-1-in", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total_in = total_out = 0
    for shard in sorted(MIX.glob("*.jsonl.gz")):
        is_side = any(f"_{s}." in shard.name for s in SIDE)
        dst = out / shard.name
        if is_side:
            shutil.copyfile(shard, dst)
            print(f"  {shard.name:28s} copied whole "
                  f"({shard.stat().st_size/1e6:.1f} MB)")
            continue
        kept = 0
        with gzip.open(shard, "rt", encoding="utf-8") as src, \
                gzip.open(dst, "wt", encoding="utf-8", compresslevel=3) as fh:
            for i, line in enumerate(src):
                if i % args.keep_1_in:
                    continue
                fh.write(line)
                kept += 1
        total_in += 1
        total_out += kept
        print(f"  {shard.name:28s} {kept:,} docs "
              f"({dst.stat().st_size/1e6:.1f} MB)")
    print(f"[subset] 1 of {args.keep_1_in} from {total_in} shards, "
          f"{total_out:,} domain documents -> {out}")


if __name__ == "__main__":
    main()
