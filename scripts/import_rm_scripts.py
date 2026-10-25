"""Import the Rick and Morty transcript datasets into the bilingual corpus.

Two Kaggle sources are merged:

* ``thebendu/rich-and-morty-transcripts`` — 71 episodes in broadcast order,
  with stage directions, but some rows are empty stubs;
* ``prarabdhasrivastava/rickmorty-transcripts`` — dialogue only, flat episode
  numbering 1..51, which lines up with the broadcast order (verified by
  n-gram overlap on the first episodes).

For each episode the longer of the two transcripts wins, so the slice contains
full dialogue for every episode both sources cover.

The text is a fan transcript: it is recorded in the manifest with provenance
and is *not* committed to the repository, only to the gitignored shard.

Usage::

    kaggle datasets download -d thebendu/rich-and-morty-transcripts -p /tmp/rm
    kaggle datasets download -d prarabdhasrivastava/rickmorty-transcripts -p /tmp/rm2
    python scripts/import_rm_scripts.py /tmp/rm/RnM_Transcripts*.xlsx \
        --csv /tmp/rm2/Rick-n-Morty.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.bilingual import append_jsonl, normalize, record_shard

ROOT = Path("data/bilingual")
SHARDS = ROOT / "shards"
MANIFEST = ROOT / "manifest.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", help="thebendu transcript workbook")
    ap.add_argument("--csv", default=None, help="prarabdhasrivastava dialogue CSV")
    ap.add_argument("--domain", default="humour")
    ap.add_argument("--min-chars", type=int, default=1000)
    args = ap.parse_args()

    import pandas as pd

    sheet = pd.read_excel(args.xlsx, sheet_name=0)
    extra: dict[int, str] = {}
    if args.csv:
        df = pd.read_csv(args.csv)
        for ep, group in df.groupby("episode no."):
            lines = []
            for _, row in group.iterrows():
                who = str(row.get("speaker") or "").strip()
                line = str(row.get("dialouge") or "").strip()
                if line:
                    lines.append(f"{who}: {line}" if who else line)
            extra[int(ep)] = "\n".join(lines)

    docs: list[dict] = []
    for i, (_, row) in enumerate(sheet.iterrows()):
        primary = normalize(str(row.get("Transcript") or ""))
        alt = normalize(extra.get(i + 1, ""))
        text = primary if len(primary) >= len(alt) else alt
        if len(text) < args.min_chars:
            continue
        docs.append({
            "title": f"Rick and Morty {row.get('Season', '')} "
                     f"{row.get('Episode', '')}".strip(),
            "text": text, "language": "en", "domain": args.domain,
            "source": "Kaggle thebendu + prarabdhasrivastava",
            "licence": "fan transcript; see the dataset pages",
            "url": "https://www.kaggle.com/datasets/thebendu/rich-and-morty-transcripts",
            "note": "easter egg: series dialogue",
        })
    out = SHARDS / f"rickandmorty_{args.domain}.jsonl"
    if out.exists():
        out.unlink()
    n = append_jsonl(out, docs)
    shard = record_shard(out, MANIFEST,
                         source="Kaggle thebendu + prarabdhasrivastava",
                         licence="fan transcript; see the dataset pages",
                         url="https://www.kaggle.com/datasets/thebendu/rich-and-morty-transcripts",
                         language="en", domain=args.domain, docs=n,
                         note="full dialogue, two sources merged, easter egg")
    print(f"[done] {n} episodes, {shard.bytes/1e6:.2f} MB -> {out.name}")


if __name__ == "__main__":
    main()
