import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
meta = {
    "title": "KronCorpus 2B bins",
    "id": "icold21/kron-corpus-2b",
    "licenses": [{"name": "other"}],
}
path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"[meta] {path}")
