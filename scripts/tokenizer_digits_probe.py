"""The tokenizers library has a Digits pre-tokenizer: keep GPT-2 chunks,
split only numeric pieces into single digits."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from kronllm.bilingual import normalize
from kronllm.tokenizer import BPETokenizer

ROOT = Path("data/bilingual")
OUT = ROOT / "tokenizer_probe"
OUT.mkdir(parents=True, exist_ok=True)


def sample(mb=12.0):
    texts, total = [], 0
    for path in sorted((ROOT / "shards").glob("*.jsonl")):
        if total >= mb * 1e6:
            break
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc.get("domain") not in {"math", "science", "encyclopedic",
                                         "literature", "ml", "logic"}:
                continue
            texts.append(normalize(doc["text"]))
            total += len(texts[-1])
            if total >= mb * 1e6:
                break
    return texts


def train(texts, path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        pre_tokenizers.Digits(individual_digits=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=8187, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[], show_progress=False)
    tok.train_from_iterator(list(texts), trainer=trainer)
    tok.save(str(path))


texts = sample()
path = OUT / "v4_digits.json"
train(texts, path)
tok = BPETokenizer(path, 8192, 1.0, 8191)
nb = sum(len(t.encode()) for t in texts[-500:])
nt = sum(len(tok.encode(t)) for t in texts[-500:])
print(f"v4 fertility {nb/nt:.3f} bytes/token")
for s in ("1234 + 5678 = 6912",
          "The result is 3.14159, not 3.14.",
          "x^2 + 2x - 15 = 0",
          "Москва, 12 октября 2026 года.",
          "2024-10-26T18:40:35+03:00"):
    enc = tok.encode(s)
    print(f"  {s!r} -> {len(enc)} {[tok.decode([int(i)]) for i in enc[:20]]}")
