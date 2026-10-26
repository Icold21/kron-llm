"""Measure the Llama-style single-digit pre-tokenizer against the current one."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from kronllm.bilingual import normalize
from kronllm.tokenizer import BPETokenizer, train_bpe_merges

ROOT = Path("data/bilingual")
OUT = ROOT / "tokenizer_probe"
OUT.mkdir(parents=True, exist_ok=True)

PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
           r"|[^\r\n\p{L}\p{N}]?\p{L}+"
           r"|\p{N}"
           r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
           r"|\s*[\r\n]+"
           r"|\s+(?!\S)|\s+")


def sample(mb=12.0):
    rng = random.Random(11)
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


def train_llama(texts, path):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=8187, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[], show_progress=False)
    tok.train_from_iterator(list(texts), trainer=trainer)
    tok.save(str(path))


def fertility(tok, texts):
    nb = sum(len(t.encode()) for t in texts)
    nt = sum(len(tok.encode(t)) for t in texts)
    return nb / max(1, nt)


texts = sample()
print(f"sample {len(texts)} docs", flush=True)

path = OUT / "v3_llama.json"
train_llama(texts, path)
tok = BPETokenizer(path, 8192, 1.0, 8191)
print(f"v3 fertility {fertility(tok, texts[-500:]):.3f} bytes/token")
for s in ("1234 + 5678 = 6912",
          "The result is 3.14159, not 3.14.",
          "x^2 + 2x - 15 = 0",
          "Москва, 12 октября 2026 года."):
    enc = tok.encode(s)
    pieces = [tok.decode([int(i)]) for i in enc[:20]]
    print(f"  {s!r} -> {len(enc)} {pieces}")
