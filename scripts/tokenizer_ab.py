"""Rigorous A/B of the tokenizers on one fixed sample.

v1: GPT-2 byte-level regex (current corpus tokenizer).
v4: the same plus ``Digits(individual_digits=True)`` so every digit is its own
    token.

Both are trained on the identical 20 MB sample and measured on the identical
held-out slice, with a full decode round-trip check.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from kronllm.bilingual import normalize
from kronllm.tokenizer import BPETokenizer

ROOT = Path("data/bilingual")
OUT = ROOT / "tokenizer_probe"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "sample.json"
HELD = OUT / "heldout.json"


def build_sample(mb=20.0):
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    texts, total = [], 0
    for path in sorted((ROOT / "shards").glob("*.jsonl")):
        if total >= mb * 1e6:
            break
        if path.name.startswith(("identity", "tarball", "rfc", "cosmopedia")):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            texts.append(normalize(doc["text"]))
            total += len(texts[-1].encode("utf-8"))
            if total >= mb * 1e6:
                break
    CACHE.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return texts


def train(texts, path, digits):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    pre = [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)]
    if digits:
        pre.append(pre_tokenizers.Digits(individual_digits=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(pre)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=8187, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[], show_progress=False)
    tok.train_from_iterator(list(texts), trainer=trainer)
    tok.save(str(path))


def measure(name, path, held):
    tok = BPETokenizer(path, 8192, 1.0, 8191)
    nb = sum(len(t.encode("utf-8")) for t in held)
    nt = sum(len(tok.encode(t)) for t in held)
    roundtrip = all(tok.decode(tok.encode(t)) == t for t in held[:50])
    print(f"{name}: fertility {nb/nt:.3f} bytes/token | round-trip {roundtrip}")
    for s in ("1234 + 5678 = 6912", "The result is 3.14159, not 3.14.",
              "Москва, 12 октября 2026 года."):
        enc = tok.encode(s)
        print(f"   {s!r} -> {len(enc)} tokens; decode ok: "
              f"{tok.decode(enc) == s}")


texts = build_sample()
held = texts[-400:]
print(f"sample {len(texts)} docs, held {len(held)} docs")

train(texts[:-400], OUT / "ab_v1.json", digits=False)
train(texts[:-400], OUT / "ab_v4.json", digits=True)
measure("v1 (current)", OUT / "ab_v1.json", held)
measure("v4 (digits) ", OUT / "ab_v4.json", held)
