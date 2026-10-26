"""Validate the digit pre-tokenizer before a full retokenisation.

The first attempt used ``Digits(individual_digits=True)`` after ByteLevel and
destroyed Cyrillic (2.72 bytes/token on Russian: every character fell back to
bytes).  This variant splits on digits *before* ByteLevel, which keeps UTF-8
sequences intact, and measures fertility separately for EN and RU.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from kronllm.bilingual import normalize
from kronllm.tokenizer import BPETokenizer

ROOT = Path("data/bilingual")


def load(lang_files, mb=10.0):
    texts, total = [], 0
    for path in lang_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            texts.append(normalize(doc["text"]))
            total += len(texts[-1].encode("utf-8"))
            if total >= mb * 1e6:
                return texts
    return texts


def train_v2(texts, path):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\d"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
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
    return nb / max(1, nt), nt


en = load([ROOT / "shards/wikipedia_en_science.jsonl"], 8)
ru = load([ROOT / "shards/wikipedia_ru_science.jsonl"], 8)
print(f"en docs {len(en)}, ru docs {len(ru)}")

path = ROOT / "tokenizer_probe" / "v2b.json"
path.parent.mkdir(exist_ok=True)
train_v2(en + ru, path)
tok = BPETokenizer(path, 8192, 1.0, 8191)

for name, texts in (("en", en[-200:]), ("ru", ru[-200:])):
    bpt, ntok = fertility(tok, texts)
    print(f"{name}: {bpt:.3f} bytes/token ({ntok} tokens)")
for s in ("1234 + 5678 = 6912", "Москва, 12 октября 2026 года.",
          "The result is 3.14159."):
    enc = tok.encode(s)
    print(repr(s), len(enc), [tok.decode([int(i)]) for i in enc[:12]],
          "roundtrip", tok.decode(enc) == s)
