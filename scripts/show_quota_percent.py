"""Quotas and the realised shares, in percent.

The per-domain token columns are apportioned from bytes with a per-language
bytes/token and then calibrated on the number the encoder measured, so the
columns add up to the real bin instead of to a 2 %-off estimate.
"""
import json
from pathlib import Path

from kronllm.bilingual import QUOTAS_EN, QUOTAS_RU

ROOT = Path(__file__).resolve().parent.parent
report = json.loads((ROOT / "data" / "bilingual" / "corpus_report.json")
                    .read_text(encoding="utf-8"))
meta = json.loads(Path(r"E:\kron_bins\epoch_v4\tokenizer_meta.json")
                  .read_text(encoding="utf-8"))
mixed = report["mixed"]
BPT = {"en": 2.917, "ru": 4.964}
TOKENS_PER_STEP = 16_384

en_b = sum(mixed["en"].values())
ru_b = sum(mixed["ru"].values())
tot_b = en_b + ru_b
raw = {"en": en_b / BPT["en"], "ru": ru_b / BPT["ru"]}
scale = meta["train_tokens"] / sum(raw.values())
tok = {k: v * scale for k, v in raw.items()}
tot_t = meta["train_tokens"]

print(f"{'домен':14s}{'EN квота':>10s}{'EN факт':>9s}{'RU квота':>10s}"
      f"{'RU факт':>9s}{'% текста':>10s}{'% токенов':>11s}")
domains = sorted(set(mixed["en"]) | set(mixed["ru"]) | set(QUOTAS_EN) | set(QUOTAS_RU),
                 key=lambda d: -(mixed["en"].get(d, 0) + mixed["ru"].get(d, 0)))
for d in domains:
    qe = QUOTAS_EN.get(d, 0) * 100
    qr = QUOTAS_RU.get(d, 0) * 100
    fe = 100 * mixed["en"].get(d, 0) / en_b
    fr = 100 * mixed["ru"].get(d, 0) / ru_b
    share_b = 100 * (mixed["en"].get(d, 0) + mixed["ru"].get(d, 0)) / tot_b
    share_t = 100 * (mixed["en"].get(d, 0) / BPT["en"]
                     + mixed["ru"].get(d, 0) / BPT["ru"]) * scale / tot_t
    print(f"{d:14s}{qe:9.1f}%{fe:8.1f}%{qr:9.1f}%{fr:8.1f}%"
          f"{share_b:9.1f}%{share_t:10.1f}%")

print(f"\nтекст:  EN {en_b/1e9:.2f} ГБ ({100*en_b/tot_b:.1f}%) / "
      f"RU {ru_b/1e9:.2f} ГБ ({100*ru_b/tot_b:.1f}%)")
print(f"токены: EN {tok['en']/1e9:.2f} млрд ({100*tok['en']/tot_t:.1f}%) / "
      f"RU {tok['ru']/1e9:.2f} млрд ({100*tok['ru']/tot_t:.1f}%)")
print(f"всего:  {tot_t/1e9:.2f} млрд токенов, "
      f"{tot_t/TOKENS_PER_STEP:,.0f} шагов, {tot_b/tot_t:.2f} байта/токен")
prose_e = (mixed["en"].get("encyclopedic", 0)
           + mixed["en"].get("literature", 0)) / en_b
prose_r = (mixed["ru"].get("encyclopedic", 0)
           + mixed["ru"].get("literature", 0)) / ru_b
print(f"проза внутри языка: EN {100*prose_e:.1f}% / RU {100*prose_r:.1f}%")
