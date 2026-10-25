"""10M @ d=448 (the 6 GB variant): step time and the first losses."""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from kronllm.config import experiment_config
from kronllm.model import KronTransformer

BATCH, STEPS, SEQ = 8, 60, 256
cfg = experiment_config("Q8g")
cfg["model"].update({"d_model": 448, "n_layers": 27, "n_heads": 7,
                     "embed_rank": 224})
cfg["model"]["kron"].update({"n1": 16, "n2": 28, "K": 12})
cfg["data"].update({"vocab_size": 8192, "seq_len": SEQ})

model = KronTransformer(cfg).cuda()
print("params", f"{sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
tokens = np.memmap("data/train.bpe4096.bin", dtype=np.uint16, mode="r")
n_seq = (len(tokens) - 1) // SEQ
rng = np.random.default_rng(0)
torch.cuda.reset_peak_memory_stats()

for step in range(1, STEPS + 1):
    starts = rng.integers(0, n_seq, size=BATCH) * SEQ
    x = torch.from_numpy(np.stack([tokens[s:s + SEQ] for s in starts]).astype(np.int64)).cuda()
    y = torch.from_numpy(np.stack([tokens[s + 1:s + 1 + SEQ] for s in starts]).astype(np.int64)).cuda()
    out = model(x, targets=y)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 10 == 0:
        print(f"step {step} loss {float(out['loss']):.4f}", flush=True)
print("peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
