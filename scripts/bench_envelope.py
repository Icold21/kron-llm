"""Find the (batch, seq) envelope of the 10M config on this GPU."""
import sys

import torch

sys.path.insert(0, ".")
from kronllm.config import experiment_config
from kronllm.model import KronTransformer

device = "cuda"
for batch, seq in [(12, 512), (8, 512), (6, 512), (24, 256), (16, 256), (24, 192)]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = experiment_config("Q8g")
    cfg["model"].update({"d_model": 512, "n_layers": 25, "n_heads": 8,
                         "embed_rank": 256})
    cfg["model"]["kron"].update({"n1": 16, "n2": 32, "K": 10})
    cfg["data"].update({"vocab_size": 8192, "seq_len": seq})
    model = KronTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, 8192, (batch, seq), device=device)
    y = torch.randint(0, 8192, (batch, seq), device=device)
    try:
        out = model(x, targets=y)
        out["loss"].backward()
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"batch {batch:2d} seq {seq} : OK  peak {peak:.2f} GB", flush=True)
    except torch.OutOfMemoryError:
        print(f"batch {batch:2d} seq {seq} : OOM", flush=True)
    del model, opt, x, y
    torch.cuda.empty_cache()
