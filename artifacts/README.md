# Artifacts

The shipped checkpoints, tracked directly in Git (each is a few megabytes).
They are the models referenced in the top-level [README](../README.md).

| file | model | precision | size | what |
|---|---|---|---|---|
| `q8g_int8.pt` | Q8g | int8 | 3.29 MB | **the shipped model** — 0.7542 bpb, 2,851,280 parameters |
| `q2_int8.pt` | Q2 | int8 | 2.34 MB | previous model, kept for comparison (0.8391 bpb, 2,079,992 parameters) |

The int8 files store symmetric per-channel scales; the forward pass is
unchanged fp32 after `load_any_checkpoint` dequantises them. Q8g int8 is
near-lossless (**+0.0003 bpb**, 3.55× smaller than fp32); Q2 int8 was
+0.0004 bpb.

The fp32 reference checkpoints stay outside Git, in the run directory:
`runs/Q8g_long/best.pt` (11.1 MB) and `runs/Q6_10k/best.pt` (9.7 MB).

## Loading

Through the CLI:

```bash
python -m kronllm.cli generate --run artifacts --checkpoint q8g_int8.pt \
    --prompt "Once upon a time" --temperature 1.0 --top-k 8
```

Through the Hugging Face wrapper:

```python
from kronllm.hf import KronForCausalLM
from kronllm.tokenizer import load_tokenizer

model = KronForCausalLM.from_kron_checkpoint("artifacts/q8g_int8.pt")
tok = load_tokenizer(model.config.kron_config)
```

`from_kron_checkpoint` accepts fp32, int8 and int4 checkpoints. The returned
model is in evaluation mode on CPU; call `.to("cuda")` to move it to a GPU.
