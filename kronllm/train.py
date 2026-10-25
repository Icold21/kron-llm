"""Training / evaluation pipeline for KronLLM."""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import save_config
from .data import ByteDataset, open_dataset
from .tokenizer import bits_per_byte, load_tokenizer
from .diagnostics import (aggregate_eval_stats, comparison_report, grad_report,
                          routed_report, sigma_theta_report, spectral_report,
                          theta_grad_norm)


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def cosine_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    if total <= warmup:
        return 1.0
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))


class CSVLogger:
    """Append-only CSV logger that tolerates a growing column set.

    Diagnostics rows add fields that training rows do not have, so the header
    is rewritten (from an in-memory copy) whenever the schema grows.  Without
    this, ``pandas.read_csv`` fails and the dashboard shows an empty table.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: List[str] = []
        self._rows: List[Dict[str, Any]] = []
        # A run can be resumed (or restarted in the same directory).  Load the
        # existing rows first: _rewrite() drops anything not in memory, so
        # without this a restart silently truncated log.csv to the new rows.
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                self._fields = list(reader.fieldnames or [])
                self._rows = [dict(r) for r in reader]

    def log(self, row: Mapping[str, Any]) -> None:
        new_keys = [k for k in row if k not in self._fields]
        self._rows.append(dict(row))
        if new_keys or not self.path.exists():
            self._fields.extend(new_keys)
            self._rewrite()
            return
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=self._fields,
                                extrasaction="ignore")
            wr.writerow({k: row.get(k, "") for k in self._fields})

    def _rewrite(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=self._fields,
                                extrasaction="ignore")
            wr.writeheader()
            for r in self._rows:
                wr.writerow({k: r.get(k, "") for k in self._fields})

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------
def build_param_groups(model: nn.Module, tc: Mapping[str, Any]) -> List[dict]:
    decay, no_decay, s_params, theta_params, scale_params = [], [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".theta"):
            theta_params.append(p)
        elif name.endswith(".tau") or name.endswith(".gate"):
            scale_params.append(p)
        elif name.endswith(".s"):
            s_params.append(p)
        elif ("norm" in name or "embed" in name or "lm_head" in name
              or "route" in name or "bank" in name or "pass_bias" in name):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "lr": float(tc["lr"]),
         "weight_decay": float(tc["weight_decay"]), "name": "decay"},
        {"params": no_decay, "lr": float(tc["lr"]),
         "weight_decay": 0.0, "name": "no_decay"},
        {"params": s_params, "lr": float(tc["lr_s"]),
         "weight_decay": 0.0, "name": "anchor_s"},
        {"params": theta_params, "lr": float(tc["lr_theta"]),
         "weight_decay": 0.0, "name": "theta"},
        {"params": scale_params, "lr": float(tc["lr"]),
         "weight_decay": 0.0, "name": "scales"},
    ]
    return [g for g in groups if g["params"]]


def build_optimizer(model: nn.Module, tc: Mapping[str, Any]):
    groups = build_param_groups(model, tc)
    betas = tuple(tc["betas"])
    if str(tc["optimizer"]).lower() == "adamw":
        return torch.optim.AdamW(groups, betas=betas, eps=1e-8)
    raise ValueError(f"unsupported optimizer {tc['optimizer']}")


def init_base_lrs(opt: torch.optim.Optimizer) -> None:
    for g in opt.param_groups:
        g["base_lr"] = float(g["lr"])


def set_lr(opt: torch.optim.Optimizer, factor: float) -> float:
    lr0 = 0.0
    for g in opt.param_groups:
        g["lr"] = float(g["base_lr"]) * factor
        lr0 = g["lr"]
    return lr0


def named_lrs(opt: torch.optim.Optimizer) -> Dict[str, float]:
    return {g.get("name", "group"): g["lr"] for g in opt.param_groups}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, ds: ByteDataset, batch_size: int,
             device: torch.device, max_batches: Optional[int] = None,
             amp: str = "fp16") -> Dict[str, float]:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    use_amp = amp == "fp16" and device.type == "cuda"
    for i, (x, y) in enumerate(ds.iter_blocks(batch_size, device)):
        if max_batches is not None and i >= max_batches:
            break
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x, y)
        total_loss += float(out["loss"].detach()) * x.numel()
        total_tokens += x.numel()
    model.train()
    if total_tokens == 0:
        return {"val_loss": float("nan"), "val_ppl": float("nan")}
    loss = total_loss / total_tokens
    result = {"val_loss": loss, "val_ppl": math.exp(min(loss, 50.0))}
    result.update(routed_report(model))
    return result


@torch.no_grad()
def diagnostics(model: nn.Module, ds: ByteDataset, batch_size: int,
                device: torch.device, spectral: bool = True,
                amp: str = "fp16") -> Dict[str, Any]:
    model.eval()
    x, y = next(iter(ds.iter_blocks(batch_size, device)))
    use_amp = amp == "fp16" and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        out = model(x, y, collect=True)
    result: Dict[str, Any] = {"batch_loss": float(out["loss"].detach())}
    result.update(aggregate_eval_stats(out["stats"]))
    result.update({k: v for k, v in sigma_theta_report(model).items()
                   if not isinstance(v, dict)})
    result.update(routed_report(model))
    result["theta_grad_norm"] = theta_grad_norm(model)
    if spectral:
        spec = spectral_report(model, out["trace"])
        result["spectral"] = spec["summary"]
        result.update({f"spec_{k}": v for k, v in spec["summary"].items()
                       if isinstance(v, (int, float))})
    model.last_trace = None
    model.train()
    return result


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def _init_process_group(rank: int, world_size: int) -> None:
    """Initialise DDP, with a Windows fallback for the rendezvous store.

    Some Windows wheels are built without libuv, so the default TCP store
    aborts with "use_libuv was requested but PyTorch was build without libuv
    support".  Retrying with an explicitly constructed store fixes it; on Linux
    (Kaggle) the default path is used unchanged.
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        dist.init_process_group(backend=backend, rank=rank,
                                world_size=world_size)
        return
    except Exception:                                   # noqa: BLE001
        if dist.is_initialized():
            dist.destroy_process_group()
    addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = int(os.environ.get("MASTER_PORT", "29500"))
    store = dist.TCPStore(addr, port, world_size, rank == 0,
                          timeout=datetime.timedelta(minutes=30),
                          use_libuv=False)
    dist.init_process_group(backend=backend, store=store, rank=rank,
                            world_size=world_size)


def train(cfg: Mapping[str, Any], run_dir: str | Path,
          resume: Optional[str] = None) -> Dict[str, Any]:
    tc = cfg["training"]
    dc = cfg["data"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed, deterministic=bool(tc.get("deterministic", True)))

    # -- distributed ----------------------------------------------------
    # `torchrun --nproc_per_node=N` sets WORLD_SIZE/RANK/LOCAL_RANK.  Without
    # it the run is the single-process path exactly as before.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0
    if ddp:
        if not dist.is_initialized():
            _init_process_group(rank, world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    run_dir = Path(run_dir)
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config.yaml")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if ddp
                              else tc.get("device", "cuda"))
    else:
        device = torch.device("cpu")
    data_dir = Path(dc.get("data_dir", "data"))
    train_ds = open_dataset(cfg, "train")
    val_ds = open_dataset(cfg, "val")
    tok = load_tokenizer(cfg)
    bpt = tok.bytes_per_token

    # -- model ---------------------------------------------------------
    from .model import KronTransformer
    model = KronTransformer(cfg)
    init_from = tc.get("init_from")
    if init_from:
        source = Path(init_from)
        if not source.is_absolute():
            source = Path.cwd() / source
        if not source.exists():
            raise FileNotFoundError(f"training.init_from not found: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state)
        print(f"[init] weights <- {source} (step {payload.get('step', 'n/a') if isinstance(payload, dict) else 'n/a'})")
    model = model.to(device)
    raw_model = model                       # unwrapped: eval, diagnostics, save
    if tc.get("compile") and hasattr(torch, "compile"):
        model = torch.compile(model)
    if ddp:
        # theta is inert when sigma_mode != "learned", so the flag is needed
        # there; the default configuration has no unused parameters.
        kc = cfg["model"]["kron"]
        unused = str(kc.get("sigma_mode", "learned")) != "learned"
        model = DDP(model,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                    find_unused_parameters=unused)

    opt = build_optimizer(model, tc)
    init_base_lrs(opt)
    use_amp = str(tc["amp"]) == "fp16" and device.type == "cuda" and tc.get("amp") != "none"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    bs = int(tc["batch_size"])
    accum = int(tc["grad_accum"])
    steps_per_epoch = max(1, len(train_ds) // (bs * accum))
    total_steps = int(tc.get("max_steps") or steps_per_epoch * int(tc["epochs"]))
    warmup = int(tc["warmup_steps"])
    min_ratio = float(tc["min_lr"]) / float(tc["lr"])
    eval_every = int(tc["eval_every"])
    log_every = int(tc["log_every"])
    save_every = int(tc.get("save_every", 0) or 0)
    spectral_every = int(cfg["metrics"].get("spectral_every", 2000))
    eval_batches = int(tc.get("eval_batches", 64))
    patience = int(tc.get("early_stop_patience", 4))
    random_recurse = (bool(tc.get("recurse_random"))
                      and getattr(model, "max_recurse", 1) > 1)

    logger = CSVLogger(run_dir / "log.csv") if is_main else None
    tb = None
    if cfg.get("logging", {}).get("tensorboard"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(run_dir / "tb")
        except Exception as exc:      # broken tensorflow/tensorboard env
            print(f"[warn] tensorboard disabled: {exc}")
    wb = None
    if cfg.get("logging", {}).get("wandb"):
        try:
            import wandb
            wb = wandb.init(project=cfg["logging"].get("wandb_project", "kronllm"),
                            name=f"{cfg.get('experiment', 'run')}",
                            config=json.loads(json.dumps(cfg, default=str)),
                            dir=str(run_dir))
        except Exception as exc:
            print(f"[warn] wandb disabled: {exc}")

    eff_bs = int(tc["batch_size"]) * int(tc["grad_accum"])
    comp = comparison_report(raw_model, int(dc["seq_len"]), batch_size=eff_bs)
    if is_main:
        print(f"[model] {raw_model.n_params:,} params "
              f"({raw_model.n_params_matrix:,} in matrix layers), "
              f"{comp['flops_per_token']/1e6:.1f} MFLOP/token "
              f"@ batch {eff_bs}"
              + (f" | ddp world_size={world_size}" if ddp else ""))

    gen = torch.Generator(device="cpu").manual_seed(seed + 1000 * rank)
    start_step = 0
    best_ppl = float("inf")
    bad_evals = 0
    history: List[Dict[str, Any]] = []

    if resume:
        ck = torch.load(resume, map_location="cpu")
        raw_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = int(ck.get("step", 0))
        best_ppl = float(ck.get("val_ppl", float("inf")))
        if is_main:
            print(f"[resume] step {start_step}, best ppl {best_ppl:.3f}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    t_start = time.time()
    step = start_step
    micro = 0
    running = 0.0
    running_n = 0
    last_log = time.time()
    stop = False

    while step < total_steps and not stop:
        for _ in range(accum):
            if random_recurse:
                raw_model.set_recurse(random.randint(1, raw_model.max_recurse))
            x, y = train_ds.get_batch(bs, device, gen)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x, y)
                loss = out["loss"] / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(loss).backward()
            running += float(loss.detach()) * accum
            running_n += 1
            micro += 1

        factor = cosine_factor(step, total_steps, warmup, min_ratio)
        set_lr(opt, factor)
        scaler.unscale_(opt)
        grads = grad_report(model)
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(tc["grad_clip"])))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if save_every and step % save_every == 0 and is_main:
            # Crash / quota resilience: a resumed run continues from here.
            torch.save({"model": raw_model.state_dict(), "cfg": dict(cfg),
                        "optimizer": opt.state_dict(), "step": step,
                        "val_ppl": best_ppl}, run_dir / "last.pt")

        if step % log_every == 0 or step == 1:
            dt = time.time() - last_log
            last_log = time.time()
            train_loss = running / max(1, running_n)
            named = named_lrs(opt)
            row = {
                "step": step,
                "split": "train",
                "loss": train_loss,
                "ppl": math.exp(min(train_loss, 50.0)),
                "bpb": bits_per_byte(train_loss, bpt),
                "grad_norm": gnorm,
                "lr": named.get("decay", float(tc["lr"])),
                "lr_s": named.get("anchor_s", float(tc["lr_s"])),
                "lr_theta": named.get("theta", float(tc["lr_theta"])),
                "sec_per_step": dt / max(1, log_every),
            }
            row.update({k: v for k, v in grads.items()})
            if is_main:
                logger.log(row)
                if tb:
                    for k, v in row.items():
                        if isinstance(v, (int, float)):
                            tb.add_scalar(f"train/{k}", v, step)
                if wb:
                    wb.log({f"train/{k}": v for k, v in row.items()
                            if isinstance(v, (int, float))}, step=step)
                print(f"step {step:5d} | loss {train_loss:.4f} | "
                      f"ppl {math.exp(min(train_loss,50)):.2f} | "
                      f"gnorm {gnorm:.3f} | {dt/max(1,log_every)*1000:.0f} ms/step")
            running, running_n = 0.0, 0

        if step % eval_every == 0 or step == total_steps:
            if random_recurse:
                raw_model.set_recurse(raw_model.max_recurse)
            # Every rank evaluates the same fixed validation batches, so the
            # numbers (and therefore early-stop decisions) agree without a
            # broadcast.  Only rank 0 logs and writes checkpoints.
            ev = evaluate(raw_model, val_ds, bs, device, eval_batches, str(tc["amp"]))
            ev["val_bpb"] = bits_per_byte(ev["val_loss"], bpt)
            if is_main:
                row = {"step": step, "split": "val", **ev}
                logger.log(row)
                if tb:
                    tb.add_scalar("val/loss", ev["val_loss"], step)
                    tb.add_scalar("val/ppl", ev["val_ppl"], step)
                    tb.add_scalar("val/bpb", ev["val_bpb"], step)
                if wb:
                    wb.log({"val/loss": ev["val_loss"], "val/ppl": ev["val_ppl"],
                            "val/bpb": ev["val_bpb"]}, step=step)
                print(f"  [val] step {step:5d} | loss {ev['val_loss']:.4f} | "
                      f"ppl {ev['val_ppl']:.2f} | bpb {ev['val_bpb']:.3f}")

            if ev["val_ppl"] < best_ppl:
                best_ppl = ev["val_ppl"]
                bad_evals = 0
                if is_main:
                    torch.save({"model": raw_model.state_dict(), "cfg": dict(cfg),
                                "step": step, "val_ppl": best_ppl},
                               run_dir / "best.pt")
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    if is_main:
                        print(f"[early-stop] no improvement for {bad_evals} evals")
                    stop = True

            if spectral_every and (step % spectral_every == 0 or step == total_steps) and is_main:
                dg = diagnostics(raw_model, val_ds, bs, device, spectral=True,
                                 amp=str(tc["amp"]))
                logger.log({"step": step, "split": "diag", **{
                    k: v for k, v in dg.items() if isinstance(v, (int, float))}})
                if tb:
                    for k, v in dg.items():
                        if isinstance(v, (int, float)):
                            tb.add_scalar(f"diag/{k}", v, step)
                if wb:
                    wb.log({f"diag/{k}": v for k, v in dg.items()
                            if isinstance(v, (int, float))}, step=step)
                history.append({"step": step, **dg})

    if random_recurse:
        raw_model.set_recurse(raw_model.max_recurse)
    if is_main:
        torch.save({"model": raw_model.state_dict(), "cfg": dict(cfg),
                    "optimizer": opt.state_dict(),
                    "step": step, "val_ppl": best_ppl}, run_dir / "last.pt")

    elapsed = time.time() - t_start
    peak_vram = 0.0
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e9

    final = evaluate(raw_model, val_ds, bs, device, eval_batches, str(tc["amp"]))
    dg = diagnostics(raw_model, val_ds, bs, device, spectral=True,
                     amp=str(tc["amp"])) if is_main else {}

    results = {
        "experiment": cfg.get("experiment", "run"),
        "desc": cfg.get("experiment_desc", ""),
        **comp,
        **final,
        "val_bpb": bits_per_byte(final["val_loss"], bpt),
        "bytes_per_token": bpt,
        "vocab_size": tok.vocab_size,
        "tokenizer": tok.name,
        "best_val_ppl": best_ppl,
        "steps": step,
        "time_total_sec": elapsed,
        "time_per_epoch_sec": elapsed / max(1.0, step / max(1, steps_per_epoch)),
        "sec_per_step": elapsed / max(1, step),
        "peak_vram_gb": peak_vram,
        "sigma_theta_mean": dg.get("sigma_theta_mean"),
        "sigma_theta_var": dg.get("sigma_theta_var"),
        "attn_entropy_mean": dg.get("attn_entropy_mean"),
        "attn_entropy_min": dg.get("attn_entropy_min"),
        "q_sigma_min": dg.get("q_sigma_min"),
        "k_sigma_min": dg.get("k_sigma_min"),
        "q_drift_mean": dg.get("q_drift_mean"),
        "h_norm_ratio_mean": dg.get("h_norm_ratio_mean"),
        "h_norm_ratio_min": dg.get("h_norm_ratio_min"),
        "h_norm_ratio_max": dg.get("h_norm_ratio_max"),
        "eff_rank_mean": dg.get("spec_eff_rank_mean"),
        "eff_rank_W": dg.get("spec_eff_rank_W"),
        "eff_rank_D": dg.get("spec_eff_rank_D"),
        "rank_excess": dg.get("spec_rank_excess"),
        "eff_rank_P": dg.get("spec_eff_rank_P"),
        "eff_rank_excess": dg.get("spec_eff_rank_excess"),
        "eff_rank_spearman": dg.get("spec_eff_rank_spearman"),
        "eff_rank_excess_spearman": dg.get("spec_eff_rank_excess_spearman"),
        "eff_rank_growth": dg.get("spec_eff_rank_growth"),
        "D_rms": dg.get("spec_D_rms"),
        "tau_mean": dg.get("spec_tau_mean"),
        "gate_mean": dg.get("spec_gate_mean"),
        "sigma_min_W": dg.get("spec_sigma_min_W"),
        "cond_max": dg.get("spec_cond_max"),
        "cos_prev_max": dg.get("spec_cos_prev_max"),
        "cos_prev_P_max": dg.get("spec_cos_prev_P_max"),
        "cos_prev_P_mean": dg.get("spec_cos_prev_P_mean"),
        "theta_grad_norm": dg.get("theta_grad_norm"),
        "config": json.loads(json.dumps(cfg, default=str)),
    }
    if is_main:
        with open(run_dir / "results.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        with open(run_dir / "diagnostics.json", "w", encoding="utf-8") as fh:
            json.dump({"final": dg, "history": history}, fh, indent=2)

        logger.close()
        if tb:
            tb.close()
        if wb:
            wb.finish()

        print(f"[done] best val ppl {best_ppl:.3f} | {elapsed/60:.1f} min | "
              f"peak VRAM {peak_vram:.2f} GB")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    return results
