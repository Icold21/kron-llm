"""Diagnostic and comparison metrics (sections 5.1 - 5.3 of the spec).

Everything here is measurement only: no gradients are created and no module
state is mutated.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .kron import KronCompose, effective_rank

ROLES = ("q", "k", "v", "o", "f")
ALL_ROLES = ("q", "k", "v", "o", "f", "fg")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _module(block: nn.Module, role: str) -> Optional[KronCompose]:
    if role == "f":
        mod = block.ffn
    elif role == "fg":
        mod = getattr(block, "ffn_g", None)
    else:
        mod = getattr(block, role, None)
    return mod if isinstance(mod, KronCompose) else None


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation without scipy."""
    n = len(x)
    if n < 3:
        return float("nan")

    def ranks(v: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


# ---------------------------------------------------------------------------
# sigma(theta) -- D1.1, D1.2, H3
# ---------------------------------------------------------------------------
@torch.no_grad()
def sigma_theta_report(model: nn.Module) -> Dict[str, Any]:
    per_type: Dict[str, List[List[float]]] = {r: [] for r in ALL_ROLES}
    for block in model.blocks:
        for role in ALL_ROLES:
            mod = _module(block, role)
            if mod is None:
                continue
            per_type[role].append(torch.sigmoid(mod.theta).tolist())

    flat = [v for rows in per_type.values() for row in rows for v in row]
    out: Dict[str, Any] = {
        "sigma_theta_mean": sum(flat) / len(flat) if flat else float("nan"),
        "sigma_theta_per_type": per_type,
        "sigma_theta_heatmap": per_type.get("q", []),
    }
    # variance across the L memory channels, per layer/type (H3)
    variances = [torch.tensor(rows).var(unbiased=False).item()
                 for rows in per_type.values() for row in rows if len(row) > 1]
    out["sigma_theta_var"] = sum(variances) / len(variances) if variances else 0.0
    out["sigma_theta_var_max"] = max(variances) if variances else 0.0
    return out


@torch.no_grad()
def theta_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for block in model.blocks:
        for role in ALL_ROLES:
            mod = _module(block, role)
            if mod is not None and mod.theta.grad is not None:
                total += float(mod.theta.grad.detach().pow(2).sum())
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# spectral analysis of W_{i,l} -- D2.1, D2.4, D3.x, H2
# ---------------------------------------------------------------------------
@torch.no_grad()
def routed_report(model: nn.Module) -> Dict[str, float]:
    """Measure router usage and effective-table diversity for a routed model.

    The report is intentionally cheap enough to run at every validation point:
    it materialises the ``(vocab_size, d_model)`` effective table once and does
    one small SVD.  Non-routed models return an empty dict, so the same training
    loop can log both families.

    Args:
        model: A ``KronTransformer`` or compatible module.

    Returns:
        A mapping with router usage perplexity, used-row counts, the fraction of
        tokens whose own bank row is top-1, the effective rank of the table, and
        the router-logit standard deviation.  Empty for a factorised model.
    """
    routed = getattr(model, "routed", None)
    if routed is None:
        return {}

    logits = routed.route_logits()
    top = torch.topk(logits, k=routed.top_k, dim=-1).indices
    counts = torch.bincount(top.flatten(), minlength=routed.v_eff).float()
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    usage_ppl = torch.exp(-(nz * nz.log()).sum())

    top1 = top[:, 0]
    own = (top1[:routed.vocab_size] ==
           torch.arange(routed.vocab_size, device=top1.device)).float().mean()

    effective = routed.effective_embeddings()
    table_rank = effective_rank(torch.linalg.svdvals(effective))
    return {
        "route_usage_ppl": float(usage_ppl),
        "route_unique_rows": int((counts > 0).sum()),
        "route_top1_unique": int(torch.unique(top1).numel()),
        "route_own_top1": float(own),
        "route_eff_rank": float(table_rank),
        "route_logit_std": float(logits.std()),
    }


@torch.no_grad()
def spectral_report(model: nn.Module, trace: List[Dict[str, Any]],
                    max_layers: Optional[int] = None) -> Dict[str, Any]:
    """SVD of every ``D_{i,l}`` and of the composed map ``P_i``.

    Two rank numbers matter and they answer different questions:

    * ``eff_rank_D`` -- rank of the *innovation* of one channel.  This is what
      the Kronecker parameterisation can actually control and what H2 ("does
      expressivity grow with depth") is about.  A single term has
      ``rank <= min(n1^2, n2^2) = 64``.
    * ``eff_rank_W`` -- rank of the materialised factor ``s Id + D``.  With the
      anchor on it is *always* ``d`` (the identity carries the missing rank),
      which is why the original spec's "full rank W" statement was vacuous.

    ``eff_rank_excess`` is the rank of ``P - s^L Id``, i.e. the part of the
    composition that is not the anchor.
    """
    rows: List[Dict[str, Any]] = []
    prev: Dict[Tuple[str, int], torch.Tensor] = {}
    prev_p: Dict[str, torch.Tensor] = {}
    n = len(trace) if max_layers is None else min(max_layers, len(trace))

    for i in range(n):
        entry = trace[i]
        block = model.blocks[i]
        for role in ALL_ROLES:
            mod = _module(block, role)
            D = entry["D"].get(role)
            if mod is None or D is None:
                continue
            s = mod.s
            s_val = float(s) if s is not None else float("nan")
            tau_val = float(mod.tau.mean())
            gate_val = float(mod.gate) if mod.gate is not None else float("nan")

            # composition-level numbers
            P = mod.compose(D).float()
            sv_p = torch.linalg.svdvals(P)
            eff_p = float(effective_rank(sv_p))
            if s is not None:
                eye = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
                P_exc = P - float(mod.anchor_term) * eye
            else:
                P_exc = P
            eff_exc = float(effective_rank(torch.linalg.svdvals(P_exc)))
            # Rank of the sum-composition alternative on the same trained D_l.
            # This is a diagnostic, not a trained model: it isolates how much of
            # the product's rank loss is due to multiplication itself.  For an
            # already-sum model this reduces to the trained sum map.
            if mod.composition == "sum":
                w = (mod._last_w_mean if getattr(mod, "dynamic_gate", None)
                     else mod.alpha)
                S = (w.float().view(-1, 1, 1) * D.float()).sum(dim=0)
            else:
                S = D.float().sum(dim=0)
            if s is not None:
                S = S + float(s) * torch.eye(S.shape[0], device=S.device,
                                               dtype=S.dtype)
            eff_sum = float(effective_rank(torch.linalg.svdvals(S)))
            cos_p = float("nan")
            if role in prev_p:
                a, b = P.flatten(), prev_p[role].flatten()
                cos_p = float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))
            prev_p[role] = P.clone()

            for l in range(D.shape[0]):
                Dl = D[l].float()
                sv_d = torch.linalg.svdvals(Dl)
                eff_d = float(effective_rank(sv_d))
                d_rms = float(Dl.pow(2).mean().sqrt())
                d_norm = float(Dl.norm())
                s_abs = abs(s_val) if math.isfinite(s_val) else float("nan")
                d_over_s = d_rms / s_abs if s_abs > 0 else float("nan")
                fro_over_s = d_norm / s_abs if s_abs > 0 else float("nan")
                sigma_max_d = float(sv_d.max())
                smax_d_over_s = sigma_max_d / s_abs if s_abs > 0 else float("nan")
                W = Dl if s is None else float(s) * torch.eye(
                    Dl.shape[0], device=Dl.device, dtype=Dl.dtype) + Dl
                sv_w = torch.linalg.svdvals(W)
                smin, smax = float(sv_w.min()), float(sv_w.max())
                cos = float("nan")
                key = (role, l)
                if key in prev:
                    a, b = W.flatten(), prev[key].flatten()
                    cos = float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))
                prev[key] = W.clone()
                rows.append({
                    "layer": i, "role": role, "channel": l,
                    "s": s_val, "tau": tau_val, "gate": gate_val,
                    "D_rms": d_rms,
                    "D_norm": d_norm,
                    "D_over_s": d_over_s,
                    "D_fro_over_s": fro_over_s,
                    "sigma_max_D": sigma_max_d,
                    "sigma_max_D_over_s": smax_d_over_s,
                    "eff_rank_D": eff_d,
                    "eff_rank_W": float(effective_rank(sv_w)),
                    "rank_excess": float(effective_rank(sv_w)) - eff_d,
                    "sigma_min": smin, "sigma_max": smax,
                    "cond": smax / max(smin, 1e-12),
                    "cos_prev": cos,
                    "eff_rank_P": eff_p,
                    "eff_rank_excess": eff_exc,
                    "eff_rank_sum": eff_sum,
                    "sum_over_prod": eff_sum / max(eff_p, 1e-9),
                    "cos_prev_P": cos_p,
                })

    summary: Dict[str, Any] = {}
    if rows:
        mean = lambda key: sum(r[key] for r in rows) / len(rows)   # noqa: E731
        summary["sigma_min_W"] = min(r["sigma_min"] for r in rows)
        summary["sigma_max_W"] = max(r["sigma_max"] for r in rows)
        summary["cond_max"] = max(r["cond"] for r in rows)
        summary["eff_rank_mean"] = mean("eff_rank_W")
        summary["eff_rank_W"] = mean("eff_rank_W")
        summary["eff_rank_D"] = mean("eff_rank_D")
        summary["rank_excess"] = mean("rank_excess")
        summary["eff_rank_P"] = mean("eff_rank_P")
        summary["eff_rank_excess"] = mean("eff_rank_excess")
        summary["eff_rank_sum"] = mean("eff_rank_sum")
        summary["sum_over_prod"] = mean("sum_over_prod")
        summary["D_norm_mean"] = mean("D_norm")
        summary["D_rms"] = mean("D_rms")
        finite = lambda key: [r[key] for r in rows if math.isfinite(r[key])]  # noqa: E731
        mean_finite = lambda key: (sum(finite(key)) / len(finite(key))  # noqa: E731
                                   if finite(key) else float("nan"))
        summary["D_over_s"] = mean_finite("D_over_s")
        summary["D_fro_over_s"] = mean_finite("D_fro_over_s")
        summary["sigma_max_D_over_s"] = mean_finite("sigma_max_D_over_s")
        summary["tau_mean"] = mean("tau")
        summary["gate_mean"] = mean("gate")
        coss = [r["cos_prev"] for r in rows if not math.isnan(r["cos_prev"])]
        summary["cos_prev_max"] = max(coss) if coss else float("nan")
        coss_p = [r["cos_prev_P"] for r in rows if not math.isnan(r["cos_prev_P"])]
        summary["cos_prev_P_max"] = max(coss_p) if coss_p else float("nan")
        summary["cos_prev_P_mean"] = (sum(coss_p) / len(coss_p)) if coss_p else float("nan")
        # H2: does the expressivity of the innovation grow with depth?
        layers = sorted({r["layer"] for r in rows})
        per_layer = [sum(r["eff_rank_D"] for r in rows if r["layer"] == i)
                     / max(1, sum(1 for r in rows if r["layer"] == i))
                     for i in layers]
        per_layer_p = [sum(r["eff_rank_excess"] for r in rows if r["layer"] == i)
                       / max(1, sum(1 for r in rows if r["layer"] == i))
                       for i in layers]
        summary["eff_rank_per_layer"] = per_layer
        summary["eff_rank_spearman"] = spearman(layers, per_layer)
        summary["eff_rank_D_spearman"] = spearman(layers, per_layer)
        summary["eff_rank_excess_per_layer"] = per_layer_p
        summary["eff_rank_excess_spearman"] = spearman(layers, per_layer_p)
        summary["eff_rank_first"] = per_layer[0]
        summary["eff_rank_last"] = per_layer[-1]
        summary["eff_rank_growth"] = per_layer[-1] / max(per_layer[0], 1e-9)
    return {"rows": rows, "summary": summary}


def eff_rank_heatmap(spectral: Mapping[str, Any], role: str = "q",
                     n_layers: Optional[int] = None,
                     L: Optional[int] = None) -> torch.Tensor:
    """(n_layers, L) matrix of effective ranks for plotting."""
    rows = [r for r in spectral["rows"] if r["role"] == role]
    if not rows:
        return torch.zeros(0, 0)
    nl = n_layers or (max(r["layer"] for r in rows) + 1)
    nc = L or (max(r["channel"] for r in rows) + 1)
    m = torch.full((nl, nc), float("nan"))
    for r in rows:
        m[r["layer"], r["channel"]] = r["eff_rank_D"]
    return m


def sigma_heatmap(sigma: Mapping[str, Any], role: str = "q") -> torch.Tensor:
    rows = sigma["sigma_theta_per_type"].get(role, [])
    return torch.tensor(rows) if rows else torch.zeros(0, 0)


# ---------------------------------------------------------------------------
# gradients -- D1.3, D2.3
# ---------------------------------------------------------------------------
@torch.no_grad()
def grad_report(model: nn.Module) -> Dict[str, float]:
    groups: Dict[str, float] = {}
    total = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().pow(2).sum())
        total += g
        key = ("theta" if name.endswith(".theta")
               else "tau" if name.endswith(".tau")
               else "gate" if name.endswith(".gate")
               else "s" if name.endswith(".s")
               else "norm" if "norm" in name
               else "embed" if ("embed" in name or "lm_head" in name)
               else "factors")
        groups[key] = groups.get(key, 0.0) + g
    out = {f"grad_{k}": math.sqrt(v) for k, v in groups.items()}
    out["grad_norm"] = math.sqrt(total)
    return out


# ---------------------------------------------------------------------------
# comparison metrics -- C1..C5
# ---------------------------------------------------------------------------
def comparison_report(model: nn.Module, seq_len: int,
                      batch_size: int = 1) -> Dict[str, float]:
    """``flops_per_token`` is amortised over ``seq_len * batch_size`` tokens.

    The Kron prelude evaluates ``L`` products of ``d x d`` matrices once per
    forward pass, so its cost per token shrinks with the batch.  Reporting it
    at ``batch_size=1`` overstates the Kron cost by ~2x.
    """
    return {
        "n_params": model.n_params,
        "n_params_matrix": model.n_params_matrix,
        "flops_per_token": model.flops_per_token(seq_len, batch_size=batch_size),
        "flops_per_token_train": model.flops_per_token(
            seq_len, training=True, batch_size=batch_size),
    }


def aggregate_eval_stats(stats: Mapping[str, float]) -> Dict[str, float]:
    """Collapse per-layer diagnostics into scalars for logging."""
    ent = [v for k, v in stats.items() if k.endswith("attn_entropy")]
    qmin = [v for k, v in stats.items() if k.endswith("q_sigma_min")]
    kmin = [v for k, v in stats.items() if k.endswith("k_sigma_min")]
    drift = [v for k, v in stats.items() if k.endswith("q_drift")]
    out = {
        "attn_entropy_mean": sum(ent) / len(ent) if ent else float("nan"),
        "attn_entropy_min": min(ent) if ent else float("nan"),
        "attn_entropy_max": max(ent) if ent else float("nan"),
        "q_sigma_min": min(qmin) if qmin else float("nan"),
        "k_sigma_min": min(kmin) if kmin else float("nan"),
        "q_drift_mean": sum(drift) / len(drift) if drift else float("nan"),
        "q_drift_last": drift[-1] if drift else float("nan"),
        "h_norm_ratio_min": stats.get("h_norm_ratio_min", float("nan")),
        "h_norm_ratio_max": stats.get("h_norm_ratio_max", float("nan")),
        "h_norm_ratio_mean": stats.get("h_norm_ratio_mean", float("nan")),
    }
    return out
