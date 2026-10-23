"""KronLLM interactive dashboard (Gradio).

Run with:
    python scripts/dashboard.py
    python scripts/dashboard.py --port 7860 --no-browser

Tabs
----
* Training    -- launch/stop runs, live log and live loss curve
* Generation  -- sample from any checkpoint, optionally A/B against another run
* Metrics     -- results.json, diagnostics, heatmaps/spectra, training table
* Experiments -- comparison table + quality-vs-budget plot, "run all" button
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from kronllm.config import EXPERIMENTS  # noqa: E402
from kronllm.generate import generate_texts  # noqa: E402
from kronllm.plots import (compare_plot, diag_curves, loss_curves,  # noqa: E402
                           read_log, read_summary, results_table)

RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts"
EXP_IDS = list(EXPERIMENTS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------
class Job:
    """A single background subprocess with a log file."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.label: Optional[str] = None
        self.log_path: Optional[Path] = None
        self.run_dir: Optional[Path] = None
        self.t0: float = 0.0
        self._fh = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: List[str], label: str, log_path: Path,
              run_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """Start ``cmd`` in the repository root and stream it to ``log_path``."""
        if self.running:
            return False, f"Already running: {self.label}"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = open(log_path, "w", encoding="utf-8", buffering=1)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=self._fh,
                                     stderr=subprocess.STDOUT, env=env, **kwargs)
        self.label, self.log_path, self.run_dir = label, log_path, run_dir
        self.t0 = time.time()
        return True, f"Started: {label}"

    def stop(self) -> str:
        """Terminate the process tree (Windows) or the process itself."""
        if not self.running:
            return "No active job"
        assert self.proc is not None
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID",
                                str(self.proc.pid)], capture_output=True)
            else:
                self.proc.terminate()
        except Exception as exc:  # pragma: no cover
            return f"Could not stop: {exc}"
        return f"Stopped: {self.label}"

    def status(self) -> str:
        if self.proc is None:
            return "idle"
        rc = self.proc.poll()
        dt = time.time() - self.t0
        if rc is None:
            return f"running: **{self.label}** ({dt:.0f}s)"
        return f"finished: **{self.label}** (exit {rc}, {dt:.0f}s)"

    def tail(self, n: int = 90) -> str:
        if not self.log_path or not Path(self.log_path).exists():
            return ""
        try:
            lines = Path(self.log_path).read_text(
                encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""
        return "\n".join(lines[-n:])


JOB = Job()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_runs() -> List[str]:
    """Return the names of all run directories that contain a checkpoint or log."""
    if not RUNS.exists():
        return []
    out = []
    for path in RUNS.iterdir():
        if not path.is_dir():
            continue
        if (path / "best.pt").exists() or (path / "last.pt").exists() or \
                (path / "log.csv").exists():
            out.append(path.name)
    return sorted(out, reverse=True)


_MODEL_CACHE: Dict[str, Any] = {}
_FIGS: Dict[str, Any] = {}


def _fig(name: str, size=(7, 3.6)):
    """Long-lived matplotlib figures (the dashboard polls every 2 s)."""
    import matplotlib.pyplot as plt

    if name not in _FIGS:
        _FIGS[name] = plt.figure(figsize=size)
    return _FIGS[name]


def load_cached(run: str, ckpt: str = "best.pt"):
    """Load a checkpoint, reusing the model while the file is unchanged."""
    from kronllm.cli import _load_model

    run_dir = RUNS / run
    path = run_dir / ckpt
    if not path.exists():
        path = run_dir / "last.pt"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    key = f"{path}:{path.stat().st_mtime}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model, cfg, ck = _load_model(run_dir, ckpt)
    model.to(DEVICE)
    _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = (model, cfg, ck)
    return model, cfg, ck


def data_status() -> str:
    """One-line summary of the prepared corpora and tokenizer."""
    parts = []
    for name, path in (("train", DATA / "train.bin"), ("val", DATA / "val.bin"),
                       ("train.bpe", DATA / "train.bpe.bin"),
                       ("val.bpe", DATA / "val.bpe.bin")):
        if path.exists():
            parts.append(f"{name}: {path.stat().st_size/1e6:.1f} MB")
    meta = DATA / "tokenizer_meta.json"
    if meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        parts.append(f"BPE {info['vocab_size']} · "
                     f"{info['bytes_per_token']:.2f} bytes/token")
    return (" · ".join(parts) if parts else "no data") + \
        ("  \n`python scripts/prepare_data.py` and "
         "`python scripts/prepare_bpe.py`" if not parts else "")


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if value != value:
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


# ---------------------------------------------------------------------------
# Training tab
# ---------------------------------------------------------------------------
def start_training(exp: str, steps: float, epochs: float, batch: float,
                   lr: float, overrides: str, amp: bool) -> Tuple[str, str]:
    """Launch ``kronllm.cli train`` with the form values."""
    if JOB.running:
        return JOB.status(), JOB.tail()

    if not (DATA / "train.bin").exists() and not (DATA / "train.bpe.bin").exists():
        return ("No `data/train.bin`. Press \"Prepare data\" or run "
                "`python scripts/prepare_data.py`."), ""

    cmd = [sys.executable, "-u", "-m", "kronllm.cli", "train"]
    if exp and exp != "custom":
        cmd += ["--exp", exp]
    if steps and steps > 0:
        cmd += ["--steps", str(int(steps))]
    if epochs and epochs > 0:
        cmd += ["--epochs", str(int(epochs))]
    if batch and batch > 0:
        cmd += ["--batch-size", str(int(batch))]
    if lr and lr > 0:
        cmd += ["--lr", f"{lr:g}"]
    if not amp:
        cmd += ["--no-amp"]
    for line in (overrides or "").splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            cmd += ["--override", line]

    run_dir = RUNS / (exp if exp and exp != "custom" else "custom")
    ok, msg = JOB.start(cmd, f"train {exp or 'custom'}", run_dir / "train.log", run_dir)
    return (f"{msg}\n\n`{' '.join(cmd)}`\n\n" + ("" if ok else "")), JOB.tail()


def start_data_prep(max_stories: float) -> Tuple[str, str]:
    """Launch ``scripts/prepare_data.py`` in the background."""
    if JOB.running:
        return JOB.status(), JOB.tail()
    cmd = [sys.executable, "-u", str(SCRIPTS / "prepare_data.py")]
    if max_stories and max_stories > 0:
        cmd += ["--max-stories", str(int(max_stories))]
    JOB.start(cmd, "prepare_data", RUNS / "_prepare.log")
    return "TinyStories preparation started...", JOB.tail()


def start_all_experiments(epochs: float, steps: float) -> Tuple[str, str]:
    """Launch the whole E0-E13 series in the background."""
    if JOB.running:
        return JOB.status(), JOB.tail()
    cmd = [sys.executable, "-u", str(SCRIPTS / "run_experiments.py"), "--all"]
    if epochs and epochs > 0:
        cmd += ["--epochs", str(int(epochs))]
    if steps and steps > 0:
        cmd += ["--steps", str(int(steps))]
    JOB.start(cmd, "E0-E13", RUNS / "_experiments.log")
    return "The E0-E13 series has started...", JOB.tail()


def refresh_training(live_run: str):
    """Poll the job status, its log tail and the live loss curve."""
    status = JOB.status()
    log = JOB.tail()
    run = live_run or (JOB.run_dir.name if JOB.run_dir else None)
    log_csv = (RUNS / run / "log.csv") if run else None
    fig = (loss_curves(log_csv, fig=_fig("live"))
           if log_csv and log_csv.exists() else None)
    return status, log, fig


# ---------------------------------------------------------------------------
# Generation tab
# ---------------------------------------------------------------------------
def _gen_one(run: str, ckpt: str, prompt: str, n: int, max_new: int,
             temp: float, top_k: int, seed: float, stop_eot: bool) -> str:
    """Sample ``n`` continuations from one checkpoint and format them."""
    if not run:
        return "Select a run (checkpoint)."
    try:
        model, cfg, ck = load_cached(run, ckpt)
    except Exception as exc:
        return f"Failed to load: {exc}"
    t0 = time.time()
    from kronllm.tokenizer import load_tokenizer
    tok = load_tokenizer(cfg)
    texts = generate_texts(
        model, [prompt], max_new_tokens=int(max_new), temperature=float(temp),
        top_k=int(top_k) if top_k and top_k > 0 else None,
        n_samples=int(n), seed=None if seed is None or seed < 0 else int(seed),
        device=DEVICE, stop_at_eot=bool(stop_eot), tokenizer=tok)
    dt = time.time() - t0
    head = (f"**{run}** · {ckpt} · step {ck.get('step', '?')} · "
            f"val_ppl {_fmt(float(ck.get('val_ppl', float('nan'))), 2)} · "
            f"{model.n_params:,} params · {tok.name}/{tok.vocab_size} · {dt:.2f}s\n\n")
    return head + ("\n\n" + "-" * 60 + "\n\n").join(texts)


def do_generate(run, ckpt, prompt, n, max_new, temp, top_k, seed, stop_eot,
                run_b, ckpt_b):
    """Sample from run A and, optionally, run B for an A/B comparison."""
    a = _gen_one(run, ckpt, prompt, n, max_new, temp, top_k, seed, stop_eot)
    b = ""
    if run_b and run_b not in ("", "none"):
        b = _gen_one(run_b, ckpt_b, prompt, n, max_new, temp, top_k, seed, stop_eot)
    return a, b


# ---------------------------------------------------------------------------
# Metrics tab
# ---------------------------------------------------------------------------
def show_metrics(run: str):
    """Render ``results.json``, the analysis images and the log tail."""
    if not run:
        return ("Select a run.", None, None, None, None, None, pd.DataFrame())
    directory = RUNS / run
    lines = [f"## `{run}`"]
    res_path = directory / "results.json"
    if res_path.exists():
        try:
            results = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception:
            results = {}
        keys = [
            ("experiment", "experiment"), ("n_params", "total parameters"),
            ("n_params_matrix", "parameters in matrices"),
            ("val_loss", "val loss"), ("val_ppl", "val PPL"),
            ("best_val_ppl", "best val PPL"),
            ("ppl_ratio_vs_E0", "PPL / E0"),
            ("flops_per_token", "FLOPs/token"),
            ("time_per_epoch_sec", "time/epoch, s"),
            ("sec_per_step", "s/step"), ("peak_vram_gb", "peak VRAM, GB"),
            ("sigma_theta_mean", "sigma(theta) mean"),
            ("sigma_theta_var", "Var_l sigma(theta)  (H3 > 0.01)"),
            ("attn_entropy_mean", "attention entropy"),
            ("q_sigma_min", "sigma_min(Q)"),
            ("q_drift_mean", "norm(Q-x)/norm(x)"),
            ("h_norm_ratio_mean", "norm(h_i)/norm(h_{i-1})"),
            ("D_rms", "RMS(D)  (= tau)"),
            ("tau_mean", "tau mean"),
            ("gate_mean", "gamma (ReZero) mean"),
            ("eff_rank_D", "eff. rank D (innovation)"),
            ("eff_rank_W", "eff. rank W = sI+D"),
            ("rank_excess", "eff.rank(W) - eff.rank(D)"),
            ("eff_rank_P", "eff. rank P (composition)"),
            ("eff_rank_excess", "eff. rank (P - s^L I)"),
            ("eff_rank_spearman", "Spearman eff.rank(D)~depth (H2)"),
            ("eff_rank_growth", "eff. rank growth (last/first)"),
            ("sigma_min_W", "sigma_min(W)"),
            ("cond_max", "cond(W) max"),
            ("cos_prev_max", "cos(W_i, W_{i-1}) max"),
            ("cos_prev_P_max", "cos(P_i, P_{i-1}) max"),
        ]
        rows = ["| metric | value |", "|---|---|"]
        for key, label in keys:
            if key in results and results[key] is not None:
                rows.append(f"| {label} | `{_fmt(results[key])}` |")
        lines.append("\n".join(rows))
        verdicts = []
        ratio = results.get("ppl_ratio_vs_E0")
        if ratio is not None and ratio <= 1.2:
            verdicts.append("**H1 confirmed**: PPL <= 1.2x E0")
        elif ratio is not None:
            verdicts.append("H1 rejected: PPL > 1.2x E0")
        spearman = results.get("eff_rank_spearman")
        if spearman is not None and spearman > 0.7:
            verdicts.append(f"**H2 confirmed**: rank grows with depth ({spearman:.2f})")
        elif spearman is not None:
            verdicts.append(f"H2 rejected: rank does not grow with depth ({spearman:.2f})")
        if results.get("eff_rank_D") is not None:
            verdicts.append(f"eff.rank(D) = {results['eff_rank_D']:.1f} "
                            f"(d = 128); eff.rank(W) = "
                            f"{results.get('eff_rank_W', float('nan')):.1f}")
        variance = results.get("sigma_theta_var")
        if variance is not None and variance > 0.01:
            verdicts.append("**H3 confirmed**: memory channels differ")
        if verdicts:
            lines.append("### Hypotheses\n- " + "\n- ".join(verdicts))
    else:
        lines.append("`results.json` is not there yet (training is not finished).")

    heatmaps = directory / "analysis" / "heatmaps.png"
    spectra = directory / "analysis" / "spectra.png"
    rank = directory / "analysis" / "rank_vs_depth.png"
    scales = directory / "analysis" / "scales.png"
    frame = read_log(directory / "log.csv")
    if not frame.empty:
        frame = frame.tail(40)
    return (
        "\n\n".join(lines),
        str(heatmaps) if heatmaps.exists() else None,
        str(spectra) if spectra.exists() else None,
        str(rank) if rank.exists() else None,
        str(scales) if scales.exists() else None,
        loss_curves(directory / "log.csv", fig=_fig("metrics")),
        frame,
    )


def run_analysis(run: str, samples: int, progress=gr.Progress()):
    """Run ``scripts/analyze.py`` synchronously and reload its images."""
    if not run:
        return "Select a run.", None, None, None, None
    if JOB.running:
        return f"Wait: {JOB.label} is still running.", None, None, None, None
    progress(0.1, desc="analysing...")
    cmd = [sys.executable, "-u", str(SCRIPTS / "analyze.py"),
           "--run", str(RUNS / run), "--samples", str(int(samples))]
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=1800)
    except subprocess.TimeoutExpired:
        return "Analysis timed out (>30 min).", None, None, None, None
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.returncode else "")
    directory = RUNS / run / "analysis"

    def img(name: str):
        path = directory / name
        return str(path) if path.exists() else None

    text = f"exit code {proc.returncode}\n\n```\n{out[-2500:]}\n```"
    return (text, img("heatmaps.png"), img("spectra.png"),
            img("rank_vs_depth.png"), img("scales.png"))


# ---------------------------------------------------------------------------
# Comparison tab
# ---------------------------------------------------------------------------
def refresh_comparison():
    """Rebuild the cross-run table and the quality-vs-budget plot."""
    return results_table(RESULTS), compare_plot(
        RESULTS, fig=_fig("cmp", (11, 3.8)))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    """Build the four-tab Gradio interface."""
    runs = list_runs()
    default_run = runs[0] if runs else None

    with gr.Blocks(title="KronLLM dashboard") as demo:
        gr.Markdown(
            "# KronLLM · interactive dashboard\n"
            f"device: `{DEVICE}` · data: {data_status()}")

        # ===============================================================
        with gr.Tab("Training"):
            with gr.Row():
                with gr.Column(scale=1):
                    exp = gr.Dropdown(EXP_IDS, value="E2", label="experiment",
                                      info="E2 - the base Kronecker model")
                    steps = gr.Number(value=0, label="max_steps (0 = full run)",
                                      precision=0)
                    epochs = gr.Number(value=0, label="epochs (0 = from config)",
                                       precision=0)
                    batch = gr.Number(value=0, label="batch_size (0 = from config)",
                                      precision=0)
                    lr = gr.Number(value=0.0, label="lr (0 = from config)")
                    amp = gr.Checkbox(
                        value=False,
                        label="AMP fp16 (slower than fp32 on the reference GPU)")
                    overrides = gr.Textbox(
                        label="extra overrides (k=v, one per line)",
                        placeholder="model.kron.L=6\nmodel.kron.K=4",
                        lines=3)
                    with gr.Row():
                        btn_start = gr.Button("Start", variant="primary")
                        btn_stop = gr.Button("Stop")
                    prep_stories = gr.Number(value=200000,
                                             label="stories for data preparation",
                                             precision=0)
                    with gr.Row():
                        btn_data = gr.Button("Prepare data")
                        btn_all = gr.Button("Run the E0-E13 series")
                    live_run = gr.Textbox(value="E2", label="run for the live curve",
                                          interactive=True)
                with gr.Column(scale=2):
                    status = gr.Markdown("idle")
                    log = gr.Textbox(label="log", lines=16, max_lines=16,
                                     autoscroll=True, interactive=False)
                    curve = gr.Plot(label="loss")
            timer = gr.Timer(2.0)
            timer.tick(refresh_training, inputs=[live_run],
                       outputs=[status, log, curve])
            btn_start.click(start_training,
                            inputs=[exp, steps, epochs, batch, lr, overrides, amp],
                            outputs=[status, log])
            btn_stop.click(lambda: (JOB.stop(), JOB.tail()),
                           outputs=[status, log])
            btn_data.click(start_data_prep, inputs=[prep_stories],
                           outputs=[status, log])
            btn_all.click(start_all_experiments, inputs=[epochs, steps],
                          outputs=[status, log])
            demo.load(refresh_training, inputs=[live_run],
                      outputs=[status, log, curve])

        # ===============================================================
        with gr.Tab("Generation"):
            gr.Markdown("Sample from a checkpoint. Byte-level models encode the "
                        "prompt as UTF-8 and decode the answer back.")
            with gr.Row():
                run_a = gr.Dropdown(runs, value=default_run, label="run A",
                                    allow_custom_value=True)
                ckpt_a = gr.Dropdown(["best.pt", "last.pt"], value="best.pt",
                                     label="checkpoint A")
                run_b = gr.Dropdown(["none"] + runs,
                                    value="none", label="run B (A/B, optional)",
                                    allow_custom_value=True)
                ckpt_b = gr.Dropdown(["best.pt", "last.pt"], value="best.pt",
                                     label="checkpoint B")
            prompt = gr.Textbox(value="Once upon a time", label="prompt", lines=2)
            with gr.Row():
                n_samples = gr.Slider(1, 20, value=3, step=1, label="samples")
                max_new = gr.Slider(20, 600, value=180, step=10,
                                    label="max new tokens")
                temp = gr.Slider(0.1, 1.8, value=1.0, step=0.05,
                                 label="temperature")
                top_k = gr.Slider(0, 256, value=8, step=1, label="top_k")
                seed = gr.Number(value=-1, label="seed (-1 = random)",
                                 precision=0)
                stop_eot = gr.Checkbox(value=True, label="stop at <|endoftext|>")
            with gr.Row():
                btn_gen = gr.Button("Generate", variant="primary")
                btn_refresh_runs = gr.Button("Refresh run list")
            with gr.Row():
                out_a = gr.Textbox(label="A", lines=18, max_lines=18,
                                   autoscroll=False)
                out_b = gr.Textbox(label="B", lines=18, max_lines=18,
                                   autoscroll=False)
            btn_gen.click(do_generate,
                          inputs=[run_a, ckpt_a, prompt, n_samples, max_new,
                                  temp, top_k, seed, stop_eot, run_b, ckpt_b],
                          outputs=[out_a, out_b])
            btn_refresh_runs.click(
                lambda: (gr.update(choices=list_runs()),
                         gr.update(choices=["none"] + list_runs())),
                outputs=[run_a, run_b])

        # ===============================================================
        with gr.Tab("Metrics"):
            with gr.Row():
                run_m = gr.Dropdown(runs, value=default_run, label="run",
                                    allow_custom_value=True)
                btn_m = gr.Button("Refresh")
                btn_an = gr.Button("Run analysis")
                n_an = gr.Number(value=10, label="generation samples",
                                 precision=0)
            info_md = gr.Markdown("Select a run.")
            with gr.Row():
                img_heat = gr.Image(label="heatmaps: eff.rank(D) / sigma(theta)",
                                    type="filepath")
                img_spec = gr.Image(label="spectra D (channel 0)", type="filepath")
            with gr.Row():
                img_rank = gr.Image(label="H2: rank vs depth", type="filepath")
                img_scales = gr.Image(label="tau and gamma per layer", type="filepath")
            fig_m = gr.Plot(label="loss curves")
            tbl = gr.Dataframe(label="log.csv (tail)", interactive=False)
            metrics_out = [info_md, img_heat, img_spec, img_rank, img_scales,
                           fig_m, tbl]
            btn_m.click(show_metrics, inputs=[run_m], outputs=metrics_out)
            run_m.change(show_metrics, inputs=[run_m], outputs=metrics_out)
            btn_an.click(run_analysis, inputs=[run_m, n_an],
                         outputs=[info_md, img_heat, img_spec, img_rank,
                                  img_scales])
            demo.load(show_metrics, inputs=[run_m], outputs=metrics_out)

        # ===============================================================
        with gr.Tab("Experiments"):
            gr.Markdown("Summary of every finished experiment "
                        "(`results/summary.csv`).")
            with gr.Row():
                btn_cmp = gr.Button("Refresh", variant="primary")
            tbl_cmp = gr.Dataframe(label="comparison", interactive=False)
            fig_cmp = gr.Plot(label="PPL")
            btn_cmp.click(refresh_comparison, outputs=[tbl_cmp, fig_cmp])
            demo.load(refresh_comparison, outputs=[tbl_cmp, fig_cmp])

        gr.Markdown(
            "---\n"
            "TensorBoard: `tensorboard --logdir runs` · "
            "CLI: `python -m kronllm.cli --help`")
    return demo


def main() -> None:
    """Parse the CLI arguments and launch the Gradio server."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    demo = build_ui()
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port,
                inbrowser=not args.no_browser, share=args.share,
                show_error=True)


if __name__ == "__main__":
    main()
