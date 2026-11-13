"""Figures for the post.

Run:  .venv/bin/python plots.py            # reads runs/main, runs/bias_on, runs/bias_off
Out:  post/figures/{loss,expert_load,balance,bias}.png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

FIG = Path("post/figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f1f6fd", "#9cc3f0", "#2a78d6", "#0d3d7a"])
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#e9e8e4", ORANGE])
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def read(run):
    return [json.loads(l) for l in open(Path(run) / "metrics.jsonl")]


def loss_plot(rows):
    s = [r["step"] for r in rows]
    v = [r for r in rows if "val" in r]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(s, [r["main"] for r in rows], color=BLUE, lw=2, label="train: next token")
    ax.plot(s, [r["mtp"] for r in rows], color=AQUA, lw=2, label="train: MTP (token after next)")
    ax.plot([r["step"] for r in v], [r["val"] for r in v], color=ORANGE, lw=2, marker="o", ms=4, label="val: next token")
    ax.set(xlabel="step", ylabel="cross-entropy (nats)", ylim=(min(r["main"] for r in rows) - 0.3, 7))
    ax.set_title("Training loss")
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(FIG / "loss.png", dpi=200); plt.close(fig)


def load_matrix(rows, layer):
    return np.array([r["load"][layer] for r in rows]).T                # [E, steps]


def expert_load_plot(on, off, layer=2):
    E = len(on[0]["load"][0])
    vmax = max(load_matrix(on, layer).max(), load_matrix(off, layer).max())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, rows, title in [(axes[0], off, "Without bias balancing"), (axes[1], on, "With aux-loss-free bias")]:
        steps = [r["step"] for r in rows]
        im = ax.imshow(load_matrix(rows, layer), aspect="auto", cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest",
                       extent=[steps[0], steps[-1], E - 0.5, -0.5])
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(False)
    axes[0].set_ylabel(f"expert (MoE layer {layer + 1})")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("share of routed slots"); cb.outline.set_visible(False)
    cb.ax.axhline(1 / E, color=INK, lw=1); cb.ax.text(1.6, 1 / E, " even", va="center", fontsize=8, color=INK2,
                                                      transform=cb.ax.get_yaxis_transform())
    fig.savefig(FIG / "expert_load.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def imbalance(rows):
    # max expert load / even share, averaged over MoE layers (1.0 = perfectly balanced)
    return [np.mean([max(l) * len(l) for l in r["load"]]) for r in rows]


def balance_plot(on, off):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot([r["step"] for r in off], imbalance(off), color=ORANGE, lw=2, label="without bias")
    ax.plot([r["step"] for r in on], imbalance(on), color=BLUE, lw=2, label="with bias")
    ax.axhline(1.0, color=INK2, lw=1, ls=":"); ax.text(on[-1]["step"], 0.97, "perfect balance", ha="right",
                                                      va="top", fontsize=8, color=INK2)
    ax.set_ylim(0.85, None)
    ax.set(xlabel="step", ylabel="busiest expert ÷ even share")
    ax.set_title("How overloaded is the busiest expert?")
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(FIG / "balance.png", dpi=200); plt.close(fig)


def bias_plot(rows, layer=2):
    b = np.array([r["bias"][layer] for r in rows]).T                   # [E, steps]
    b = b - b.mean(0, keepdims=True)    # only differences between experts change the top-k; drop the common drift
    lim = np.abs(b).max() or 1.0
    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(b, aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest",
                   extent=[steps[0], steps[-1], b.shape[0] - 0.5, -0.5])
    ax.set(xlabel="step", ylabel=f"expert (MoE layer {layer + 1})"); ax.grid(False)
    ax.set_title("Routing bias per expert, relative to the layer mean (main run)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("bias − mean  (− = pushed away)")
    cb.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "bias.png", dpi=200); plt.close(fig)


def main(argv=None):
    runs = dict(main="runs/main", on="runs/bias_on", off="runs/bias_off")
    FIG.mkdir(parents=True, exist_ok=True)
    if Path(runs["on"], "metrics.jsonl").exists() and Path(runs["off"], "metrics.jsonl").exists():
        on, off = read(runs["on"]), read(runs["off"])
        expert_load_plot(on, off); balance_plot(on, off)
        print("ablation: final busiest/even  bias on %.2f  off %.2f" % (imbalance(on)[-1], imbalance(off)[-1]))
    if Path(runs["main"], "metrics.jsonl").exists():
        rows = read(runs["main"]); loss_plot(rows); bias_plot(rows)
        v = [r for r in rows if "val" in r]
        print(f"main: step {rows[-1]['step']}  train {rows[-1]['main']:.3f}  val {v[-1]['val']:.3f}" if v else "")
    print("figures in", FIG, file=sys.stderr)


if __name__ == "__main__":
    main()
