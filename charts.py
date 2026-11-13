"""Real-data charts behind the blog diagrams: every number here comes from a log, the checkpoint or the corpus.

Run:  JAX_PLATFORMS=cpu .venv/bin/python charts.py
Out:  post/figures/{lr,throughput,matmul,expert_load_final,router_scores,next_word,books,rope}.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plots                                    # shared style (rcParams, colors)
from plots import FIG, INK, INK2, GRID, BLUE, ORANGE, AQUA, read

PURPLE, GREY = "#ae3ec9", "#c9c8c3"
PROMPT = "The fog came in off the"


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=200); plt.close(fig); print("wrote", name)


def lr_plot(rows):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot([r["step"] for r in rows], [r["lr"] for r in rows], color=AQUA, lw=2)
    ax.set(xlabel="step", ylabel="learning rate", ylim=(0, None))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title("Learning rate, as logged: 200 warmup steps, cosine down to 6e-5")
    save(fig, "lr")


def throughput_plot(rows):
    s, t = np.array([r["step"] for r in rows]), np.array([r["tok_per_s"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(s, t / 1e3, color=ORANGE, lw=1.2, alpha=0.35, label="logged every 25 steps")
    k = 20
    avg = np.convolve(t, np.ones(k) / k, "valid")                  # centered, so it doesn't lag the raw line
    ax.plot(s[k // 2:k // 2 + len(avg)], avg / 1e3, color=ORANGE, lw=2.2, label="500-step average")
    ax.set(xlabel="step", ylabel="tokens / s (thousands)")
    ax.set_title("Training speed")
    ax.legend(frameon=False, labelcolor=INK2)
    save(fig, "throughput")


def matmul_plot():
    # measured by smoke_metal.py on this machine (post/screenshots/smoke_metal.png)
    names, tf = ["float32", "bfloat16"], [2.76, 11.10]
    fig, ax = plt.subplots(figsize=(7, 1.9))
    ax.barh(names, tf, color=[GREY, BLUE], height=0.55)
    for i, v in enumerate(tf): ax.text(v + 0.15, i, f"{v:.2f} TFLOP/s", va="center", fontsize=10, color=INK)
    ax.invert_yaxis(); ax.set_xlim(0, 13); ax.grid(axis="y", visible=False)
    ax.set(xlabel="TFLOP/s"); ax.set_title("2048 × 2048 matmul on the M5 GPU (jax-metal)")
    save(fig, "matmul")


def expert_load_final(on, off, layer=2):
    E = len(on[-1]["load"][layer]); x = np.arange(E)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(x - 0.2, np.array(off[-1]["load"][layer]) * 100, 0.4, color=ORANGE, label="without bias")
    ax.bar(x + 0.2, np.array(on[-1]["load"][layer]) * 100, 0.4, color=BLUE, label="with bias")
    ax.axhline(100 / E, color=INK2, lw=1, ls=":"); ax.text(E - 0.5, 100 / E + 0.3, "even: 6.25%", ha="right",
                                                          fontsize=8, color=INK2)
    ax.set_xticks(x); ax.grid(axis="x", visible=False)
    ax.set(xlabel=f"expert (MoE layer {layer + 1})", ylabel="share of routed slots (%)")
    ax.set_title(f"Expert load at the last step (step {on[-1]['step']})")
    ax.legend(frameon=False, labelcolor=INK2)
    save(fig, "expert_load_final")


def model_charts():
    import jax, jax.numpy as jnp
    from tokenizers import Tokenizer
    import deepseek_v3 as ds
    from train import load_checkpoint
    params, biases, cfg = load_checkpoint("runs/main/ckpt.pkl")
    tok = Tokenizer.from_file("data/tokenizer.json")
    ids = tok.encode(PROMPT).ids
    tokens = jnp.array([ids])

    # --- router scores of the last prompt token at MoE layer 3 (same math as ds.forward / ds.route)
    layer = 3                                    # model layer index; MoE layer 3 = biases[2]
    cos, sin = ds.rope(cfg, jnp.arange(len(ids)))
    x = params["embed"][tokens]
    for i, p in enumerate(params["layers"][:layer]):
        bias = biases[i - cfg.n_dense] if i >= cfg.n_dense else None
        x, _, _ = ds.block(p, bias, x, cos, sin, cfg)
    p = params["layers"][layer]
    a, _ = ds.mla(p["attn"], ds.rms_norm(x, p["norm_attn"]), cos, sin, cfg)
    h = ds.rms_norm(x + a, p["norm_ffn"])[0, -1:]
    b = biases[layer - cfg.n_dense]
    gates, scores = ds.route(p["moe"]["router"], b, h, cfg)
    s, g, bb = np.array(scores[0]), np.array(gates[0], np.float32), np.array(b)
    picked = g > 0
    fig, ax = plt.subplots(figsize=(7, 3.4))
    E = np.arange(len(s))
    ax.bar(E, s, color=[BLUE if k else GREY for k in picked], width=0.7, label="sigmoid score")
    ax.scatter(E, s + bb, marker="_", s=260, color=PURPLE, lw=2.2, zorder=3, label="score + bias (used to pick)")
    for e in E[picked]: ax.text(e, max(s[e], s[e] + bb[e]) + 0.02, f"{g[e]:.2f}", ha="center", fontsize=9,
                               color=BLUE, weight="bold")
    ax.set_xticks(E); ax.grid(axis="x", visible=False); ax.set_ylim(0, 1.12)
    last = tok.decode([ids[-1]])
    ax.set(xlabel=f"expert (MoE layer {layer - cfg.n_dense + 1})", ylabel="score")
    ax.set_title(f'Router scores for "{last.strip()}" at the end of "{PROMPT}"')
    ax.legend(frameon=False, labelcolor=INK2, loc="upper center", fontsize=8, ncol=2)
    ax.text(15.4, 1.0, "blue numbers: gates\n(picked scores ÷ their sum)", ha="right", va="top", fontsize=8, color=BLUE)
    save(fig, "router_scores")
    print("router: picked", list(E[picked]), "scores", np.round(s, 3).tolist(), "bias", np.round(bb, 4).tolist(),
          "gates", np.round(g[picked], 3).tolist())

    # --- next-word distribution after the prompt, as sample.py would draw it (temp 0.8, top-50)
    logits = np.array(ds.forward(params, biases, tokens, cfg)[0][0, -1], np.float64) / 0.8
    top = np.argsort(-logits)[:50]
    p50 = np.exp(logits[top] - logits[top].max()); p50 /= p50.sum()
    n = 10
    words = [repr(tok.decode([int(t)]))[1:-1] for t in top[:n]]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(range(n), p50[:n] * 100, color=[AQUA] + ["#a6e3c8"] * (n - 1), height=0.65)
    for i, v in enumerate(p50[:n]): ax.text(v * 100 + 0.4, i, f"{v * 100:.1f}%", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(n), words, fontfamily="monospace"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set(xlabel="probability after temperature 0.8 and top-50 (%)")
    ax.set_title(f'What the trained model says comes after "{PROMPT}"')
    save(fig, "next_word")
    print("next word:", list(zip(words, np.round(p50[:n], 4).tolist())))

    # --- RoPE: the angle each of the 8 pairs has turned to, by position (from ds.rope itself)
    cos_r, sin_r = ds.rope(cfg, jnp.arange(cfg.seq_len))
    c = np.array(cos_r).T                                                      # [d_rope/2, T]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    im = ax.imshow(c, aspect="auto", cmap=plots.DIV, vmin=-1, vmax=1, interpolation="nearest",
                   extent=[-0.5, cfg.seq_len - 0.5, c.shape[0] - 0.5, -0.5])
    ax.set(xlabel="position in the window", ylabel="RoPE pair"); ax.grid(False)
    ax.set_title("cos(position × θ_i) for the 8 pairs: fast pairs flicker, slow pairs drift")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False)
    save(fig, "rope")


def books_plot():
    import prepare_data as pd
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("data/tokenizer.json")
    rows = []
    for kind, url, title in pd.load_sources():
        if kind == "standardebooks": text = pd.standard_ebook(pd.fetch(url))
        else: text = pd.clean_text(pd.strip_gutenberg(pd.fetch(url).decode("utf-8-sig", errors="replace"), title))
        rows.append((kind == "standardebooks", title, text))
    counts = [len(e.ids) for e in tok.encode_batch([t for _, _, t in rows])]
    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(order)), [counts[i] / 1e3 for i in order], width=1.0,
           color=[ORANGE if rows[i][0] else GREY for i in order])
    ax.set(xlabel=f"the {len(rows)} books, largest first", ylabel="tokens (thousands)", xlim=(-1, len(rows)))
    ax.grid(axis="x", visible=False); ax.set_xticks([])
    ax.set_title(f"The corpus: {sum(counts) / 1e6:.1f}M tokens before repeating Hammett (orange) 3×")
    save(fig, "books")
    print("books:", len(rows), "total tokens", sum(counts), "largest", rows[order[0]][1], counts[order[0]])


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = read("runs/main")
    lr_plot(rows); throughput_plot(rows); matmul_plot()
    expert_load_final(read("runs/bias_on"), read("runs/bias_off"))
    model_charts(); books_plot()


if __name__ == "__main__":
    main()
