"""Compare this machine's runs against the published M5 metrics (downloaded from release v1.0 into
rocm/published/ on first run).

Run:  rocm/run.sh python rocm/compare.py        # prints tables, writes rocm/figures/val_compare.png
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB = Path("rocm/published")
FIG = Path("rocm/figures")
RUNS = {"main": "runs/main", "bias_on": "runs/bias_on", "bias_off": "runs/bias_off"}


RELEASE = "https://github.com/Vaibhaav-Tiwari/deepseek-v3-jax/releases/download/v1.0"


def fetch_published():
    import requests
    PUB.mkdir(parents=True, exist_ok=True)
    for name in RUNS:
        f = PUB / f"metrics-{name}.jsonl"
        if not f.exists():
            r = requests.get(f"{RELEASE}/{f.name}", timeout=60)
            r.raise_for_status()
            f.write_bytes(r.content)


def load(path):
    p = Path(path)
    return [json.loads(l) for l in p.open()] if p.exists() else []


def vals(rows, key="val"):
    return {d["step"]: d[key] for d in rows if key in d}


def tok_s(rows):
    xs = [d["tok_per_s"] for d in rows if "tok_per_s" in d]
    return statistics.median(xs) if xs else float("nan")


fetch_published()
FIG.mkdir(parents=True, exist_ok=True)
fig, (ax, axd) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
for name, run in RUNS.items():
    ours, pub = load(f"{run}/metrics.jsonl"), load(PUB / f"metrics-{name}.jsonl")
    if not ours:
        print(f"[{name}] no local run yet")
        continue
    vo, vp = vals(ours), vals(pub)
    common = sorted(set(vo) & set(vp))
    diffs = [vo[s] - vp[s] for s in common]
    print(f"\n[{name}]  median tok/s: ours {tok_s(ours):,.0f}  published {tok_s(pub):,.0f}")
    print(f"{'step':>6} {'ours':>7} {'M5':>7} {'diff':>7}")
    show = [s for s in common if s in (250, 500, 1000, 2000) or s % 2000 == 0 or s == common[-1]]
    for s in show:
        print(f"{s:>6} {vo[s]:7.3f} {vp[s]:7.3f} {vo[s] - vp[s]:+7.3f}")
    if diffs:
        print(f"max |diff| over {len(diffs)} evals: {max(map(abs, diffs)):.3f}; mean diff {statistics.mean(diffs):+.3f}")
    if name == "main":
        ax.plot(sorted(vp), [vp[s] for s in sorted(vp)], label="published (M5, jax-metal)", lw=2)
        ax.plot(sorted(vo), [vo[s] for s in sorted(vo)], "--", label="this run (RX 7600 XT, ROCm)", lw=2)
        axd.axhline(0, color="gray", lw=0.8)
        axd.plot(common, diffs, color="C1", lw=1.5)
        axd.set(ylabel="ours − published", xlabel="step")
        axd.grid(alpha=0.3)

ax.set(ylabel="validation loss", title="deepseek-v3-jax: val loss, M5 vs ROCm", ylim=(3.3, 5.2))
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "val_compare.png", dpi=150)
print(f"\nwrote {FIG / 'val_compare.png'}")
