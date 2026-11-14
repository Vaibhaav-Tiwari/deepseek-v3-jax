"""Train DeepSeek V3 on the prepared corpus.

Run:  .venv/bin/python train.py                     # default config, ~2.6 h on an M5 Air
      .venv/bin/python train.py --config small      # for 8-16 GB Macs (untested)
      .venv/bin/python train.py --no-bias --steps 2000 --out runs/bias_off   # balancing ablation
"""
import argparse, dataclasses, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import optax
import deepseek_v3 as ds

CONFIGS = {"default": ds.Config(), "small": ds.SMALL, "tiny": ds.TINY}


def get_batch(data, B, T, rng):
    """B random windows of T+1+n_mtp tokens (inputs, next-token and MTP targets)."""
    starts = rng.integers(0, len(data) - T, B)
    return np.stack([data[s:s + T] for s in starts]).astype(np.int32)


def save_checkpoint(path, params, biases, cfg):
    tmp = Path(str(path) + ".tmp")                     # write then rename, so a crash never leaves half a file
    with open(tmp, "wb") as f:
        pickle.dump({"params": jax.tree.map(np.asarray, params), "biases": np.asarray(biases),
                     "cfg": dataclasses.asdict(cfg)}, f)
    os.replace(tmp, path)


def all_finite(tree):
    return all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(tree))


def load_checkpoint(path):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    return jax.tree.map(jnp.asarray, ck["params"]), jnp.asarray(ck["biases"]), ds.Config(**ck["cfg"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default", choices=CONFIGS)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--no-bias", action="store_true", help="disable aux-loss-free balancing (gamma = 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args(argv)

    cfg = CONFIGS[args.config]
    if args.no_bias:
        cfg = dataclasses.replace(cfg, gamma=0.0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_data = np.memmap(Path(args.data) / "train.bin", np.uint16, mode="r")
    val_data = np.memmap(Path(args.data) / "val.bin", np.uint16, mode="r")
    window = cfg.seq_len + 1 + cfg.n_mtp

    warmup = min(args.warmup, args.steps // 2)
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, max(args.steps, warmup + 1), args.lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=0.1))
    params = ds.init_params(cfg, jax.random.key(args.seed))
    biases = jnp.zeros((cfg.n_layers - cfg.n_dense, cfg.n_routed))
    opt_state = opt.init(params)
    step_fn = ds.make_train_step(cfg, opt)
    eval_fn = jax.jit(lambda p, b, x: ds.loss_fn(p, b, x, cfg)[1])
    print(f"{jax.default_backend()} | config={args.config} | {ds.n_params(params) / 1e6:.1f}M params | "
          f"{len(train_data) / 1e6:.1f}M train tokens", file=sys.stderr)

    rng, val_rng = np.random.default_rng(args.seed), np.random.default_rng(1234)
    val_batches = [get_batch(val_data, cfg.batch, window, val_rng) for _ in range(8)]
    log = open(out / "metrics.jsonl", "w")
    t0, tokens = time.perf_counter(), 0
    for step in range(1, args.steps + 1):
        params, opt_state, biases, m = step_fn(params, opt_state, biases, get_batch(train_data, cfg.batch, window, rng))
        tokens += cfg.batch * cfg.seq_len
        if not np.isfinite(float(m["loss"])):          # checked every step, so the error names the real step
            log.write(json.dumps({"step": step, "error": "non-finite loss"}) + "\n"); log.close()
            raise FloatingPointError(f"non-finite loss at step {step}")
        if step % args.log_every == 0 or step == args.steps:
            row = {"step": step, **{k: float(m[k]) for k in ("loss", "main", "mtp", "seq")},
                   "load": np.asarray(m["load"]).tolist(), "bias": np.asarray(biases).tolist(),
                   "tok_per_s": tokens / (time.perf_counter() - t0), "lr": float(sched(step))}
            if step % args.eval_every == 0 or step == args.steps:
                if not all_finite(params):             # the update itself went bad: keep the last checkpoint
                    log.write(json.dumps({"step": step, "error": "non-finite weights"}) + "\n"); log.close()
                    raise FloatingPointError(f"non-finite weights at step {step}; last checkpoint kept")
                vals = [eval_fn(params, biases, jnp.asarray(b)) for b in val_batches]
                row["val"] = float(np.mean([v["main"] for v in vals]))
                row["val_mtp"] = float(np.mean([v["mtp"] for v in vals]))
                save_checkpoint(out / "ckpt.pkl", params, biases, cfg)
            log.write(json.dumps(row) + "\n"); log.flush()
            print(f"step {step:5d} | loss {row['loss']:.3f} main {row['main']:.3f} mtp {row['mtp']:.3f}"
                  + (f" | val {row['val']:.3f}" if "val" in row else "") + f" | {row['tok_per_s']:.0f} tok/s",
                  file=sys.stderr)
    log.close()


if __name__ == "__main__":
    main()
