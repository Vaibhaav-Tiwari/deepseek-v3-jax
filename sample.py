"""Generate text from a trained checkpoint.

Run:  .venv/bin/python sample.py --prompt "The fog came in off the bay" --num 3
The config comes from the checkpoint. Text longer than the context window is produced by
re-prompting with the last half of the window.
"""
import argparse
import jax, jax.numpy as jnp
from tokenizers import Tokenizer
import deepseek_v3 as ds
from train import load_checkpoint


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/main/ckpt.pkl")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="I first heard")
    ap.add_argument("--n", type=int, default=300, help="new tokens per sample")
    ap.add_argument("--num", type=int, default=3, help="number of samples")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    params, biases, cfg = load_checkpoint(args.ckpt)
    tok = Tokenizer.from_file(args.tokenizer)
    key = jax.random.key(args.seed)
    for i in range(args.num):
        ids, left = tok.encode(args.prompt).ids, args.n
        while left > 0:
            ctx = ids[-(cfg.seq_len // 2):] if len(ids) > cfg.seq_len // 2 else ids
            n = min(left, cfg.seq_len - len(ctx))
            key, sub = jax.random.split(key)
            out = ds.generate(params, biases, jnp.array([ctx]), n, cfg, sub, args.temp, args.top_k)
            ids += [int(t) for t in out[0, len(ctx):]]; left -= n
        print(f"--- sample {i + 1} ---\n{tok.decode(ids)}\n")


if __name__ == "__main__":
    main()
