"""CPU vs GPU numerics on the very first training batch (same init, same tokens).

Run (GPU must be free):  rocm/run.sh python rocm/numerics.py
Prints the loss and per-leaf gradient error of the GPU against the CPU, in f32 and in the bf16 mixed
precision that train.py uses, so a backend difference can be traced to a specific parameter group.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
import deepseek_v3 as ds  # noqa: E402
from train import get_batch  # noqa: E402

cfg = ds.Config()
cpu, gpu = jax.devices("cpu")[0], jax.devices()[0]
train = np.memmap("data/train.bin", np.uint16, mode="r")
batch = get_batch(train, cfg.batch, cfg.seq_len + 1 + cfg.n_mtp, np.random.default_rng(0))
params = ds.init_params(cfg, jax.random.key(0))
biases = jnp.zeros((cfg.n_layers - cfg.n_dense, cfg.n_routed))


def loss_and_grad(dtype):
    cast = lambda p: jax.tree.map(lambda a: a.astype(dtype), p)  # same as make_train_step
    return jax.jit(jax.value_and_grad(lambda p, b, x: ds.loss_fn(cast(p), b, x, cfg)[0]))


def run(dtype, device):
    put = lambda t: jax.device_put(t, device)
    loss, g = loss_and_grad(dtype)(put(params), put(biases), put(batch))
    return float(loss), jax.tree.map(np.asarray, g)


def report(label, dtype):
    lc, gc = run(dtype, cpu)
    lg, gg = run(dtype, gpu)
    print(f"\n== {label}: loss cpu {lc:.5f}  gpu {lg:.5f}  diff {lg - lc:+.5f}")
    rows = []
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(gc)[0], jax.tree.leaves(gg)):
        a, b = a.astype(np.float64), b.astype(np.float64)
        rel = np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-30)
        cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        rows.append((rel, cos, jax.tree_util.keystr(path)))
    for rel, cos, name in sorted(rows, reverse=True)[:8]:
        print(f"  grad rel err {rel:8.2e}  cos {cos:.5f}  {name}")
    print(f"  median grad rel err {np.median([r[0] for r in rows]):.2e} over {len(rows)} leaves")


def vs_reference():
    """Which backend's bf16 step is closer to the exact (f32) gradient?"""
    _, ref = run(jnp.float32, cpu)
    print("\n== bf16 gradient error vs the f32 reference (lower is better)")
    print(f"  {'group':<34} {'cpu bf16':>9} {'gpu bf16':>9}")
    groups = {}
    for d, label in ((cpu, "cpu"), (gpu, "gpu")):
        _, g = run(jnp.bfloat16, d)
        for (path, r), x in zip(jax.tree_util.tree_flatten_with_path(ref)[0], jax.tree.leaves(g)):
            key = "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path[-2:])
            num, den = groups.setdefault(key, {}).get(label, (0.0, 0.0))
            r64, x64 = r.astype(np.float64), x.astype(np.float64)
            groups[key][label] = (num + ((x64 - r64) ** 2).sum(), den + (r64 ** 2).sum())
    rel = lambda nd: (nd[0] / nd[1]) ** 0.5
    for key, v in sorted(groups.items(), key=lambda kv: -rel(kv[1]["gpu"]))[:12]:
        print(f"  {key:<34} {rel(v['cpu']):9.2e} {rel(v['gpu']):9.2e}")


print("devices:", cpu, gpu)
report("float32", jnp.float32)
with jax.default_matmul_precision("highest"):
    report("float32, matmul precision=highest", jnp.float32)
report("bfloat16 (train.py mixed precision)", jnp.bfloat16)
vs_reference()
