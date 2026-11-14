"""Does JAX on Apple's Metal backend support everything DeepSeek V3 needs?

Run:  .venv/bin/python smoke_metal.py
Each check prints PASS or FAIL with the error, so you can see what your Mac's jax-metal supports.
"""
import time, traceback
import numpy as np
import jax, jax.numpy as jnp

cpu = jax.devices("cpu")[0]
print("devices:", jax.devices(), "| default backend:", jax.default_backend())

def check(name, fn):
    try:
        msg = fn()
        print(f"PASS  {name}" + (f"  ({msg})" if msg else ""))
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")

def close(a, b, tol):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    err = np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
    assert err < tol, f"rel err {err:.2e}"
    return f"rel err {err:.1e}"

rng = np.random.default_rng(0)
A, B = rng.standard_normal((2048, 2048), np.float32), rng.standard_normal((2048, 2048), np.float32)
x, W, W2 = (rng.standard_normal(s, np.float32) for s in [(64, 32), (32, 16), (32, 16)])
f = lambda x, W, W2: jnp.sum(jax.nn.sigmoid(x @ W) * (x @ W2))

check("matmul f32", lambda: close(jnp.asarray(A) @ jnp.asarray(B), A @ B, 1e-3))
check("matmul bf16", lambda: close(jnp.asarray(A, jnp.bfloat16) @ jnp.asarray(B, jnp.bfloat16), A @ B, 2e-2))
check("lax.top_k", lambda: close(jax.lax.top_k(jnp.asarray(x[:, :16]), 4)[0], np.sort(x[:, :16], -1)[:, ::-1][:, :4], 1e-6))
check("take_along_axis", lambda: close(jnp.take_along_axis(jnp.asarray(x), jnp.zeros((64, 3), jnp.int32), 1), x[:, [0, 0, 0]], 1e-6))
check(".at[].set scatter", lambda: close(jnp.zeros((64, 16)).at[jnp.arange(64)[:, None], jnp.zeros((64, 2), jnp.int32)].set(1.0).sum(), 64.0, 1e-6))
check("dynamic_update_slice", lambda: close(jax.lax.dynamic_update_slice(jnp.zeros((2, 8, 4)), jnp.ones((2, 1, 4)), (0, 3, 0)).sum(), 8.0, 1e-6))
check("grad sigmoid-gated", lambda: close(jax.grad(f, (1, 2))(x, W, W2)[0], jax.jit(jax.grad(f, (1, 2)), device=cpu)(x, W, W2)[0], 1e-4))
check("softmax + causal mask", lambda: close(jax.nn.softmax(jnp.where(jnp.tril(jnp.ones((8, 8), bool)), 1.0, -jnp.inf), -1).sum(), 8.0, 1e-5))

def deep_mlp():
    Ws = [jnp.asarray(rng.standard_normal((256, 256), np.float32) * 0.05) for _ in range(6)]
    def net(Ws, h):
        for w in Ws: h = jax.nn.silu(jnp.einsum("bd,de->be", h, w))
        return h.sum()
    g = jax.jit(jax.grad(net))(Ws, jnp.ones((32, 256)))
    return f"grad norm {float(jnp.linalg.norm(g[0])):.3f}"
check("jit grad 6-layer einsum MLP", deep_mlp)

def two_axis_contraction():
    # grad of einsum("bthd,bsd->bhts") w.r.t. the head-less operand contracts over 2 axes at once
    q, k = jnp.ones((2, 8, 3, 4)), jnp.ones((2, 8, 4))
    g = jax.grad(lambda k: jnp.einsum("bthd,bsd->bhts", q, k).sum())(k)
    return close(g, np.full((2, 8, 4), 24.0), 1e-6)
check("grad through 2-axis contraction", two_axis_contraction)

def ragged():
    lhs, rhs = jnp.ones((12, 8)), jnp.ones((3, 8, 4))
    out = jax.lax.ragged_dot(lhs, rhs, jnp.array([4, 4, 4], jnp.int32))
    return close(out, np.full((12, 4), 8.0), 1e-6)
check("lax.ragged_dot", ragged)

def tflops(dtype):
    a, b = jnp.asarray(A, dtype), jnp.asarray(B, dtype)
    mm = jax.jit(lambda a, b: a @ b); mm(a, b).block_until_ready()
    t = time.perf_counter(); n = 20
    for _ in range(n): out = mm(a, b)
    out.block_until_ready()
    return f"{2 * 2048**3 * n / (time.perf_counter() - t) / 1e12:.2f} TFLOP/s"
check("throughput f32", lambda: tflops(jnp.float32))
check("throughput bf16", lambda: tflops(jnp.bfloat16))
