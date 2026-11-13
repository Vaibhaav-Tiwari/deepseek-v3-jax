"""DeepSeek V3 in pure JAX: MLA, DeepSeekMoE, auxiliary-loss-free balancing, multi-token prediction.

Params are plain nested dicts of arrays. Shapes: B batch, T time, d d_model, H heads, E routed experts.
"""
from dataclasses import dataclass
import jax, jax.numpy as jnp
import optax


@dataclass(frozen=True)
class Config:
    vocab: int = 8192
    d_model: int = 384
    n_layers: int = 6
    n_dense: int = 1           # first layers use a dense FFN instead of MoE
    n_heads: int = 6
    q_rank: int = 192          # MLA: low-rank query compression
    kv_rank: int = 128         # MLA: low-rank joint key/value compression (this is what gets cached)
    d_nope: int = 32           # per-head query/key dims without RoPE
    d_rope: int = 16           # per-head query/key dims with RoPE (key part shared across heads)
    d_v: int = 32              # per-head value dim
    d_ff: int = 1024           # dense FFN hidden size
    n_shared: int = 1          # MoE: always-on shared experts
    n_routed: int = 16         # MoE: routed experts
    top_k: int = 4             # MoE: routed experts per token
    d_expert: int = 256        # MoE: hidden size of each expert
    route_scale: float = 1.0   # MoE: scale on the normalized gate weights
    gamma: float = 1e-3        # bias update speed for aux-loss-free balancing
    alpha_seq: float = 1e-4    # weight of the complementary sequence-wise balance loss
    n_mtp: int = 1             # multi-token prediction depth
    mtp_lambda: float = 0.3    # weight of the MTP loss
    seq_len: int = 256
    batch: int = 32
    rope_theta: float = 10000.0
    init_std: float = 0.02
    dtype: str = "bfloat16"    # compute dtype; weights and optimizer state stay float32


SMALL = Config(n_layers=4, n_routed=8, top_k=2, seq_len=128, batch=16)
TINY = Config(vocab=64, d_model=32, n_layers=2, n_heads=2, q_rank=16, kv_rank=8, d_nope=8, d_rope=4, d_v=8,
              d_ff=64, n_routed=4, top_k=2, d_expert=16, seq_len=16, batch=2, dtype="float32")   # tests


# ---------------------------------------------------------------- building blocks
def rms_norm(x, w, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * w


def rope(cfg, positions):
    freqs = cfg.rope_theta ** -(jnp.arange(0, cfg.d_rope, 2) / cfg.d_rope)    # [d_rope/2]
    angles = positions[:, None] * freqs[None, :]                               # [T, d_rope/2]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    # x: [B, T, ..., d_rope]; rotate each (first-half, second-half) pair by its position's angle
    cos, sin = (a.reshape(a.shape[0], *[1] * (x.ndim - 3), a.shape[1]) for a in (cos, sin))
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


# ---------------------------------------------------------------- Multi-head Latent Attention
def init_cache(cfg, B, n_layers):
    # MLA caches only the compressed latent c and the shared RoPE key, not per-head K and V
    return [{"c": jnp.zeros((B, cfg.seq_len, cfg.kv_rank)), "kr": jnp.zeros((B, cfg.seq_len, cfg.d_rope))}
            for _ in range(n_layers)]


def mla(p, x, cos, sin, cfg, cache=None, pos=0):
    B, T, _ = x.shape
    H, dn, dr, dv = cfg.n_heads, cfg.d_nope, cfg.d_rope, cfg.d_v
    # queries: d -> q_rank latent -> H heads of (nope | rope)
    q = (rms_norm(x @ p["wq_a"], p["q_norm"]) @ p["wq_b"]).reshape(B, T, H, dn + dr)   # [B,T,H,dn+dr]
    q_nope, q_rope = q[..., :dn], apply_rope(q[..., dn:], cos, sin)
    # keys/values: d -> kv_rank latent c (cached) + one RoPE key shared by all heads (cached)
    kv_a = x @ p["wkv_a"]                                                               # [B,T,kv_rank+dr]
    c, kr = rms_norm(kv_a[..., :cfg.kv_rank], p["kv_norm"]), apply_rope(kv_a[..., cfg.kv_rank:], cos, sin)
    if cache is not None:                                    # write this chunk at pos, attend over the cache
        cache = {"c": jax.lax.dynamic_update_slice(cache["c"], c, (0, pos, 0)),
                 "kr": jax.lax.dynamic_update_slice(cache["kr"], kr, (0, pos, 0))}
        c, kr = cache["c"], cache["kr"]                                                 # [B,S,kv_rank], [B,S,dr]
    S = c.shape[1]
    kv = (c @ p["wkv_b"]).reshape(B, S, H, dn + dv)          # decompress latent into per-head K_nope and V
    k_nope, v = kv[..., :dn], kv[..., dn:]                                              # [B,S,H,dn], [B,S,H,dv]
    # every head sees the same RoPE key; score = content (nope) part + position (rope) part
    q = jnp.concatenate([q_nope, q_rope], -1)                                           # [B,T,H,dn+dr]
    k = jnp.concatenate([k_nope, jnp.broadcast_to(kr[:, :, None], (B, S, H, dr))], -1)  # [B,S,H,dn+dr]
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(dn + dr)                    # [B,H,T,S]
    causal = jnp.arange(S)[None, :] <= (pos + jnp.arange(T))[:, None]                   # [T,S]
    attn = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf).astype(jnp.float32), -1).astype(x.dtype)
    out = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(B, T, H * dv)                  # [B,T,H*dv]
    return out @ p["wo"], cache


def swiglu(p, x):
    return (jax.nn.silu(x @ p["w1"]) * (x @ p["w3"])) @ p["w2"]            # [.., d] -> [.., h] -> [.., d]


# ---------------------------------------------------------------- DeepSeekMoE
def route(w_router, bias, x, cfg):
    scores = jax.nn.sigmoid(x.astype(jnp.float32) @ w_router.astype(jnp.float32))   # [N,E] affinity, in f32
    _, idx = jax.lax.top_k(scores + jax.lax.stop_gradient(bias), cfg.top_k)   # bias decides WHO is picked...
    picked = jax.nn.one_hot(idx, cfg.n_routed).sum(-2) > 0              # [N,E]
    gated = jnp.where(picked, scores, 0.0)                              # ...but gate values use raw scores
    return (gated / gated.sum(-1, keepdims=True) * cfg.route_scale).astype(x.dtype), scores


def seq_balance_loss(scores, picked, cfg):
    # DeepSeek's tiny complementary loss: discourage any single sequence from piling onto few experts
    E, K, T = cfg.n_routed, cfg.top_k, scores.shape[1]
    f = picked.sum(1) * E / (K * T)                                     # [B,E] load, 1.0 = perfectly even
    P = (scores / scores.sum(-1, keepdims=True)).mean(1)                # [B,E] mean normalized affinity
    return cfg.alpha_seq * (f * P).sum(-1).mean()


def update_bias(biases, loads, gamma):
    # aux-loss-free balancing: after each step, nudge overloaded experts down and underloaded ones up
    return biases + gamma * jnp.sign(loads.mean(-1, keepdims=True) - loads)


def moe(p, bias, x, cfg):
    B, T, d = x.shape
    xf = x.reshape(B * T, d)
    gates, scores = route(p["router"], bias, xf, cfg)                   # [N,E], [N,E]
    w = p["experts"]   # every expert runs on every token; unpicked experts get gate 0 (see post)
    h = jax.nn.silu(jnp.einsum("nd,edh->neh", xf, w["w1"])) * jnp.einsum("nd,edh->neh", xf, w["w3"])
    routed = jnp.einsum("ne,ned->nd", gates, jnp.einsum("neh,ehd->ned", h, w["w2"]))   # [N,d]
    out = routed + swiglu(p["shared"], xf)
    load = (gates > 0).mean(0) / cfg.top_k                              # [E] share of routed slots
    seq = seq_balance_loss(scores.reshape(B, T, -1), (gates > 0).reshape(B, T, -1), cfg)
    return out.reshape(B, T, d), {"load": load, "seq_loss": seq}


# ---------------------------------------------------------------- model
def block(p, bias, x, cos, sin, cfg, cache=None, pos=0):
    a, cache = mla(p["attn"], rms_norm(x, p["norm_attn"]), cos, sin, cfg, cache, pos)
    x = x + a
    h = rms_norm(x, p["norm_ffn"])
    f, aux = moe(p["moe"], bias, h, cfg) if "moe" in p else (swiglu(p["ffn"], h), None)
    return x + f, aux, cache


def forward(params, biases, tokens, cfg, caches=None, pos=0):
    cos, sin = rope(cfg, pos + jnp.arange(tokens.shape[1]))
    x, auxes, new_caches = params["embed"][tokens], [], []                # [B,T,d]
    for i, p in enumerate(params["layers"]):
        bias = biases[i - cfg.n_dense] if i >= cfg.n_dense else None     # one bias vector per MoE layer
        x, aux, c = block(p, bias, x, cos, sin, cfg, None if caches is None else caches[i], pos)
        new_caches.append(c)
        if aux is not None: auxes.append(aux)
    logits = rms_norm(x, params["norm_f"]) @ params["head"]               # [B,T,V]
    aux = {"load": jnp.stack([a["load"] for a in auxes]), "seq_loss": sum(a["seq_loss"] for a in auxes)}
    return logits, x, aux, new_caches


def cross_entropy(logits, targets):
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    return -jnp.take_along_axis(logp, targets[..., None], -1).mean()


def loss_fn(params, biases, tokens, cfg):
    # tokens: [B, T + 1 + n_mtp]. Main model predicts t+1; MTP module k predicts t+1+k.
    T = tokens.shape[1] - 1 - cfg.n_mtp
    logits, h, aux, _ = forward(params, biases, tokens[:, :T], cfg)
    main = cross_entropy(logits, tokens[:, 1:T + 1])
    cos, sin = rope(cfg, jnp.arange(T))
    mtp = 0.0
    for k, m in enumerate(params["mtp"], start=1):
        # combine depth k-1's hidden state at i with the embedding of token i+k, then one more block
        e = params["embed"][tokens[:, k:T + k]]                           # shared embedding
        h = jnp.concatenate([rms_norm(h, m["norm_h"]), rms_norm(e, m["norm_e"])], -1) @ m["proj"]   # [B,T,d]
        h, _, _ = block(m["block"], None, h, cos, sin, cfg)
        mtp += cross_entropy(rms_norm(h, params["norm_f"]) @ params["head"], tokens[:, k + 1:T + k + 1])  # shared head
    mtp = mtp / cfg.n_mtp
    loss = main + cfg.mtp_lambda * mtp + aux["seq_loss"]
    return loss, {"loss": loss, "main": main, "mtp": mtp, "seq": aux["seq_loss"], "load": aux["load"]}


# ---------------------------------------------------------------- training
def make_train_step(cfg, optimizer):
    cast = lambda p: jax.tree.map(lambda a: a.astype(cfg.dtype), p)      # mixed precision: f32 master weights

    @jax.jit
    def train_step(params, opt_state, biases, batch):
        loss = lambda p: loss_fn(cast(p), biases, batch, cfg)
        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, update_bias(biases, metrics["load"], cfg.gamma), metrics
    return train_step


# ---------------------------------------------------------------- generation (with the MLA latent cache)
_forward = jax.jit(forward, static_argnames="cfg")


def sample_token(logits, key, temperature, top_k):
    if temperature == 0:
        return logits.argmax(-1)
    kth = jax.lax.top_k(logits, top_k)[0][:, -1:]                      # keep only the top_k logits
    return jax.random.categorical(key, jnp.where(logits < kth, -jnp.inf, logits) / temperature)


def generate(params, biases, prompt, n_new, cfg, key, temperature=0.8, top_k=50):
    B, P = prompt.shape
    if P + n_new > cfg.seq_len:
        raise ValueError(f"prompt ({P}) + new tokens ({n_new}) exceeds seq_len ({cfg.seq_len})")
    logits, _, _, caches = _forward(params, biases, prompt, cfg, init_cache(cfg, B, cfg.n_layers), 0)   # prefill
    out = [prompt]
    for i in range(n_new):                                              # then one token at a time
        key, sub = jax.random.split(key)
        out.append(sample_token(logits[:, -1], sub, temperature, top_k)[:, None])
        if i < n_new - 1:
            logits, _, _, caches = _forward(params, biases, out[-1], cfg, caches, P + i)
    return jnp.concatenate(out, 1)


# ---------------------------------------------------------------- init
def normal(key, shape, std):
    return jax.random.normal(key, shape) * std


def init_swiglu(cfg, key, hidden):
    k1, k2, k3 = jax.random.split(key, 3)
    d, s = cfg.d_model, cfg.init_std
    return {"w1": normal(k1, (d, hidden), s), "w3": normal(k2, (d, hidden), s), "w2": normal(k3, (hidden, d), s)}


def init_mla(cfg, key):
    k = jax.random.split(key, 5)
    d, H, s = cfg.d_model, cfg.n_heads, cfg.init_std
    return {"wq_a": normal(k[0], (d, cfg.q_rank), s), "q_norm": jnp.ones(cfg.q_rank),
            "wq_b": normal(k[1], (cfg.q_rank, H * (cfg.d_nope + cfg.d_rope)), s),
            "wkv_a": normal(k[2], (d, cfg.kv_rank + cfg.d_rope), s), "kv_norm": jnp.ones(cfg.kv_rank),
            "wkv_b": normal(k[3], (cfg.kv_rank, H * (cfg.d_nope + cfg.d_v)), s),
            "wo": normal(k[4], (H * cfg.d_v, d), s)}


def init_moe(cfg, key):
    k = jax.random.split(key, 5)
    d, E, h, s = cfg.d_model, cfg.n_routed, cfg.d_expert, cfg.init_std
    return {"router": normal(k[0], (d, E), s), "shared": init_swiglu(cfg, k[1], cfg.n_shared * h),
            "experts": {"w1": normal(k[2], (E, d, h), s), "w3": normal(k[3], (E, d, h), s),
                        "w2": normal(k[4], (E, h, d), s)}}


def init_block(cfg, key, moe):
    k1, k2 = jax.random.split(key)
    ffn = {"moe": init_moe(cfg, k2)} if moe else {"ffn": init_swiglu(cfg, k2, cfg.d_ff)}
    return {"norm_attn": jnp.ones(cfg.d_model), "attn": init_mla(cfg, k1), "norm_ffn": jnp.ones(cfg.d_model), **ffn}


def init_params(cfg, key):
    ke, kh, kl, km = jax.random.split(key, 4)
    d = cfg.d_model
    layers = [init_block(cfg, k, moe=i >= cfg.n_dense) for i, k in enumerate(jax.random.split(kl, cfg.n_layers))]
    mtp = [{"norm_h": jnp.ones(d), "norm_e": jnp.ones(d), "proj": normal(k1, (2 * d, d), cfg.init_std),
            "block": init_block(cfg, k2, moe=False)}
           for k1, k2 in (jax.random.split(k) for k in jax.random.split(km, cfg.n_mtp))]
    return {"embed": normal(ke, (cfg.vocab, d), cfg.init_std), "head": normal(kh, (d, cfg.vocab), cfg.init_std),
            "norm_f": jnp.ones(d), "layers": layers, "mtp": mtp}


def n_params(params):
    return sum(x.size for x in jax.tree.leaves(params))
