"""Correctness checks. Run: JAX_PLATFORMS=cpu .venv/bin/pytest tests.py -v"""
import io, os, sys, subprocess, tarfile
import numpy as np
import optax
import pytest
import jax, jax.numpy as jnp
from deepseek_v3 import Config, rms_norm, rope, apply_rope, swiglu, init_params, n_params, init_mla, init_cache, mla, init_moe, route, moe, update_bias, seq_balance_loss, forward, loss_fn, TINY, make_train_step, generate
import deepseek_v3
import train
from prepare_data import strip_gutenberg, strip_xhtml, clean_text, split_docs, standard_ebook



# ---------------------------------------------------------------- data
def test_strip_gutenberg_removes_boilerplate():
    t = "junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nSam Spade.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense"
    assert strip_gutenberg(t, "x").strip() == "Sam Spade."

def test_strip_gutenberg_requires_markers():
    with pytest.raises(ValueError, match="mybook"):
        strip_gutenberg("no markers here", "mybook")

def test_strip_xhtml_keeps_paragraph_text():
    assert strip_xhtml("<html><body><p>The Op.</p><p>Poisonville.</p></body></html>") == "The Op.\n\nPoisonville."

def test_clean_text_unwraps_paragraphs():
    t = "Spade had a long,\nbony jaw.\n\n\n\n_She_ came in.\n[Illustration: a gun]\n"
    assert clean_text(t) == "Spade had a long, bony jaw.\n\nShe came in."

def test_split_docs_is_disjoint_and_deterministic():
    docs = [f"d{i}" for i in range(100)]
    a, b = split_docs(docs); a2, b2 = split_docs(docs)
    assert (a, b) == (a2, b2) and not set(a) & set(b) and len(b) >= 1

def test_clean_text_drops_credits_and_contents_and_normalizes_quotes():
    t = ("Produced by Jane Doe and the Online Distributed Proofreading Team\n\nCONTENTS\n\nI. THE BODY\n\n"
         "II. THE GUN\n\n\u201cDrop it,\u201d she said. It\u2019s late.Ver.04.29.93*END*\n\nEnd of Project Gutenberg's X, by Y")
    assert clean_text(t) == '"Drop it," she said. It\'s late.'

def _tgz(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode(); info = tarfile.TarInfo(f"repo-HEAD/src/epub/{name}"); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()

def test_standard_ebook_reads_chapters_in_spine_order():
    opf = ('<manifest><item href="text/chapter-2.xhtml" id="chapter-2.xhtml" media-type="x"/>'
           '<item href="text/chapter-1.xhtml" id="chapter-1.xhtml" media-type="x"/>'
           '<item href="text/colophon.xhtml" id="colophon.xhtml" media-type="x"/></manifest>'
           '<spine><itemref idref="chapter-1.xhtml"/><itemref idref="chapter-2.xhtml"/><itemref idref="colophon.xhtml"/></spine>')
    tgz = _tgz({"content.opf": opf, "text/chapter-1.xhtml": "<body><p>One.</p></body>",
                "text/chapter-2.xhtml": "<body><p>Two \u201cquoted\u201d.</p></body>", "text/colophon.xhtml": "<body><p>Credits</p></body>"})
    assert standard_ebook(tgz) == 'One.\n\nTwo "quoted".'


# ---------------------------------------------------------------- basics
def test_rms_norm_unit_rms():
    x = jax.random.normal(jax.random.key(0), (3, 32)) * 5
    y = rms_norm(x, jnp.ones(32))
    assert jnp.allclose(jnp.sqrt((y**2).mean(-1)), 1.0, atol=1e-3)

def test_rope_is_relative():
    cos, sin = rope(TINY, jnp.arange(16))
    q = jax.random.normal(jax.random.key(1), (TINY.d_rope,)); k = jax.random.normal(jax.random.key(2), (TINY.d_rope,))
    # the same q,k placed at positions (3,1) and (10,8) give the same dot product
    qq = apply_rope(jnp.tile(q, (16, 1))[None], cos, sin)[0]
    kk = apply_rope(jnp.tile(k, (16, 1))[None], cos, sin)[0]
    assert jnp.allclose(qq[3] @ kk[1], qq[10] @ kk[8], atol=1e-4)
    assert not jnp.allclose(qq[3] @ kk[1], qq[3] @ kk[3], atol=1e-4)

def test_rope_preserves_norm_and_handles_heads():
    cos, sin = rope(TINY, jnp.arange(16))
    x = jax.random.normal(jax.random.key(3), (2, 16, 3, TINY.d_rope))     # [B,T,H,d_rope]
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    assert jnp.allclose(jnp.linalg.norm(y, axis=-1), jnp.linalg.norm(x, axis=-1), atol=1e-4)

def test_swiglu_shape():
    p = {"w1": jnp.ones((32, 64)), "w3": jnp.ones((32, 64)), "w2": jnp.ones((64, 32))}
    assert swiglu(p, jnp.ones((2, 5, 32))).shape == (2, 5, 32)

def test_init_params_structure():
    p = init_params(TINY, jax.random.key(0))
    assert p["embed"].shape == (64, 32) and p["head"].shape == (32, 64) and p["norm_f"].shape == (32,)
    assert len(p["layers"]) == 2 and "ffn" in p["layers"][0] and "moe" in p["layers"][1]
    assert len(p["mtp"]) == 1 and p["mtp"][0]["proj"].shape == (64, 32) and "ffn" in p["mtp"][0]["block"]
    assert n_params(p) > 0


# ---------------------------------------------------------------- MLA
def test_mla_shapes():
    p = init_mla(TINY, jax.random.key(0)); x = jnp.ones((2, 16, TINY.d_model))
    cos, sin = rope(TINY, jnp.arange(16))
    out, _ = mla(p, x, cos, sin, TINY)
    assert out.shape == (2, 16, TINY.d_model)

def test_mla_is_causal():
    p = init_mla(TINY, jax.random.key(0)); cos, sin = rope(TINY, jnp.arange(16))
    x = jax.random.normal(jax.random.key(1), (1, 16, TINY.d_model))
    x2 = x.at[:, 10:].set(0.0)
    assert jnp.allclose(mla(p, x, cos, sin, TINY)[0][:, :10], mla(p, x2, cos, sin, TINY)[0][:, :10], atol=1e-5)

def test_mla_cache_matches_full_forward():
    p = init_mla(TINY, jax.random.key(0)); x = jax.random.normal(jax.random.key(1), (2, 16, TINY.d_model))
    cos, sin = rope(TINY, jnp.arange(16)); full, _ = mla(p, x, cos, sin, TINY)
    cache = init_cache(TINY, 2, 1)[0]
    out0, cache = mla(p, x[:, :10], cos[:10], sin[:10], TINY, cache, pos=0)
    steps = [out0]
    for t in range(10, 16):
        o, cache = mla(p, x[:, t:t+1], cos[t:t+1], sin[t:t+1], TINY, cache, pos=t); steps.append(o)
    assert jnp.allclose(jnp.concatenate(steps, 1), full, atol=1e-4)

def test_mla_cache_size():
    c = Config()
    cache = init_cache(c, 1, 1)[0]
    mla_floats = cache["c"].shape[-1] + cache["kr"].shape[-1]
    naive_floats = c.n_heads * ((c.d_nope + c.d_rope) + c.d_v)   # full K and V per token per layer
    assert (mla_floats, naive_floats) == (144, 480)


# ---------------------------------------------------------------- MoE
def test_route_selects_top_k_and_normalizes():
    w = jax.random.normal(jax.random.key(0), (TINY.d_model, TINY.n_routed)); x = jax.random.normal(jax.random.key(1), (8, TINY.d_model))
    gates, scores = route(w, jnp.zeros(TINY.n_routed), x, TINY)
    assert ((gates > 0).sum(-1) == TINY.top_k).all()
    assert jnp.allclose(gates.sum(-1), TINY.route_scale, atol=1e-5)
    assert ((scores > 0) & (scores < 1)).all()

def test_moe_with_identical_experts_equals_dense():
    p = init_moe(TINY, jax.random.key(0))
    one = jax.tree.map(lambda a: a[0], p["experts"])
    p["experts"] = jax.tree.map(lambda a: jnp.broadcast_to(a, (TINY.n_routed,) + a.shape), one)
    x = jax.random.normal(jax.random.key(1), (2, 16, TINY.d_model))
    out, _ = moe(p, jnp.zeros(TINY.n_routed), x, TINY)
    assert jnp.allclose(out, TINY.route_scale * swiglu(one, x) + swiglu(p["shared"], x), atol=1e-4)

def test_moe_load_sums_to_one():
    p = init_moe(TINY, jax.random.key(0)); x = jax.random.normal(jax.random.key(1), (2, 16, TINY.d_model))
    _, aux = moe(p, jnp.zeros(TINY.n_routed), x, TINY)
    assert aux["load"].shape == (TINY.n_routed,) and jnp.allclose(aux["load"].sum(), 1.0)


# ---------------------------------------------------------------- aux-loss-free balancing
def test_bias_changes_selection_not_gate_values():
    w = jax.random.normal(jax.random.key(0), (TINY.d_model, TINY.n_routed)); x = jax.random.normal(jax.random.key(1), (8, TINY.d_model))
    _, s = route(w, jnp.zeros(TINY.n_routed), x, TINY)
    g1, _ = route(w, jnp.zeros(TINY.n_routed).at[0].set(100.0), x, TINY)
    assert (g1[:, 0] > 0).all()                               # expert 0 forced in
    sel = g1 > 0
    expect = jnp.where(sel, s, 0) / jnp.where(sel, s, 0).sum(-1, keepdims=True) * TINY.route_scale
    assert jnp.allclose(g1, expect, atol=1e-5)                # gates use raw scores only

def test_update_bias_pushes_toward_balance():
    b = update_bias(jnp.zeros((1, 4)), jnp.array([[0.7, 0.1, 0.1, 0.1]]), 1e-3)
    assert b[0, 0] < 0 and (b[0, 1:] > 0).all()

def test_seq_loss_minimal_when_balanced():
    # uniform scores + perfectly balanced selection => f_i = 1, P_i = 1/E => sum_i f_i P_i = 1
    E, K, T = TINY.n_routed, TINY.top_k, 8
    scores = jnp.ones((1, T, E))
    selected = (jnp.arange(E)[None, :] - K * jnp.arange(T)[:, None]) % E < K   # each expert chosen T*K/E times
    assert jnp.allclose(seq_balance_loss(scores, selected[None], TINY), TINY.alpha_seq, atol=1e-6)

def test_seq_loss_higher_when_collapsed():
    E, K, T = TINY.n_routed, TINY.top_k, 8
    scores = jnp.ones((1, T, E)).at[..., :K].set(10.0)
    selected = jnp.zeros((1, T, E), bool).at[..., :K].set(True)   # every token picks the same K experts
    assert seq_balance_loss(scores, selected, TINY) > 1.5 * TINY.alpha_seq


# ---------------------------------------------------------------- forward, MTP, loss
def _setup():
    p = init_params(TINY, jax.random.key(0)); b = jnp.zeros((TINY.n_layers - TINY.n_dense, TINY.n_routed))
    return p, b

def test_forward_shapes_and_aux():
    p, b = _setup()
    tok = jax.random.randint(jax.random.key(1), (2, 16), 0, TINY.vocab)
    logits, h, aux, _ = forward(p, b, tok, TINY)
    assert logits.shape == (2, 16, TINY.vocab) and h.shape == (2, 16, TINY.d_model)
    assert aux["load"].shape == b.shape

def test_initial_loss_near_uniform():
    p, b = _setup()
    tok = jax.random.randint(jax.random.key(1), (2, 18), 0, TINY.vocab)
    _, m = loss_fn(p, b, tok, TINY)
    assert abs(m["main"] - jnp.log(TINY.vocab)) < 0.3 and abs(m["mtp"] - jnp.log(TINY.vocab)) < 0.3

def test_mtp_gets_gradient_into_shared_head():
    p, b = _setup()
    tok = jax.random.randint(jax.random.key(1), (2, 18), 0, TINY.vocab)
    g = jax.grad(lambda q: loss_fn(q, b, tok, TINY)[1]["mtp"])(p)
    assert jnp.abs(g["head"]).sum() > 0 and jnp.abs(g["mtp"][0]["proj"]).sum() > 0

def test_mtp_predicts_two_ahead():
    # batch holds T+2 tokens: main targets stop at index T, only MTP sees the very last token (t+2)
    p, b = _setup()
    tok = jax.random.randint(jax.random.key(1), (2, 18), 0, TINY.vocab)
    tok2 = tok.at[:, -1].set((tok[:, -1] + 1) % TINY.vocab)
    _, m1 = loss_fn(p, b, tok, TINY); _, m2 = loss_fn(p, b, tok2, TINY)
    assert jnp.allclose(m1["main"], m2["main"]) and not jnp.allclose(m1["mtp"], m2["mtp"])


# ---------------------------------------------------------------- training
def test_overfit_one_batch():
    p, b = _setup()
    opt = optax.adam(3e-3); s = opt.init(p); step = make_train_step(TINY, opt)
    batch = jax.random.randint(jax.random.key(1), (2, 18), 0, TINY.vocab)
    for _ in range(300): p, s, b, m = step(p, s, b, batch)
    assert m["main"] < 0.1

def test_bias_moves_during_training():
    p, b = _setup()
    opt = optax.adam(3e-3); s = opt.init(p); step = make_train_step(TINY, opt)
    batch = jax.random.randint(jax.random.key(1), (2, 18), 0, TINY.vocab)
    for _ in range(20): p, s, b, m = step(p, s, b, batch)
    assert jnp.abs(b).sum() > 0

def test_checkpoint_roundtrip_restores_config(tmp_path):
    p = init_params(TINY, jax.random.key(0)); b = jnp.ones((1, TINY.n_routed))
    train.save_checkpoint(tmp_path / "c.pkl", p, b, TINY); p2, b2, cfg2 = train.load_checkpoint(tmp_path / "c.pkl")
    assert cfg2 == TINY and jnp.allclose(b2, b) and jnp.allclose(p2["embed"], p["embed"])

def write_fake_data(d):
    for n in ("train.bin", "val.bin"):
        np.random.default_rng(0).integers(0, TINY.vocab, 4096).astype(np.uint16).tofile(d / n)

def test_train_stops_on_nan(tmp_path, monkeypatch):
    write_fake_data(tmp_path)
    real = deepseek_v3.make_train_step
    def nan_after_3(cfg, opt):
        step, n = real(cfg, opt), [0]
        def wrapped(*a):
            n[0] += 1; p, s, b, m = step(*a)
            return p, s, b, {**m, "loss": jnp.nan if n[0] >= 3 else m["loss"]}
        return wrapped
    monkeypatch.setattr(deepseek_v3, "make_train_step", nan_after_3)
    with pytest.raises(FloatingPointError, match="step 3"):
        train.main(["--config", "tiny", "--steps", "10", "--log-every", "1", "--eval-every", "100",
                    "--data", str(tmp_path), "--out", str(tmp_path / "run")])
    assert not (tmp_path / "run" / "ckpt.pkl").exists()

def test_train_cli_cpu_smoke(tmp_path):
    write_fake_data(tmp_path)
    r = subprocess.run([sys.executable, "train.py", "--config", "tiny", "--steps", "2", "--log-every", "1",
                        "--data", str(tmp_path), "--out", str(tmp_path / "run")],
                       env={**os.environ, "JAX_PLATFORMS": "cpu"}, capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    assert r.returncode == 0, r.stderr
    assert len((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()) >= 1
    assert (tmp_path / "run" / "ckpt.pkl").exists()

def test_bf16_step_keeps_f32_master_weights():
    import dataclasses; cfg = dataclasses.replace(TINY, dtype="bfloat16")
    p, b = _setup()
    opt = optax.adam(1e-3); s = opt.init(p); step = make_train_step(cfg, opt)
    batch = jax.random.randint(jax.random.key(1), (2, 18), 0, TINY.vocab)
    p2, _, _, m = step(p, s, b, batch)
    assert p2["embed"].dtype == jnp.float32 and jnp.isfinite(m["loss"])


# ---------------------------------------------------------------- generation
def test_generate_matches_greedy_full_forward():
    p, b = _setup()
    prompt = jnp.array([[1, 2, 3]])
    out = generate(p, b, prompt, 5, TINY, jax.random.key(0), temperature=0.0)
    seq = prompt
    for _ in range(5):
        logits, _, _, _ = forward(p, b, seq, TINY); seq = jnp.concatenate([seq, logits[:, -1].argmax(-1)[:, None]], 1)
    assert (out == seq).all()

def test_generate_rejects_overflow():
    p, b = _setup()
    with pytest.raises(ValueError):
        generate(p, b, jnp.ones((1, 10), jnp.int32), TINY.seq_len, TINY, jax.random.key(0))

def test_generate_sampling_respects_top_k():
    p, b = _setup()
    out = generate(p, b, jnp.array([[1, 2, 3]]), 4, TINY, jax.random.key(0), temperature=1.0, top_k=1)
    greedy = generate(p, b, jnp.array([[1, 2, 3]]), 4, TINY, jax.random.key(0), temperature=0.0)
    assert (out == greedy).all()                       # top_k=1 sampling is greedy


# ---------------------------------------------------------------- review fixes
def _nan_from(n_bad, field):
    """make_train_step wrapper: from call n_bad on, poison metrics['loss'] or the returned params."""
    real = deepseek_v3.make_train_step
    def make(cfg, opt):
        step, n = real(cfg, opt), [0]
        def wrapped(*a):
            n[0] += 1; p, s, b, m = step(*a)
            if n[0] >= n_bad:
                if field == "loss": m = {**m, "loss": jnp.nan}
                else: p = jax.tree.map(lambda x: x * jnp.nan, p)
            return p, s, b, m
        return wrapped
    return make

def test_nan_stop_names_exact_step_between_logs(tmp_path, monkeypatch):
    write_fake_data(tmp_path)
    monkeypatch.setattr(deepseek_v3, "make_train_step", _nan_from(3, "loss"))
    with pytest.raises(FloatingPointError, match="step 3"):
        train.main(["--config", "tiny", "--steps", "10", "--log-every", "25", "--eval-every", "100",
                    "--data", str(tmp_path), "--out", str(tmp_path / "run")])

def test_nan_params_never_overwrite_good_checkpoint(tmp_path, monkeypatch):
    write_fake_data(tmp_path)
    train.main(["--config", "tiny", "--steps", "2", "--log-every", "1", "--eval-every", "2",
                "--data", str(tmp_path), "--out", str(tmp_path / "run")])
    good = (tmp_path / "run" / "ckpt.pkl").read_bytes()
    monkeypatch.setattr(deepseek_v3, "make_train_step", _nan_from(2, "params"))
    with pytest.raises(FloatingPointError):
        train.main(["--config", "tiny", "--steps", "4", "--log-every", "1", "--eval-every", "2",
                    "--data", str(tmp_path), "--out", str(tmp_path / "run")])
    assert (tmp_path / "run" / "ckpt.pkl").read_bytes() == good

def test_router_scores_in_f32_under_bf16():
    w = jax.random.normal(jax.random.key(0), (TINY.d_model, TINY.n_routed)).astype(jnp.bfloat16)
    x = jax.random.normal(jax.random.key(1), (8, TINY.d_model)).astype(jnp.bfloat16)
    gates, scores = route(w, jnp.zeros(TINY.n_routed), x, TINY)
    assert scores.dtype == jnp.float32 and gates.dtype == jnp.bfloat16
