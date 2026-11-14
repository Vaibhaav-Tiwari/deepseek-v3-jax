# DeepSeek V3 in pure JAX

DeepSeek V3's architecture (Multi-head Latent Attention, DeepSeekMoE with shared and
routed experts, auxiliary-loss-free load balancing and multi-token prediction) in
**190 lines of pure JAX**, pretrained from scratch on a MacBook Air on public-domain
detective fiction.

No Flax, no Equinox: params are plain dicts of arrays, everything is a function.

**Write-up** with diagrams, code walkthrough and results: [DeepSeek V3 in 190 lines of pure JAX](https://vaibhaav.in/writing/deepseek-v3-jax/)

## Quickstart (Apple Silicon Mac)

```bash
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python smoke_metal.py          # optional: check what jax-metal supports on your Mac
.venv/bin/python prepare_data.py         # download + clean ~84 MB of PD crime fiction, train BPE
.venv/bin/python train.py                # ~2.5 h on an M5 Air (12k steps, bf16 compute)
.venv/bin/python sample.py --prompt "The fog came in off the bay" --num 3
.venv/bin/python plots.py                # figures into post/figures/
```

- `--config small` is a lighter preset for 8–16 GB Macs (fewer layers/experts, shorter
  context). It should work, but it's untested.
- No working jax-metal? Everything also runs on CPU: `JAX_PLATFORMS=cpu .venv/bin/python train.py --config small`
  (checked by hand for 2 steps; the automated test covers the `tiny` config).
- Tests: `JAX_PLATFORMS=cpu .venv/bin/pytest tests.py -v`.

`jax-metal` 0.1.1 needs exactly `jax==0.4.34`; `requirements.txt` pins it.

## Files

| file | what it is | counted in "190 lines"? |
|---|---|---|
| `deepseek_v3.py` | config, MLA, MoE, balancing, MTP, loss, train step, cached sampling | **yes** |
| `train.py` | CLI training loop, logging, checkpoints | no |
| `prepare_data.py`, `sources.txt` | corpus download/cleaning, tokenizer | no |
| `sample.py`, `plots.py` | generation CLI, figures | no |
| `tests.py` | 39 correctness checks | no |
| `smoke_metal.py` | which JAX ops work on Apple's GPU | no |

Line count = non-blank, non-comment lines: `grep -cvE '^\s*(#|$)' deepseek_v3.py`.

## Data and licensing

All training text is **public domain in the United States**: Dashiell Hammett's *Red
Harvest*, *The Dain Curse*, *The Maltese Falcon* and Continental Op stories (Standard
Ebooks CC0 editions), plus ~200 detective and crime novels from Project Gutenberg.
Every source is pinned in `sources.txt`. Public-domain status outside the US may differ.

## What's simplified (see the post for details)

- MoE runs every expert on every token and zeroes the unpicked ones (exact routing math,
  dense compute). No sparse dispatch, no expert parallelism.
- No FP8, DualPipe, node-limited routing, YaRN, or post-training.
- MLA decompresses the latent explicitly instead of using the weight-absorption trick.
