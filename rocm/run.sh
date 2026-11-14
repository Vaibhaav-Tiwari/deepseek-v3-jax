#!/usr/bin/env bash
# Run a command in the JAX-ROCm container with the GPU and this repo mounted at /work.
#   rocm/run.sh python train.py ...        (interactive)
#   DETACH=1 NAME=dsv3-main rocm/run.sh python train.py --out runs/main   (background)
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
mode=(--rm); [[ -t 0 && -t 1 ]] && mode+=(-it)
[[ "${DETACH:-}" == 1 ]] && mode=(-d --name "${NAME:-dsv3}")
exec podman run "${mode[@]}" \
  --device /dev/kfd --device /dev/dri \
  --security-opt label=disable --group-add keep-groups --ipc=host \
  -v "$repo":/work -w /work \
  -e PYTHONUNBUFFERED=1 ${HSA_OVERRIDE_GFX_VERSION:+-e HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION} \
  localhost/dsv3-jax-rocm "$@"
