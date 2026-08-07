#!/usr/bin/env bash
# Evaluate the UQ baselines from the paper under the identical protocol, so a new
# method can be compared against them on equal footing.
#
# Methods (select with METHODS=, comma-separated):
#   conf         MASt3R's own predicted confidence, used as an uncertainty score
#                -> needs CKPT_CONF (the public MASt3R metric checkpoint)
#   hetero       heteroscedastic Gaussian head (per-pixel variance, single pass)
#                -> needs CKPT_HETERO
#   mc_dropout   MC Dropout, T stochastic passes over one model
#                -> needs MC_CKPT (or falls back to CKPT_CONF), MC_SAMPLES
#   ensemble     Deep Ensembles over K independently trained models
#                -> needs ENSEMBLE_CKPTS (comma-separated list of K checkpoints)
#   ours_niw     Trust3R NIW  -> needs CKPT_NIW
#   ours_nig     Trust3R NIG  -> needs CKPT_NIG
#
# `ours_*` methods accept a `_total` / `_alea` / `_epi` suffix to pick the
# uncertainty readout; the paper uses `_epi`.
#
# Example -- MASt3R confidence baseline on ScanNet++ and TUM:
#   METHODS=conf \
#   CKPT_CONF=checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
#   BENCHMARKS=ScanNetpp,TUM \
#   SCANNETPP_ROOT=... TUM_ROOT=... \
#   OUT_DIR=eval_out/baseline_conf \
#   bash eval/evaluate_baselines.sh
#
# Numbers produced here are directly comparable to the Trust3R rows only when the
# benchmark list, method list and their order match the reproduction run -- see the
# "NOTE ON EXACTNESS" block in reproduce_table1_table2.sh.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/dust3r:$REPO:${PYTHONPATH:-}"

PY="${PY:-python}"

METHODS="${METHODS:-conf}"
BENCHMARKS="${BENCHMARKS:-ScanNetpp,ETH3D,KITTI,TUM}"
OUT_DIR="${OUT_DIR:-$REPO/eval_out/baselines}"

CKPT_CONF="${CKPT_CONF:-$REPO/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
CKPT_HETERO="${CKPT_HETERO:-}"
CKPT_NIW="${CKPT_NIW:-}"
CKPT_NIG="${CKPT_NIG:-}"
MC_CKPT="${MC_CKPT:-}"
ENSEMBLE_CKPTS="${ENSEMBLE_CKPTS:-}"
MC_SAMPLES="${MC_SAMPLES:-16}"
MAX_PAIRS="${MAX_PAIRS:-5000}"

SCANNETPP_ROOT="${SCANNETPP_ROOT:-$REPO/data/scannetpp_test_set_processed}"
ETH3D_ROOT="${ETH3D_ROOT:-$REPO/data/eth3d_processed_dust3r}"
KITTI_ROOT="${KITTI_ROOT:-$REPO/data/kitti_val_selection_processed_dust3r}"
TUM_ROOT="${TUM_ROOT:-$REPO/data/tum_processed_v1}"

# Build the benchmark spec from the requested short names, preserving their order.
declare -A BENCH_EXPR=(
  [ScanNetpp]="ScanNetpp:ScanNetpp(split='test', ROOT='${SCANNETPP_ROOT}', resolution=224, aug_crop=0)"
  [ETH3D]="ETH3D:ETH3DProcessedDust3R(split='test', ROOT='${ETH3D_ROOT}', resolution=224, aug_crop=0)"
  [KITTI]="KITTI:KITTIDust3RProcessed(split='test', ROOT='${KITTI_ROOT}', resolution=224, aug_crop=0)"
  [TUM]="TUM:TUMRGBD(split='test', ROOT='${TUM_ROOT}', resolution=224, aug_crop=0)"
)

BENCH=""
IFS=',' read -ra WANTED <<< "$BENCHMARKS"
for b in "${WANTED[@]}"; do
  b="$(echo "$b" | xargs)"
  if [[ -z "${BENCH_EXPR[$b]:-}" ]]; then
    echo "unknown benchmark '$b' (expected one of: ${!BENCH_EXPR[*]})" >&2
    exit 1
  fi
  BENCH+="${BENCH_EXPR[$b]};"
done
BENCH="${BENCH%;}"

# Any MASt3R-shaped model expression works; evaluate_uq.py swaps head_type per method.
MODEL_SRC="${MODEL_SRC:-$CKPT_CONF}"
MODEL="${MODEL:-$("$PY" - "$MODEL_SRC" <<'PY'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["args"].model)
PY
)}"

ARGS=(
  --benchmarks "$BENCH"
  --model "$MODEL"
  --methods "$METHODS"
  --max_pairs "$MAX_PAIRS"
  --scene_fraction 1.0
  --subset_seed 0
  --sim3_align
  --mc_samples "$MC_SAMPLES"
  --out_dir "$OUT_DIR"
)
[[ -n "$CKPT_CONF"      ]] && ARGS+=(--ckpt_conf "$CKPT_CONF")
[[ -n "$CKPT_HETERO"    ]] && ARGS+=(--ckpt_hetero "$CKPT_HETERO")
[[ -n "$CKPT_NIW"       ]] && ARGS+=(--ckpt_niw "$CKPT_NIW")
[[ -n "$CKPT_NIG"       ]] && ARGS+=(--ckpt_nig "$CKPT_NIG")
[[ -n "$MC_CKPT"        ]] && ARGS+=(--mc_ckpt "$MC_CKPT")
[[ -n "$ENSEMBLE_CKPTS" ]] && ARGS+=(--ensemble_ckpts "$ENSEMBLE_CKPTS")

echo "[Trust3R] methods    = $METHODS"
echo "[Trust3R] benchmarks = $BENCHMARKS"
echo "[Trust3R] out_dir    = $OUT_DIR"

"$PY" "$REPO/eval/evaluate_uq.py" "${ARGS[@]}"

echo
echo "[Trust3R] results:"
cat "$OUT_DIR/table1_uq.csv" || true
echo
cat "$OUT_DIR/table2_recon.csv" || true
