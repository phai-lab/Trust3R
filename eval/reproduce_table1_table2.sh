#!/usr/bin/env bash
# Reproduce Trust3R paper Table 1 (uncertainty ranking) and Table 2 (reconstruction
# accuracy). The same run also produces the Table 5 / Table 6 ETH3D ablation rows and
# the Figure 3 risk-coverage / sparsification curves.
#
#   Table 1 -> <OUT_DIR>/table1_uq.csv     read rows `ours_niw_epi`
#   Table 2 -> <OUT_DIR>/table2_recon.csv  read rows `ours_niw_epi`
#   Table 6 -> same files, rows `ours_nig_epi` vs `ours_niw_epi`
#
# Usage:
#   CKPT_NIW=checkpoints/trust3r_niw_mast3r_224.pth \
#   CKPT_NIG=checkpoints/trust3r_nig_mast3r_224.pth \
#   SCANNETPP_ROOT=... ETH3D_ROOT=... KITTI_ROOT=... TUM_ROOT=... \
#   bash eval/reproduce_table1_table2.sh
#
# NOTE ON EXACTNESS
#   Do not change --methods, the benchmark list, or their order. AURC / AUSE / MAE /
#   RMSE are deterministic, but Spearman rho is computed on a 200k-point subsample
#   drawn from a single RNG stream shared across the whole run. Dropping a method or
#   a benchmark shifts that stream and perturbs rho in the third decimal place. The
#   six-method list below is exactly what produced the published numbers, which is
#   why both checkpoints are needed even if you only care about the NIW rows.
#
#   To evaluate a subset -- or your own method -- call eval/evaluate_uq.py directly;
#   see eval/README.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/dust3r:$REPO:${PYTHONPATH:-}"

PY="${PY:-python}"

# ---- checkpoints ------------------------------------------------------------
CKPT_NIW="${CKPT_NIW:-$REPO/checkpoints/trust3r_niw_mast3r_224.pth}"
CKPT_NIG="${CKPT_NIG:-$REPO/checkpoints/trust3r_nig_mast3r_224.pth}"

# ---- preprocessed test sets -------------------------------------------------
SCANNETPP_ROOT="${SCANNETPP_ROOT:-$REPO/data/scannetpp_test_set_processed}"
ETH3D_ROOT="${ETH3D_ROOT:-$REPO/data/eth3d_processed_dust3r}"
KITTI_ROOT="${KITTI_ROOT:-$REPO/data/kitti_val_selection_processed_dust3r}"
TUM_ROOT="${TUM_ROOT:-$REPO/data/tum_processed_v1}"

OUT_DIR="${OUT_DIR:-$REPO/eval_out/trust3r}"

# ---- model expression -------------------------------------------------------
# evaluate_uq.py overrides head_type per method, so any MASt3R-shaped expression
# works. We read it straight out of the released checkpoint.
MODEL="${MODEL:-$("$PY" - "$CKPT_NIW" <<'PY'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["args"].model)
PY
)}"

BENCH="ScanNetpp:ScanNetpp(split='test', ROOT='${SCANNETPP_ROOT}', resolution=224, aug_crop=0);\
ETH3D:ETH3DProcessedDust3R(split='test', ROOT='${ETH3D_ROOT}', resolution=224, aug_crop=0);\
KITTI:KITTIDust3RProcessed(split='test', ROOT='${KITTI_ROOT}', resolution=224, aug_crop=0);\
TUM:TUMRGBD(split='test', ROOT='${TUM_ROOT}', resolution=224, aug_crop=0)"

echo "[Trust3R] NIW checkpoint = $CKPT_NIW"
echo "[Trust3R] NIG checkpoint = $CKPT_NIG"
echo "[Trust3R] out_dir        = $OUT_DIR"

"$PY" "$REPO/eval/evaluate_uq.py" \
  --benchmarks "$BENCH" \
  --model "$MODEL" \
  --methods "ours_nig_total,ours_nig_alea,ours_nig_epi,ours_niw_total,ours_niw_alea,ours_niw_epi" \
  --ckpt_nig "$CKPT_NIG" \
  --ckpt_niw "$CKPT_NIW" \
  --max_pairs 5000 \
  --scene_fraction 1.0 \
  --subset_seed 0 \
  --sim3_align \
  --do_nll \
  --nll_no_sim3 1 \
  --out_dir "$OUT_DIR"

echo
echo "[Trust3R] Table 1 (uncertainty ranking) -- paper rows:"
grep -E "^Method|ours_niw_epi" "$OUT_DIR/table1_uq.csv" || true
echo
echo "[Trust3R] Table 2 (reconstruction accuracy) -- paper rows:"
grep -E "^Method|ours_niw_epi" "$OUT_DIR/table2_recon.csv" || true
