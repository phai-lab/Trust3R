#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# NIW root-fix training (2-stage):
#   1) geometry clarity with bilinear residual-gated upsampling
#   2) NIW UQ training with frozen geometry
# =========================================================

# --- training length ---
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
TRAIN_N="${TRAIN_N:-150000}"
VAL_N="${VAL_N:-2000}"
BATCH_GLOBAL="${BATCH_GLOBAL:-10}"

# --- UQ loss weights ---
LAMBDA_UQ_XYZ="${LAMBDA_UQ_XYZ:-0.05}"
LAMBDA_EVI_XYZ="${LAMBDA_EVI_XYZ:-1e-3}"

# --- GR seam-fix knobs ---
GR_RESIDUAL_MODE="${GR_RESIDUAL_MODE:-bilinear}"     # pixel_shuffle | bilinear
GR_POST_SMOOTH="${GR_POST_SMOOTH:-1}"                # 1 recommended for bilinear mode
GR_POST_KS="${GR_POST_KS:-5}"
GR_POST_LR="${GR_POST_LR:-1e-5}"                     # used in stage2 post_only group
GR_POST_WD="${GR_POST_WD:-0.0}"                      # used in stage2 post_only group
STAGE1_GR_TRAIN_MODE="${STAGE1_GR_TRAIN_MODE:-gr_only}"    # full | post_only | gr_only
STAGE2_GR_TRAIN_MODE="${STAGE2_GR_TRAIN_MODE:-post_only}"  # full | post_only

# --- Gate-TV regularization ---
LAMBDA_GATE_TV_STAGE1="${LAMBDA_GATE_TV_STAGE1:-1e-4}"
LAMBDA_GATE_TV_STAGE2="${LAMBDA_GATE_TV_STAGE2:-1e-4}"
GATE_TV_WARMUP_STEPS="${GATE_TV_WARMUP_STEPS:-2000}"

# --- dataset / image settings ---
RES="${RES:-448}"
AUG_CROP="${AUG_CROP:-16}"
SCANNET_ROOT="${SCANNET_ROOT:-/DATA/zihao/projects/crashtwin/data/scannetpp_processed_cut3r}"
ARKIT_ROOT="${ARKIT_ROOT:-/DATA2/EviP3R/train_set/arkitscenes_processed_sub500}"
WAYMO_ROOT="${WAYMO_ROOT:-/DATA2/EviP3R/train_set/waymo_mast3r_preprocessed}"
MEGA_ROOT="${MEGA_ROOT:-/DATA2/EviP3R/train_set/megadepth_processed_mast3r}"

# --- per-dataset sampling ---
TRAIN_N_SCANNET="${TRAIN_N_SCANNET:-25000}"
TRAIN_N_ARKIT="${TRAIN_N_ARKIT:-25000}"
TRAIN_N_WAYMO="${TRAIN_N_WAYMO:-50000}"
TRAIN_N_MEGA="${TRAIN_N_MEGA:-50000}"

# --- runtime ---
GPU="${GPU:-2}"
PORT="${PORT:-29652}"
SEED="${SEED:-0}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
BLR="${BLR:-3e-4}"

PRETRAINED="${PRETRAINED:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
OUT_ROOT="${OUT_ROOT:-./output_zzh/gatedres_niw_xyz_mix4_${TRAIN_N}_seed${SEED}_uq${LAMBDA_UQ_XYZ}_grpost_rootfix_res${RES}}"
OUT_STAGE1="${OUT_STAGE1:-${OUT_ROOT}/stage1_geom}"
OUT_STAGE2="${OUT_STAGE2:-${OUT_ROOT}/stage2_uq}"
mkdir -p "${OUT_STAGE1}" "${OUT_STAGE2}"

TRAIN_DS="${TRAIN_N_SCANNET} @ ScanNetpp(split='train', ROOT='${SCANNET_ROOT}', resolution=${RES}, aug_crop=${AUG_CROP})+\
${TRAIN_N_ARKIT} @ ARKitScenes(split='train', ROOT='${ARKIT_ROOT}', resolution=${RES}, aug_crop=${AUG_CROP})+\
${TRAIN_N_WAYMO} @ Waymo(split='train', ROOT='${WAYMO_ROOT}', resolution=${RES}, aug_crop=${AUG_CROP})+\
${TRAIN_N_MEGA} @ MegaDepth(split='train', ROOT='${MEGA_ROOT}', resolution=${RES}, aug_crop=${AUG_CROP})"

VAL_DS="${VAL_N} @ ScanNetpp(split='val', ROOT='${SCANNET_ROOT}', resolution=${RES}, aug_crop=0)"

MODEL_STR="AsymmetricMASt3R(
  pos_embed='RoPE100',
  patch_embed_cls='ManyAR_PatchEmbed',
  img_size=(${RES}, ${RES}),
  head_type='catmlp+dpt+xyz_niw_dpt+residual_gated',
  output_mode='pts3d+desc24',
  depth_mode=('exp', -float('inf'), float('inf')),
  conf_mode=('exp', 1, float('inf')),
  enc_embed_dim=1024,
  enc_depth=24,
  enc_num_heads=16,
  dec_embed_dim=768,
  dec_depth=12,
  dec_num_heads=12,
  two_confs=True,
  desc_conf_mode=('exp', 0, float('inf')),
  landscape_only=False
)"

pick_ckpt() {
  local out_dir="$1"
  local last_ckpt="${out_dir}/checkpoint-last.pth"
  if [ -f "${last_ckpt}" ]; then
    echo "${last_ckpt}"
    return
  fi
  ls -1 "${out_dir}"/checkpoint-step-*.pth 2>/dev/null | tail -n 1 || true
}

echo "[ROOTFIX] OUT_ROOT=${OUT_ROOT}"
echo "[ROOTFIX] PRETRAINED=${PRETRAINED}"
echo "[ROOTFIX] TRAIN_DS=${TRAIN_DS}"
echo "[ROOTFIX] VAL_DS=${VAL_DS}"
echo "[ROOTFIX] RES=${RES} AUG_CROP=${AUG_CROP}"
echo "[ROOTFIX] GR_RESIDUAL_MODE=${GR_RESIDUAL_MODE} GR_POST_SMOOTH=${GR_POST_SMOOTH} GR_POST_KS=${GR_POST_KS}"
echo "[ROOTFIX] Stage1 epochs=${STAGE1_EPOCHS} (lambda_uq_xyz=0, gr_train_mode=${STAGE1_GR_TRAIN_MODE})"
echo "[ROOTFIX] Stage2 epochs=${STAGE2_EPOCHS} (freeze_geom_for_uq, uq_mode=xyz, gr_train_mode=${STAGE2_GR_TRAIN_MODE})"

set -x

# -------------------------
# Stage 1: geometry / mean clarity
# -------------------------
CUDA_VISIBLE_DEVICES="${GPU}" \
torchrun --nproc_per_node=1 --master_port="${PORT}" train.py \
  --seed "${SEED}" \
  --batch_size "${BATCH_GLOBAL}" \
  --output_dir "${OUT_STAGE1}" \
  --train_dataset "${TRAIN_DS}" \
  --test_dataset "${VAL_DS}" \
  --model "${MODEL_STR}" \
  --pretrained "${PRETRAINED}" \
  --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
  --test_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
  --epochs "${STAGE1_EPOCHS}" \
  --warmup_epochs 0 \
  --save_steps "${SAVE_STEPS}" \
  --blr "${BLR}" \
  --lambda_uq 0.0 \
  --lambda_evi 0.0 \
  --lambda_uq_xyz 0.0 \
  --lambda_evi_xyz 0.0 \
  --lambda_hetero_xyz 0.0 \
  --uq_mode geom \
  --lambda_evi_feat 0.0 \
  --lambda_mean_gt 0.0 \
  --lambda_mean_distill 0.0 \
  --lambda_var_distill 0.0 \
  --gr_residual_mode "${GR_RESIDUAL_MODE}" \
  --gr_post_smooth "${GR_POST_SMOOTH}" \
  --gr_post_ks "${GR_POST_KS}" \
  --gr_train_mode "${STAGE1_GR_TRAIN_MODE}" \
  --gr_post_lr "${GR_POST_LR}" \
  --gr_post_wd "${GR_POST_WD}" \
  --lambda_gate_tv "${LAMBDA_GATE_TV_STAGE1}" \
  --gate_tv_warmup_steps "${GATE_TV_WARMUP_STEPS}" \
  2>&1 | tee -a "${OUT_STAGE1}/train.log"

CKPT_STAGE1="$(pick_ckpt "${OUT_STAGE1}")"
if [ -z "${CKPT_STAGE1}" ] || [ ! -f "${CKPT_STAGE1}" ]; then
  echo "[ROOTFIX][FATAL] Stage1 checkpoint not found." >&2
  exit 1
fi

# -------------------------
# Stage 2: NIW UQ training
# -------------------------
CUDA_VISIBLE_DEVICES="${GPU}" \
torchrun --nproc_per_node=1 --master_port="$((PORT + 1))" train.py \
  --seed "${SEED}" \
  --batch_size "${BATCH_GLOBAL}" \
  --output_dir "${OUT_STAGE2}" \
  --train_dataset "${TRAIN_DS}" \
  --test_dataset "${VAL_DS}" \
  --model "${MODEL_STR}" \
  --pretrained "${CKPT_STAGE1}" \
  --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
  --test_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
  --epochs "${STAGE2_EPOCHS}" \
  --warmup_epochs 0 \
  --save_steps "${SAVE_STEPS}" \
  --blr "${BLR}" \
  --freeze_geom_for_uq \
  --uq_mode xyz \
  --lambda_uq 0.0 \
  --lambda_evi 0.0 \
  --lambda_uq_xyz "${LAMBDA_UQ_XYZ}" \
  --lambda_evi_xyz "${LAMBDA_EVI_XYZ}" \
  --lambda_hetero_xyz 0.0 \
  --lambda_evi_feat 0.0 \
  --lambda_mean_gt 0.0 \
  --lambda_mean_distill 0.0 \
  --lambda_var_distill 0.0 \
  --gr_residual_mode "${GR_RESIDUAL_MODE}" \
  --gr_post_smooth "${GR_POST_SMOOTH}" \
  --gr_post_ks "${GR_POST_KS}" \
  --gr_train_mode "${STAGE2_GR_TRAIN_MODE}" \
  --gr_post_lr "${GR_POST_LR}" \
  --gr_post_wd "${GR_POST_WD}" \
  --lambda_gate_tv "${LAMBDA_GATE_TV_STAGE2}" \
  --gate_tv_warmup_steps "${GATE_TV_WARMUP_STEPS}" \
  2>&1 | tee -a "${OUT_STAGE2}/train.log"

set +x
echo "Done: ${OUT_STAGE2}"
