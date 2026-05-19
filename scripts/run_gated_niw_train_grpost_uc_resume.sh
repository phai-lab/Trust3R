#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Resume / continue stage2_uq_uc_l001_total for grpost root-fix runs.
#
# Notes:
# - train.py auto-resumes from ${OUT_STAGE_UC}/checkpoint-last.pth when present.
# - --epochs is the TOTAL epoch budget, not the number of extra epochs.
# - If OUT_STAGE_UC has no checkpoint yet, we initialize from stage2_uq.
# =========================================================

# --- target run ---
OUT_ROOT="${OUT_ROOT:-./output_zzh/gatedres_niw_xyz_mix4_150000_seed0_uq0.05_grpost_rootfix_res512}"
OUT_STAGE2="${OUT_STAGE2:-${OUT_ROOT}/stage2_uq}"
OUT_STAGE_UC="${OUT_STAGE_UC:-${OUT_ROOT}/stage2_uq_uc_l001_total}"

# --- total epoch budget for the UC stage ---
UC_EPOCHS_TOTAL="${UC_EPOCHS_TOTAL:-20}"

# --- dataset / image settings ---
TRAIN_N="${TRAIN_N:-150000}"
VAL_N="${VAL_N:-2000}"
BATCH_GLOBAL="${BATCH_GLOBAL:-10}"
RES="${RES:-512}"
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
GPU="${GPU:-0}"
PORT="${PORT:-29680}"
SEED="${SEED:-0}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
BLR="${BLR:-3e-4}"

# --- NIW + UC loss weights ---
LAMBDA_UQ_XYZ="${LAMBDA_UQ_XYZ:-0.05}"
LAMBDA_EVI_XYZ="${LAMBDA_EVI_XYZ:-1e-3}"
LAMBDA_UC_CORR="${LAMBDA_UC_CORR:-0.01}"
UC_CORR_USE="${UC_CORR_USE:-total}"
UC_CORR_LOG="${UC_CORR_LOG:-1}"
UC_CONF_KEY="${UC_CONF_KEY:-conf}"

# --- GR seam-fix knobs ---
GR_RESIDUAL_MODE="${GR_RESIDUAL_MODE:-bilinear}"
GR_POST_SMOOTH="${GR_POST_SMOOTH:-1}"
GR_POST_KS="${GR_POST_KS:-5}"
GR_POST_LR="${GR_POST_LR:-1e-5}"
GR_POST_WD="${GR_POST_WD:-0.0}"
GR_TRAIN_MODE="${GR_TRAIN_MODE:-post_only}"

# --- Gate-TV regularization ---
LAMBDA_GATE_TV="${LAMBDA_GATE_TV:-1e-4}"
GATE_TV_WARMUP_STEPS="${GATE_TV_WARMUP_STEPS:-2000}"

mkdir -p "${OUT_STAGE_UC}"

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

CKPT_INIT="$(pick_ckpt "${OUT_STAGE_UC}")"
if [ -z "${CKPT_INIT}" ]; then
  CKPT_INIT="$(pick_ckpt "${OUT_STAGE2}")"
fi
if [ -z "${CKPT_INIT}" ] || [ ! -f "${CKPT_INIT}" ]; then
  echo "[UC-RESUME][FATAL] Could not find an init checkpoint from ${OUT_STAGE_UC} or ${OUT_STAGE2}" >&2
  exit 1
fi

echo "[UC-RESUME] OUT_ROOT=${OUT_ROOT}"
echo "[UC-RESUME] OUT_STAGE2=${OUT_STAGE2}"
echo "[UC-RESUME] OUT_STAGE_UC=${OUT_STAGE_UC}"
echo "[UC-RESUME] INIT_CKPT=${CKPT_INIT}"
echo "[UC-RESUME] UC_EPOCHS_TOTAL=${UC_EPOCHS_TOTAL}"
echo "[UC-RESUME] TRAIN_DS=${TRAIN_DS}"
echo "[UC-RESUME] VAL_DS=${VAL_DS}"
echo "[UC-RESUME] lambda_uq_xyz=${LAMBDA_UQ_XYZ} lambda_evi_xyz=${LAMBDA_EVI_XYZ} lambda_uc_corr=${LAMBDA_UC_CORR}"
echo "[UC-RESUME] uc_corr_use=${UC_CORR_USE} uc_corr_log=${UC_CORR_LOG} uc_conf_key=${UC_CONF_KEY}"
echo "[UC-RESUME] gr_train_mode=${GR_TRAIN_MODE} gr_post_lr=${GR_POST_LR} gr_post_wd=${GR_POST_WD}"

set -x

CUDA_VISIBLE_DEVICES="${GPU}" \
torchrun --nproc_per_node=1 --master_port="${PORT}" train.py \
  --seed "${SEED}" \
  --batch_size "${BATCH_GLOBAL}" \
  --output_dir "${OUT_STAGE_UC}" \
  --train_dataset "${TRAIN_DS}" \
  --test_dataset "${VAL_DS}" \
  --model "${MODEL_STR}" \
  --pretrained "${CKPT_INIT}" \
  --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
  --test_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
  --epochs "${UC_EPOCHS_TOTAL}" \
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
  --lambda_uc_corr "${LAMBDA_UC_CORR}" \
  --uc_corr_use "${UC_CORR_USE}" \
  --uc_corr_log "${UC_CORR_LOG}" \
  --uc_conf_key "${UC_CONF_KEY}" \
  --lambda_evi_feat 0.0 \
  --lambda_mean_gt 0.0 \
  --lambda_mean_distill 0.0 \
  --lambda_var_distill 0.0 \
  --gr_residual_mode "${GR_RESIDUAL_MODE}" \
  --gr_post_smooth "${GR_POST_SMOOTH}" \
  --gr_post_ks "${GR_POST_KS}" \
  --gr_train_mode "${GR_TRAIN_MODE}" \
  --gr_post_lr "${GR_POST_LR}" \
  --gr_post_wd "${GR_POST_WD}" \
  --lambda_gate_tv "${LAMBDA_GATE_TV}" \
  --gate_tv_warmup_steps "${GATE_TV_WARMUP_STEPS}" \
  2>&1 | tee -a "${OUT_STAGE_UC}/train_uc.log"

set +x
echo "[UC-RESUME] Done: ${OUT_STAGE_UC}"
