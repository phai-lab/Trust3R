#!/usr/bin/env bash
set -euo pipefail

# =========================
# NIW UQ head retrain (mix4)
# =========================

# --- training length ---
EPOCHS="${EPOCHS:-10}"
TRAIN_N="${TRAIN_N:-150000}"
VAL_N="${VAL_N:-2000}"
BATCH_GLOBAL="${BATCH_GLOBAL:-10}"

# --- UQ loss weights (safer defaults than 0.2) ---
LAMBDA_UQ_XYZ="${LAMBDA_UQ_XYZ:-0.05}"
LAMBDA_EVI_XYZ="${LAMBDA_EVI_XYZ:-1e-3}"
LAMBDA_GATE_TV="${LAMBDA_GATE_TV:-0.0}"
GATE_TV_WARMUP_STEPS="${GATE_TV_WARMUP_STEPS:-2000}"

# --- dataset roots ---
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

PRETRAINED="${PRETRAINED:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

OUT="./output_zzh/gatedres_niw_xyz_mix4_${TRAIN_N}_seed${SEED}_uq${LAMBDA_UQ_XYZ}"
mkdir -p "$OUT"

FINETUNE="${FINETUNE:-0}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
if [ "${FINETUNE}" = "1" ]; then
  PRETRAINED="${OUT}/checkpoint-last.pth"
  EPOCHS="${FINETUNE_EPOCHS}"
fi

TRAIN_DS="${TRAIN_N_SCANNET} @ ScanNetpp(split='train', ROOT='${SCANNET_ROOT}', resolution=224, aug_crop=16)+\
${TRAIN_N_ARKIT} @ ARKitScenes(split='train', ROOT='${ARKIT_ROOT}', resolution=224, aug_crop=16)+\
${TRAIN_N_WAYMO} @ Waymo(split='train', ROOT='${WAYMO_ROOT}', resolution=224, aug_crop=16)+\
${TRAIN_N_MEGA} @ MegaDepth(split='train', ROOT='${MEGA_ROOT}', resolution=224, aug_crop=16)"

VAL_DS="${VAL_N} @ ScanNetpp(split='val', ROOT='${SCANNET_ROOT}', resolution=224, aug_crop=0)"

echo "[NIW] OUT=$OUT"
echo "[NIW] PRETRAINED=$PRETRAINED"
echo "[NIW] TRAIN_DS=$TRAIN_DS"
echo "[NIW] VAL_DS=$VAL_DS"
echo "[NIW] LAMBDA_UQ_XYZ=$LAMBDA_UQ_XYZ  LAMBDA_EVI_XYZ=$LAMBDA_EVI_XYZ"
echo "[NIW] LAMBDA_GATE_TV=$LAMBDA_GATE_TV  GATE_TV_WARMUP_STEPS=$GATE_TV_WARMUP_STEPS"
echo "[NIW] CMD: CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT train.py ..."

set -x

CUDA_VISIBLE_DEVICES="$GPU" \
torchrun --nproc_per_node=1 --master_port="$PORT" train.py \
  --seed "$SEED" \
  --batch_size "$BATCH_GLOBAL" \
  --output_dir "$OUT" \
  --train_dataset "${TRAIN_DS}" \
  --test_dataset  "${VAL_DS}" \
  --model "AsymmetricMASt3R(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(224, 224),
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
        )" \
  --pretrained "$PRETRAINED" \
  --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
  --test_criterion  "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
  --epochs "${EPOCHS}" \
  --warmup_epochs 0 \
  --save_steps 3000 \
  --blr 3e-4 \
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
  --gr_post_smooth 1 \
  --gr_train_mode full \
  --lambda_gate_tv "${LAMBDA_GATE_TV}" \
  --gate_tv_warmup_steps "${GATE_TV_WARMUP_STEPS}" \
  2>&1 | tee -a "${OUT}/train.log"

set +x
echo "[NIW] Training done: ${OUT}"
echo "Done: ${OUT}"
