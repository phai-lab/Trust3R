#!/usr/bin/env bash
set -euo pipefail

# =========================
# Heteroscedastic XYZ head retrain (mix4)
# =========================

EPOCHS="${EPOCHS:-10}"
TRAIN_N="${TRAIN_N:-150000}"
VAL_N="${VAL_N:-2000}"
BATCH_GLOBAL="${BATCH_GLOBAL:-10}"

# hetero weight (keep 1.0 by default)
LAMBDA_HETERO_XYZ="${LAMBDA_HETERO_XYZ:-1.0}"

SCANNET_ROOT="${SCANNET_ROOT:-/DATA/zihao/projects/crashtwin/data/scannetpp_processed_cut3r}"
ARKIT_ROOT="${ARKIT_ROOT:-/DATA2/EviP3R/train_set/arkitscenes_processed_sub500}"
WAYMO_ROOT="${WAYMO_ROOT:-/DATA2/EviP3R/train_set/waymo_mast3r_preprocessed}"
MEGA_ROOT="${MEGA_ROOT:-/DATA2/EviP3R/train_set/megadepth_processed_mast3r}"

TRAIN_N_SCANNET="${TRAIN_N_SCANNET:-25000}"
TRAIN_N_ARKIT="${TRAIN_N_ARKIT:-25000}"
TRAIN_N_WAYMO="${TRAIN_N_WAYMO:-50000}"
TRAIN_N_MEGA="${TRAIN_N_MEGA:-50000}"

GPU="${GPU:-0}"
PORT="${PORT:-29650}"
SEED="${SEED:-0}"

PRETRAINED="${PRETRAINED:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

OUT="./output_zzh/hetero_xyz_mix4_${TRAIN_N}_seed${SEED}_lam${LAMBDA_HETERO_XYZ}"
mkdir -p "$OUT"

TRAIN_DS="${TRAIN_N_SCANNET} @ ScanNetpp(split='train', ROOT='${SCANNET_ROOT}', resolution=224, aug_crop=16)+\
${TRAIN_N_ARKIT} @ ARKitScenes(split='train', ROOT='${ARKIT_ROOT}', resolution=224, aug_crop=16)+\
${TRAIN_N_WAYMO} @ Waymo(split='train', ROOT='${WAYMO_ROOT}', resolution=224, aug_crop=16)+\
${TRAIN_N_MEGA} @ MegaDepth(split='train', ROOT='${MEGA_ROOT}', resolution=224, aug_crop=16)"

VAL_DS="${VAL_N} @ ScanNetpp(split='val', ROOT='${SCANNET_ROOT}', resolution=224, aug_crop=0)"

echo "[HETERO] OUT=$OUT"
echo "[HETERO] LAMBDA_HETERO_XYZ=$LAMBDA_HETERO_XYZ"
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
            head_type='catmlp+dpt+xyz_hetero_dpt',
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
  --save_steps 10000 \
  --blr 3e-4 \
  --freeze_geom_for_uq \
  --uq_mode hetero_xyz \
  --lambda_hetero_xyz "${LAMBDA_HETERO_XYZ}" \
  --lambda_uq 0.0 \
  --lambda_evi 0.0 \
  --lambda_uq_xyz 0.0 \
  --lambda_evi_xyz 0.0 \
  --lambda_evi_feat 0.0 \
  --lambda_mean_gt 0.0 \
  --lambda_mean_distill 0.0 \
  --lambda_var_distill 0.0 \
  2>&1 | tee -a "${OUT}/train.log"

set +x
echo "Done: ${OUT}"
