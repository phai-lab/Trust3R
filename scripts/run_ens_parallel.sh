#!/usr/bin/env bash
set -euo pipefail

# =========================
# Deep Ensemble (K runs) baseline
# - same mix4 sampling / same epochs / same batch / same save_steps
# - default head: catmlp+dpt (vanilla MASt3R)  <-- recommended baseline
# =========================

K="${K:-10}"

EPOCHS="${EPOCHS:-10}"
TRAIN_N="${TRAIN_N:-150000}"
VAL_N="${VAL_N:-2000}"
BATCH_GLOBAL="${BATCH_GLOBAL:-10}"

# dataset roots
SCANNET_ROOT="${SCANNET_ROOT:-/DATA/zihao/projects/crashtwin/data/scannetpp_processed_cut3r}"
ARKIT_ROOT="${ARKIT_ROOT:-/DATA2/EviP3R/train_set/arkitscenes_processed_sub500}"
WAYMO_ROOT="${WAYMO_ROOT:-/DATA2/EviP3R/train_set/waymo_mast3r_preprocessed}"
MEGA_ROOT="${MEGA_ROOT:-/DATA2/EviP3R/train_set/megadepth_processed_mast3r}"

# per-dataset sampling
TRAIN_N_SCANNET="${TRAIN_N_SCANNET:-25000}"
TRAIN_N_ARKIT="${TRAIN_N_ARKIT:-25000}"
TRAIN_N_WAYMO="${TRAIN_N_WAYMO:-50000}"
TRAIN_N_MEGA="${TRAIN_N_MEGA:-50000}"

# runtime
GPUS_STR="${GPUS_STR:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPUS_STR"
MAX_PARALLEL=${#GPUS[@]}

BASE_PORT="${BASE_PORT:-29600}"
PRETRAINED="${PRETRAINED:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

# output
OUT_ROOT="${OUT_ROOT:-./output_zzh}"
EXP_NAME="${EXP_NAME:-ens_mix4_catmlpdpt_${TRAIN_N}_K${K}}"
OUT_BASE="${OUT_ROOT}/${EXP_NAME}"
mkdir -p "$OUT_BASE"

TRAIN_DS="${TRAIN_N_SCANNET} @ ScanNetpp(split='train', ROOT='${SCANNET_ROOT}', resolution=224, aug_crop=16)+\
${TRAIN_N_ARKIT} @ ARKitScenes(split='train', ROOT='${ARKIT_ROOT}', resolution=224, aug_crop=16)+\
${TRAIN_N_WAYMO} @ Waymo(split='train', ROOT='${WAYMO_ROOT}', resolution=224, aug_crop=16)+\
${TRAIN_N_MEGA} @ MegaDepth(split='train', ROOT='${MEGA_ROOT}', resolution=224, aug_crop=16)"

VAL_DS="${VAL_N} @ ScanNetpp(split='val', ROOT='${SCANNET_ROOT}', resolution=224, aug_crop=0)"

# Common args: keep aligned with NIW/NIG/Hetero where relevant
common_args=(
  --train_dataset "${TRAIN_DS}"
  --test_dataset  "${VAL_DS}"
  --model "AsymmetricMASt3R(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(224, 224),
            head_type='catmlp+dpt',
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
  --pretrained "${PRETRAINED}"
  --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"
  --test_criterion  "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"
  --epochs "${EPOCHS}"
  --warmup_epochs 0
  --save_steps 10000
  --blr 3e-4
  --lambda_uq 0.0
  --lambda_evi 0.0
  --lambda_uq_xyz 0.0
  --lambda_evi_xyz 0.0
  --lambda_hetero_xyz 0.0
  --lambda_evi_feat 0.0
  --lambda_mean_gt 0.0
  --lambda_mean_distill 0.0
  --lambda_var_distill 0.0
)

run_one () {
  local seed="$1"
  local gpu="$2"
  local port="$3"

  local out="${OUT_BASE}/seed${seed}"
  mkdir -p "$out"
  local log="${out}/train.log"

  echo "[ens seed=$seed] gpu=$gpu port=$port out=$out"

  CUDA_VISIBLE_DEVICES="$gpu" \
  torchrun --nproc_per_node=1 --master_port="$port" train.py \
    --seed "$seed" \
    --batch_size "$BATCH_GLOBAL" \
    --output_dir "$out" \
    "${common_args[@]}" \
    2>&1 | tee -a "$log"
}

echo "[ENS] OUT_BASE=$OUT_BASE"
echo "[ENS] PRETRAINED=$PRETRAINED"
echo "[ENS] GPUS=${GPUS_STR}  MAX_PARALLEL=$MAX_PARALLEL  BASE_PORT=$BASE_PORT"
echo "[ENS] TRAIN_DS=$TRAIN_DS"
echo "[ENS] VAL_DS=$VAL_DS"

for SEED in $(seq 0 $((K-1))); do
  GPU="${GPUS[$((SEED % MAX_PARALLEL))]}"
  PORT=$((BASE_PORT + SEED))

  run_one "$SEED" "$GPU" "$PORT" &

  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n
  done
done

wait
echo "[ENS] All done: ${OUT_BASE}"
