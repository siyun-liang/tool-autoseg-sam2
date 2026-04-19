#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-chickchicken}"
ROOT="/home/liang/Documents/students/zhiyu/4DSuperGSeg"
AUTOSEG_DIR="${ROOT}/preprocess/AutoSeg-SAM2"
AE_DIR="${ROOT}/preprocess/AutoSeg-SAM2/autoencoder"
DATASET_PATH="${ROOT}/data/hypernerf_new/output/${SCENE}"
LEVELS="${LEVELS:-default,small,middle,large}"
MASK_INPUT_ROOT="${MASK_INPUT_ROOT:-${ROOT}/data/hypernerf_new/autoseg-sam2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/data/hypernerf_new/output}"
RGB_ROOT="${RGB_ROOT:-${ROOT}/data/hypernerf_new}"

# 0) build per-level contiguous object masks (object_masks + global2slot + meta)
python "${AUTOSEG_DIR}/build_slot_masks_from_binary_stack.py" \
  --input_root "${MASK_INPUT_ROOT}" \
  --scene "${SCENE}" \
  --levels "${LEVELS}" \
  --output_root "${OUTPUT_ROOT}"

# 1) build per-level CLIP features: clip_features/{frame_id:06d}_f.npy
python "${AUTOSEG_DIR}/build_level_clip_features.py" \
  --output_root "${OUTPUT_ROOT}" \
  --scene "${SCENE}" \
  --rgb_root "${RGB_ROOT}" \
  --levels "${LEVELS}"

cd "${AE_DIR}"

# 2) train AE on aggregated multi-level clip_features (512D -> 3D)
python train.py \
  --dataset_path "${DATASET_PATH}" \
  --language_name clip_features \
  --levels "${LEVELS}" \
  --model_name "${SCENE}_clip" \
  --feature_dims 512 \
  --encoder_dims 256 128 64 32 3 \
  --decoder_dims 16 32 64 128 256 512 \
  --hidden_dims 3 \
  --batch_size 256 \
  --lr 7e-4

# 3) export per-level dim3 features, keep per-frame obj_id row order
python test.py \
  --dataset_path "${DATASET_PATH}" \
  --language_name clip_features \
  --levels "${LEVELS}" \
  --model_name "${SCENE}_clip" \
  --feature_dims 512 \
  --encoder_dims 256 128 64 32 3 \
  --decoder_dims 16 32 64 128 256 512 \
  --hidden_dims 3 \
  --output_name clip_features \
  --output_suffix _f.npy
