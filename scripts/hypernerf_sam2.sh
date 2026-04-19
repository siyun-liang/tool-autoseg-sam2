#!/usr/bin/env bash
set -euo pipefail

ALL_SCENES=(
    "americano"
    "chickchicken"
    "espresso"
    "keyboard"
    "split-cookie"
    "torchocolate"
)

if [ "$#" -ge 1 ]; then
    SCENES=("$1")
else
    SCENES=("${ALL_SCENES[@]}")
fi

for scene in "${SCENES[@]}"; do
    for level in "default" "small" "middle" "large"; do
        echo "[scene=${scene}] [level=${level}]"
        python auto-mask-fast.py \
            --video_path "../../data/hypernerf_new/${scene}/rgb/2x" \
            --output_dir "../../data/hypernerf_new/autoseg-sam2/${scene}" \
            --batch_size 4 \
            --detect_stride 10 \
            --level "${level}"
        python visulization.py \
            --video_path "../../data/hypernerf_new/${scene}/rgb/2x" \
            --output_dir "../../data/hypernerf_new/autoseg-sam2/${scene}" \
            --level "${level}"
    done
done
