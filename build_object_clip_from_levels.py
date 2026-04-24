#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build per-level object CLIP tables by averaging per-frame slot features "
            "into a single object_clip.npy with shape [K, C]."
        )
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root like data/hypernerf_new/output",
    )
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument(
        "--levels",
        type=str,
        default="default,small,middle,large",
        help="Comma-separated levels to process",
    )
    parser.add_argument(
        "--mask_dir_name",
        type=str,
        default="object_masks",
        help="Per-level mask directory name",
    )
    parser.add_argument(
        "--feature_dir_name",
        type=str,
        default="clip_features",
        help="Per-level per-frame feature directory name",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="object_clip.npy",
        help="Output filename under each level folder",
    )
    parser.add_argument(
        "--weight_mode",
        type=str,
        default="uniform",
        choices=["uniform", "area"],
        help=(
            "Cross-frame weighting per slot: "
            "uniform=each valid frame counts equally; area=weight by slot pixel area in frame."
        ),
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=1,
        help="Minimum visible pixels for a slot in a frame to be included.",
    )
    parser.add_argument(
        "--l2_normalize_input",
        action="store_true",
        help="L2-normalize each frame slot feature before aggregation.",
    )
    parser.add_argument(
        "--l2_normalize_output",
        action="store_true",
        help="L2-normalize final object_clip vectors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing object_clip.npy",
    )
    return parser.parse_args()


def _load_level_meta(level_dir: Path) -> Tuple[int, int]:
    meta_path = level_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    num_frames = int(meta["num_frames"])
    k = int(meta["K"])
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames={num_frames} in {meta_path}")
    if k < 0:
        raise ValueError(f"Invalid K={k} in {meta_path}")
    return num_frames, k


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def _load_mask(mask_path: Path) -> np.ndarray:
    m = np.load(mask_path)
    if m.ndim != 2:
        raise ValueError(f"Expected mask [H,W], got {m.shape} at {mask_path}")
    return m.astype(np.int32, copy=False)


def _load_feat(feat_path: Path) -> np.ndarray:
    f = np.load(feat_path)
    if f.ndim != 2:
        raise ValueError(f"Expected feature [K,C], got {f.shape} at {feat_path}")
    return f.astype(np.float32, copy=False)


def process_level(args, scene_root: Path, level: str):
    level_dir = scene_root / level
    if not level_dir.is_dir():
        raise FileNotFoundError(f"Level dir not found: {level_dir}")

    out_path = level_dir / args.save_name
    meta_out_path = level_dir / "object_clip_meta.json"
    if out_path.is_file() and not args.overwrite:
        print(f"[skip] {level}: {out_path} already exists")
        return

    num_frames, k = _load_level_meta(level_dir)
    mask_dir = level_dir / args.mask_dir_name
    feat_dir = level_dir / args.feature_dir_name
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
    if not feat_dir.is_dir():
        raise FileNotFoundError(f"Feature dir not found: {feat_dir}")

    feat_dim = None
    feat_sum = None
    feat_weight_sum = np.zeros((k,), dtype=np.float64)
    frames_seen = np.zeros((k,), dtype=np.int64)
    pixels_seen = np.zeros((k,), dtype=np.int64)

    used_frames = 0
    for frame_id in range(1, num_frames + 1):
        mask_path = mask_dir / f"{frame_id:06d}_s.npy"
        feat_path = feat_dir / f"{frame_id:06d}_f.npy"
        if not mask_path.is_file() or not feat_path.is_file():
            continue

        mask = _load_mask(mask_path)
        feat = _load_feat(feat_path)
        if feat.shape[0] < k:
            raise ValueError(
                f"Feature row count < K for {feat_path}: feat_rows={feat.shape[0]}, K={k}"
            )

        if feat_dim is None:
            feat_dim = int(feat.shape[1])
            feat_sum = np.zeros((k, feat_dim), dtype=np.float64)
        elif int(feat.shape[1]) != feat_dim:
            raise ValueError(
                f"Feature dim mismatch at {feat_path}: got {feat.shape[1]}, expected {feat_dim}"
            )

        used_frames += 1
        uniq_ids = np.unique(mask)
        uniq_ids = uniq_ids[(uniq_ids > 0) & (uniq_ids <= k)]

        # Per-frame slot representation (frame-level average):
        # Each slot has one feature row in this pipeline (feat[slot-1]).
        # We keep the same slot-level reduction interface as the original dense remap.
        for sid in uniq_ids.tolist():
            sid_i = int(sid)
            pix = int((mask == sid_i).sum())
            if pix < int(args.min_pixels):
                continue
            w = float(pix) if args.weight_mode == "area" else 1.0

            row = sid_i - 1
            vec = feat[row : row + 1]
            if args.l2_normalize_input:
                vec = _l2_normalize_rows(vec)
            vec = vec[0]

            feat_sum[row] += w * vec.astype(np.float64, copy=False)
            feat_weight_sum[row] += w
            frames_seen[row] += 1
            pixels_seen[row] += pix

    if feat_dim is None:
        raise RuntimeError(f"No valid frame feature files found in {feat_dir}")

    object_clip = np.zeros((k, feat_dim), dtype=np.float32)
    valid = feat_weight_sum > 0
    if valid.any():
        object_clip[valid] = (
            feat_sum[valid] / feat_weight_sum[valid, None]
        ).astype(np.float32)
    if args.l2_normalize_output and valid.any():
        object_clip[valid] = _l2_normalize_rows(object_clip[valid])

    np.save(out_path, object_clip)
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene": args.scene,
                "level": level,
                "K": int(k),
                "feature_dim": int(feat_dim),
                "num_frames_meta": int(num_frames),
                "num_frames_used": int(used_frames),
                "weight_mode": args.weight_mode,
                "min_pixels": int(args.min_pixels),
                "l2_normalize_input": bool(args.l2_normalize_input),
                "l2_normalize_output": bool(args.l2_normalize_output),
                "frames_seen_per_slot": frames_seen.tolist(),
                "pixels_seen_per_slot": pixels_seen.tolist(),
                "valid_slot_count": int(valid.sum()),
                "output_file": args.save_name,
                "input_mask_dir": args.mask_dir_name,
                "input_feature_dir": args.feature_dir_name,
            },
            f,
            indent=2,
        )
    print(
        f"[done] level={level} -> {out_path} "
        f"shape={object_clip.shape}, valid_slots={int(valid.sum())}/{k}, frames_used={used_frames}"
    )


def main():
    args = parse_args()
    levels: List[str] = [x.strip() for x in args.levels.split(",") if x.strip()]
    if not levels:
        raise ValueError("No valid levels parsed from --levels")

    scene_root = Path(args.output_root).expanduser().resolve() / args.scene
    if not scene_root.is_dir():
        raise FileNotFoundError(f"Scene root not found: {scene_root}")

    for level in levels:
        process_level(args, scene_root, level)


if __name__ == "__main__":
    main()
