#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision

try:
    import open_clip
except ImportError as ex:
    raise RuntimeError("open_clip is required. Install with `pip install open-clip-torch`.") from ex


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build per-frame per-level CLIP features from object_masks. "
            "For each frame, output clip_features/{frame_id:06d}_f.npy with shape [K,512]."
        )
    )
    parser.add_argument("--output_root", type=str, required=True, help="Root like data/hypernerf_new/output")
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument(
        "--rgb_root",
        type=str,
        required=True,
        help="Root where RGB exists as <rgb_root>/<scene>/rgb/2x/*.png",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="default,small,middle,large",
        help="Comma-separated level list",
    )
    parser.add_argument("--model_type", type=str, default="ViT-B-16")
    parser.add_argument("--model_pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing *_f.npy")
    return parser.parse_args()


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    side = max(h, w)
    out = np.zeros((side, side, c), dtype=np.uint8)
    if h >= w:
        x0 = (h - w) // 2
        out[:, x0 : x0 + w] = img
    else:
        y0 = (w - h) // 2
        out[y0 : y0 + h, :] = img
    return out


def crop_masked_object(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    masked = rgb.copy()
    masked[~mask] = 0
    crop = masked[y0 : y1 + 1, x0 : x1 + 1]
    crop = pad_to_square(crop)
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    return crop


def build_rgb_list(rgb_dir: Path) -> List[Path]:
    files = sorted(rgb_dir.glob("*.png"))
    if not files:
        raise RuntimeError(f"No RGB png found in {rgb_dir}")
    return files


def load_mask(mask_path: Path) -> np.ndarray:
    m = np.load(mask_path)
    if m.ndim != 2:
        raise ValueError(f"Expected mask shape [H,W], got {m.shape} at {mask_path}")
    return m.astype(np.int32, copy=False)


def frame_rgb_by_id(rgb_files: List[Path], frame_id_1based: int) -> Path:
    # Prefer explicit 000001.png if present, fallback to sorted index.
    expected = f"{frame_id_1based:06d}.png"
    for p in rgb_files[max(0, frame_id_1based - 5) : min(len(rgb_files), frame_id_1based + 4)]:
        if p.name == expected:
            return p
    idx = frame_id_1based - 1
    if idx < 0 or idx >= len(rgb_files):
        raise IndexError(f"frame_id {frame_id_1based} out of range for rgb files {len(rgb_files)}")
    return rgb_files[idx]


def main():
    args = parse_args()
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    if not levels:
        raise ValueError("No valid levels parsed from --levels")

    scene_root = Path(args.output_root).expanduser().resolve() / args.scene
    if not scene_root.is_dir():
        raise FileNotFoundError(f"Scene output root not found: {scene_root}")

    rgb_dir = Path(args.rgb_root).expanduser().resolve() / args.scene / "rgb" / "2x"
    rgb_files = build_rgb_list(rgb_dir)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        args.model_type,
        pretrained=args.model_pretrained,
        precision="fp16" if device.startswith("cuda") else "fp32",
    )
    model = model.to(device).eval()
    process = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    for level in levels:
        level_dir = scene_root / level
        meta_path = level_dir / "meta.json"
        masks_dir = level_dir / "object_masks"
        if not (meta_path.is_file() and masks_dir.is_dir()):
            raise FileNotFoundError(f"Missing level data for {level}: {meta_path} / {masks_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        num_frames = int(meta["num_frames"])
        k = int(meta["K"])
        if k < 0:
            raise ValueError(f"Invalid K={k} in {meta_path}")

        out_dir = level_dir / "clip_features"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[level={level}] num_frames={num_frames} K={k} -> {out_dir}")

        for frame_id in range(1, num_frames + 1):
            out_f = out_dir / f"{frame_id:06d}_f.npy"
            if out_f.is_file() and not args.overwrite:
                continue

            mask_path = masks_dir / f"{frame_id:06d}_s.npy"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask = load_mask(mask_path)

            rgb_path = frame_rgb_by_id(rgb_files, frame_id)
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(f"Cannot read rgb frame: {rgb_path}")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != mask.shape[:2]:
                rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

            feats = np.zeros((k, 512), dtype=np.float32)
            crops: List[np.ndarray] = []
            ids: List[int] = []
            for obj_id in range(1, k + 1):
                fg = mask == obj_id
                if not fg.any():
                    continue
                crop = crop_masked_object(rgb, fg)
                crops.append(crop)
                ids.append(obj_id)

            if crops:
                tiles = np.stack(crops, axis=0).astype(np.float32) / 255.0  # [M,224,224,3]
                tiles = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(device)
                with torch.no_grad():
                    inp = process(tiles)
                    if device.startswith("cuda"):
                        inp = inp.half()
                    else:
                        inp = inp.float()
                    emb = model.encode_image(inp)
                    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                emb_np = emb.detach().float().cpu().numpy()
                for row, obj_id in enumerate(ids):
                    feats[obj_id - 1] = emb_np[row]

            np.save(out_f, feats)

        clip_meta_path = out_dir / "meta.json"
        with open(clip_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scene": args.scene,
                    "level": level,
                    "num_frames": num_frames,
                    "K": k,
                    "feature_dim": 512,
                    "model_type": args.model_type,
                    "model_pretrained": args.model_pretrained,
                    "filename_pattern": "{frame_id:06d}_f.npy",
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
