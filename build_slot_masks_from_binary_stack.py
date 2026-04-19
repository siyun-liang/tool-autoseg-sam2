#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


LEVELS_DEFAULT = ["default", "small", "middle", "large"]
PREFER_SMALL_LEVELS = {"default", "small"}


@dataclass
class FrameResult:
    assigned_global: np.ndarray  # [H, W], global id (0 bg)
    assigned_slot: np.ndarray  # [H, W], remapped contiguous slot id (0 bg)
    used_global_ids: np.ndarray  # sorted unique foreground global ids used in this frame


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert overlapping binary mask stacks (N_obj,1,H,W) into single-label id maps, "
            "with per-level NMS, level-aware overlap assignment, and contiguous global2slot mapping."
        )
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help=(
            "Root containing level folders, e.g. "
            "data/hypernerf_new/autoseg-sam2-postnms/<scene>."
        ),
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name used for locating RGB frames and writing outputs.",
    )
    parser.add_argument(
        "--rgb_root",
        type=str,
        default="",
        help=(
            "Optional root for RGB frames. If set, expects <rgb_root>/<scene>/rgb/2x/*.png. "
            "Required when --visualize is enabled."
        ),
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="default,small,middle,large",
        help="Comma-separated level names in channel order for output stack.",
    )
    parser.add_argument(
        "--level_mask_subdir",
        type=str,
        default="final-output",
        help="Subdir under each level that contains mask_*.npy.",
    )
    parser.add_argument(
        "--mask_glob",
        type=str,
        default="mask_*.npy",
        help="Glob pattern for mask stacks under each level subdir.",
    )
    parser.add_argument(
        "--iou_thr",
        type=float,
        default=0.85,
        help="Greedy NMS IoU threshold (used only when --enable_nms is set).",
    )
    parser.add_argument(
        "--contain_thr",
        type=float,
        default=0.95,
        help=(
            "Containment threshold. If intersection/min(area_a,area_b) >= contain_thr, "
            "the lower-priority mask is removed (used only when --enable_nms is set)."
        ),
    )
    parser.add_argument(
        "--enable_nms",
        action="store_true",
        help="Enable NMS before conflict assignment. Default is disabled.",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=20,
        help="Drop masks with area < min_area before NMS.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help=(
            "Output root. Will create "
            "<output_root>/<scene>/<level>/object_masks/*.npy, "
            "<output_root>/<scene>/<level>/global2slot.json, and "
            "<output_root>/<scene>/<level>/meta.json."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization export.",
    )
    parser.add_argument(
        "--viz_fps",
        type=int,
        default=24,
        help="FPS for output videos.",
    )
    parser.add_argument(
        "--max_object_videos",
        type=int,
        default=0,
        help=(
            "Maximum number of object-id videos per level. "
            "0 means export all."
        ),
    )
    return parser.parse_args()


def load_mask_stack(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected [N,1,H,W] or [N,H,W], got {arr.shape} at {path}")
    if arr.dtype != np.bool_:
        arr = arr.astype(bool, copy=False)
    return arr


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    if inter <= 0:
        return 0.0
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def containment(mask_a: np.ndarray, area_a: float, mask_b: np.ndarray, area_b: float) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    denom = min(area_a, area_b)
    if denom <= 0:
        return 0.0
    return inter / denom


def greedy_nms(
    masks: np.ndarray,  # [N, H, W] bool
    global_ids: np.ndarray,  # [N] int (1-based source id)
    iou_thr: float,
    contain_thr: float,
) -> np.ndarray:
    n = masks.shape[0]
    if n <= 1:
        return np.ones((n,), dtype=bool)
    areas = masks.reshape(n, -1).sum(axis=1).astype(np.int64)
    # Stable deterministic order for suppression: larger first; tie -> smaller global id first.
    order = np.lexsort((global_ids, -areas))
    keep_flags = np.ones((n,), dtype=bool)
    selected: List[int] = []
    for idx in order:
        if not keep_flags[idx]:
            continue
        m = masks[idx]
        a = float(areas[idx])
        suppressed = False
        for kept in selected:
            mk = masks[kept]
            ak = float(areas[kept])
            ov_iou = iou(m, mk)
            if ov_iou >= iou_thr:
                suppressed = True
                break
            ov_contain = containment(m, a, mk, ak)
            if ov_contain >= contain_thr:
                suppressed = True
                break
        if suppressed:
            keep_flags[idx] = False
        else:
            selected.append(idx)
    return keep_flags


def assignment_order(global_ids: np.ndarray, areas: np.ndarray, prefer_small: bool) -> np.ndarray:
    # Overwrite rule: later assignment wins on overlap.
    # We sort low-priority -> high-priority.
    # Tie-break for equal area: smaller global id has higher priority (wins later).
    if prefer_small:
        # low-priority: larger area / larger id first
        return np.lexsort((-global_ids, -areas))
    # prefer large: low-priority is smaller area; equal area larger id first
    return np.lexsort((-global_ids, areas))


def assign_single_label(
    masks: np.ndarray,  # [N, H, W] bool
    global_ids: np.ndarray,  # [N], 1-based ids
    prefer_small: bool,
) -> np.ndarray:
    h, w = masks.shape[-2], masks.shape[-1]
    out = np.zeros((h, w), dtype=np.int32)
    if masks.shape[0] == 0:
        return out
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.int64)
    order = assignment_order(global_ids, areas, prefer_small=prefer_small)
    for idx in order:
        out[masks[idx]] = int(global_ids[idx])
    return out


def build_slot_mapping(used_global_ids: Sequence[int]) -> Dict[int, int]:
    # 0 reserved for background.
    mapping = {0: 0}
    for slot, gid in enumerate(sorted(int(x) for x in used_global_ids), start=1):
        mapping[int(gid)] = int(slot)
    return mapping


def remap_global_to_slot(global_map: np.ndarray, g2s: Dict[int, int]) -> np.ndarray:
    if global_map.size == 0:
        return global_map.astype(np.int32)
    max_gid = int(global_map.max())
    lut = np.zeros((max_gid + 1,), dtype=np.int32)
    for gid, slot in g2s.items():
        if gid <= max_gid:
            lut[int(gid)] = int(slot)
    out = np.zeros_like(global_map, dtype=np.int32)
    valid = (global_map >= 0) & (global_map <= max_gid)
    out[valid] = lut[global_map[valid]]
    return out


def id_to_color(mask: np.ndarray) -> np.ndarray:
    # Deterministic pseudo-random color by id.
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(mask)
    for gid in ids:
        gid_i = int(gid)
        if gid_i == 0:
            continue
        x = np.uint32(gid_i * 2654435761 % (2**32))
        r = np.uint8((x >> 16) & 255)
        g = np.uint8((x >> 8) & 255)
        b = np.uint8(x & 255)
        rgb[mask == gid_i] = np.array([r, g, b], dtype=np.uint8)
    return rgb


def ensure_video_writer(path: Path, fps: int, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return vw


def _level_output_paths(scene_output_root: Path, level: str):
    level_dir_out = scene_output_root / level
    object_masks_dir = level_dir_out / "object_masks"
    mapping_path = level_dir_out / "global2slot.json"
    meta_path = level_dir_out / "meta.json"
    return object_masks_dir, mapping_path, meta_path


def _check_level_complete(
    scene_output_root: Path,
    level: str,
    level_input_paths: List[Path],
) -> Tuple[bool, List[np.ndarray]]:
    object_masks_dir, mapping_path, meta_path = _level_output_paths(scene_output_root, level)
    if not (object_masks_dir.is_dir() and mapping_path.is_file() and meta_path.is_file()):
        return False, []

    expected_stems = [f"{i + 1:06d}_s" for i in range(len(level_input_paths))]
    slot_maps: List[np.ndarray] = []
    for stem in expected_stems:
        mp = object_masks_dir / f"{stem}.npy"
        if not mp.is_file():
            return False, []
        slot_maps.append(np.load(mp))

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("num_frames", -1)) != len(expected_stems):
            return False, []
        if int(meta.get("K", -1)) < 0:
            return False, []
    except Exception:
        return False, []
    return True, slot_maps


def export_visualizations(
    scene_name: str,
    levels: List[str],
    frame_stacks: List[np.ndarray],  # each [L,H,W] slot id
    output_scene_root: Path,
    rgb_paths: List[Path],
    fps: int,
    max_object_videos: int,
):
    if not frame_stacks:
        return
    l_count, h, w = frame_stacks[0].shape
    if l_count != len(levels):
        raise RuntimeError("Level count mismatch in visualization export.")

    level_frames = {level: [frame[li] for frame in frame_stacks] for li, level in enumerate(levels)}

    for level in levels:
        level_dir = output_scene_root / level / "viz"
        level_dir.mkdir(parents=True, exist_ok=True)
        full_writer = ensure_video_writer(level_dir / "full_mask.mp4", fps=fps, width=w, height=h)
        try:
            for m in level_frames[level]:
                colored = id_to_color(m)
                full_writer.write(cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        finally:
            full_writer.release()

    if not rgb_paths:
        print("[warn] visualization: RGB paths missing; skip per-object videos.")
        return
    if len(rgb_paths) != len(frame_stacks):
        print(
            f"[warn] visualization: rgb frame count {len(rgb_paths)} != mask frame count {len(frame_stacks)}; "
            "truncate to min length."
        )
    n = min(len(rgb_paths), len(frame_stacks))

    for li, level in enumerate(levels):
        level_maps = [frame_stacks[i][li] for i in range(n)]
        all_ids = sorted(set(int(x) for m in level_maps for x in np.unique(m) if int(x) > 0))
        if max_object_videos > 0:
            all_ids = all_ids[:max_object_videos]
        if not all_ids:
            continue
        level_dir = output_scene_root / level / "viz" / "objects"
        level_dir.mkdir(parents=True, exist_ok=True)

        writers = {
            obj_id: ensure_video_writer(
                level_dir / f"object_{obj_id:04d}.mp4",
                fps=fps,
                width=w,
                height=h,
            )
            for obj_id in all_ids
        }
        try:
            for i in range(n):
                rgb = cv2.imread(str(rgb_paths[i]), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise FileNotFoundError(f"Cannot read RGB frame: {rgb_paths[i]}")
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                rgb_dark = (rgb.astype(np.float32) * 0.25).astype(np.uint8)
                m = level_maps[i]
                for obj_id, writer in writers.items():
                    frame = rgb_dark.copy()
                    fg = m == int(obj_id)
                    if fg.any():
                        frame[fg] = rgb[fg]
                    cv2.putText(
                        frame,
                        f"scene={scene_name} level={level} object={obj_id}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    writer.write(frame)
        finally:
            for writer in writers.values():
                writer.release()


def main():
    args = parse_args()
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    if not levels:
        raise ValueError("No valid levels parsed from --levels")
    if len(levels) != 4:
        raise ValueError(f"Expected 4 levels for output stack, got {levels}")

    scene_input_root = Path(args.input_root).expanduser().resolve() / args.scene
    if not scene_input_root.is_dir():
        raise FileNotFoundError(f"Scene input root not found: {scene_input_root}")

    scene_output_root = Path(args.output_root).expanduser().resolve() / args.scene
    scene_output_root.mkdir(parents=True, exist_ok=True)

    # Load per-level frame paths
    level_paths: Dict[str, List[Path]] = {}
    for level in levels:
        lvl_dir = scene_input_root / level / args.level_mask_subdir
        if not lvl_dir.is_dir():
            raise FileNotFoundError(f"Level mask dir not found: {lvl_dir}")
        files = sorted(lvl_dir.glob(args.mask_glob))
        if not files:
            raise RuntimeError(f"No mask files in {lvl_dir} matching {args.mask_glob}")
        level_paths[level] = files
        print(f"[scan] level={level}, frames={len(files)}, sample={files[0].name}")

    frame_count = len(level_paths[levels[0]])
    for level in levels[1:]:
        if len(level_paths[level]) != frame_count:
            raise RuntimeError(
                f"Frame count mismatch: {levels[0]}={frame_count}, {level}={len(level_paths[level])}"
            )

    per_level_results: Dict[str, List[FrameResult]] = {level: [] for level in levels}

    # Process each level independently
    for level in levels:
        already_done, cached_slot_maps = _check_level_complete(
            scene_output_root=scene_output_root,
            level=level,
            level_input_paths=level_paths[level],
        )
        if already_done:
            print(f"[skip] level={level} already complete; reuse existing outputs.")
            for slot_map in cached_slot_maps:
                per_level_results[level].append(
                    FrameResult(
                        assigned_global=np.zeros_like(slot_map, dtype=np.int32),
                        assigned_slot=slot_map.astype(np.int32),
                        used_global_ids=np.array([], dtype=np.int32),
                    )
                )
            continue

        print(f"[process] level={level}")
        frame_global_maps: List[np.ndarray] = []
        used_ids_level: set = set()
        prefer_small = level in PREFER_SMALL_LEVELS
        for path in level_paths[level]:
            stack = load_mask_stack(path)  # [N,H,W] bool
            n_obj = stack.shape[0]
            global_ids = np.arange(1, n_obj + 1, dtype=np.int32)
            areas = stack.reshape(n_obj, -1).sum(axis=1).astype(np.int64)
            valid = areas >= int(args.min_area)
            stack_v = stack[valid]
            ids_v = global_ids[valid]

            if stack_v.shape[0] > 0 and bool(args.enable_nms):
                keep = greedy_nms(
                    masks=stack_v,
                    global_ids=ids_v,
                    iou_thr=float(args.iou_thr),
                    contain_thr=float(args.contain_thr),
                )
                stack_kept = stack_v[keep]
                ids_kept = ids_v[keep]
            else:
                stack_kept = stack_v
                ids_kept = ids_v

            assigned_global = assign_single_label(
                masks=stack_kept,
                global_ids=ids_kept,
                prefer_small=prefer_small,
            )
            used_ids = np.unique(assigned_global)
            used_ids = used_ids[used_ids > 0]
            used_ids_level.update(int(x) for x in used_ids.tolist())
            frame_global_maps.append(assigned_global)

        g2s = build_slot_mapping(sorted(used_ids_level))
        s2g = {int(slot): int(gid) for gid, slot in g2s.items()}

        level_dir_out = scene_output_root / level
        level_dir_out.mkdir(parents=True, exist_ok=True)
        object_masks_dir = level_dir_out / "object_masks"
        object_masks_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = level_dir_out / "global2slot.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scene": args.scene,
                    "level": level,
                    "global_to_slot": {str(k): int(v) for k, v in sorted(g2s.items())},
                    "slot_to_global": {str(k): int(v) for k, v in sorted(s2g.items())},
                    "num_global_ids_kept": int(len(used_ids_level)),
                    "num_slots_fg": int(max(s2g.keys()) if s2g else 0),
                },
                f,
                indent=2,
            )

        for fi, (src_path, gmap) in enumerate(zip(level_paths[level], frame_global_maps)):
            slot_map = remap_global_to_slot(gmap, g2s)
            frame_name = f"{fi + 1:06d}_s.npy"
            np.save(object_masks_dir / frame_name, slot_map.astype(np.int32))
            per_level_results[level].append(
                FrameResult(
                    assigned_global=gmap.astype(np.int32),
                    assigned_slot=slot_map.astype(np.int32),
                    used_global_ids=np.unique(gmap[gmap > 0]).astype(np.int32),
                )
            )

        with open(level_dir_out / "meta.json", "w", encoding="utf-8") as f:
            max_slot_seen = max((int(x.assigned_slot.max()) for x in per_level_results[level]), default=0)
            json.dump(
                {
                    "scene": args.scene,
                    "level": level,
                    "num_frames": len(per_level_results[level]),
                    "K": int(max_slot_seen),
                    "max_slot_seen": int(max_slot_seen),
                    "min_area": int(args.min_area),
                    "enable_nms": bool(args.enable_nms),
                    "iou_thr": float(args.iou_thr),
                    "contain_thr": float(args.contain_thr),
                    "conflict_policy": "prefer_small" if prefer_small else "prefer_large",
                },
                f,
                indent=2,
            )

        print(
            f"[done] level={level} frames={len(per_level_results[level])} "
            f"fg_global={len(used_ids_level)} fg_slot_max={max((g2s.values()), default=0)}"
        )

    # Build in-memory stacked [4,H,W] per frame (for optional visualization only).
    frame_stacks: List[np.ndarray] = []
    for fi in range(frame_count):
        ch = [per_level_results[level][fi].assigned_slot for level in levels]
        out = np.stack(ch, axis=0).astype(np.int32)
        frame_stacks.append(out)

    if args.visualize:
        rgb_paths: List[Path] = []
        if args.rgb_root:
            rgb_dir = Path(args.rgb_root).expanduser().resolve() / args.scene / "rgb" / "2x"
            if not rgb_dir.is_dir():
                raise FileNotFoundError(f"RGB dir not found for visualization: {rgb_dir}")
            rgb_paths = sorted(rgb_dir.glob("*.png"))
        export_visualizations(
            scene_name=args.scene,
            levels=levels,
            frame_stacks=frame_stacks,
            output_scene_root=scene_output_root,
            rgb_paths=rgb_paths,
            fps=int(args.viz_fps),
            max_object_videos=int(args.max_object_videos),
        )
        print(f"[done] visualization exported under {scene_output_root}")


if __name__ == "__main__":
    main()
