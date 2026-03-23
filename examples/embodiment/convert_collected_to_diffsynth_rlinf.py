#!/usr/bin/env python3
"""Convert RLinf collected pickle episodes to DiffSynth RLinfNpyDataset format.

Target format (consumed by diffsynth-studio `RLinfNpyDataset`):

output_dir/
  train_data/
    step_0/
      seed_0/
        rgb.npy         # [T, N, H, W, 3] (N defaults to 50)
        actions.npy     # [T, N, action_dim] (N defaults to 50)
        success.npy     # [N]
        instruction.npy # [N]
  val_data/
    step_0/
      seed_x/
        ...
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable, **kwargs):
        return iterable


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    # Torch Tensor support without hard dependency.
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_uint8_hwc(image: Any) -> np.ndarray:
    arr = _to_numpy(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape={arr.shape}")

    # Convert CHW -> HWC if needed.
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels after conversion, got shape={arr.shape}")

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    return arr


def _extract_instruction(observations: list[dict[str, Any]]) -> str:
    for obs in observations:
        if isinstance(obs, dict) and "task_descriptions" in obs:
            value = obs["task_descriptions"]
            if isinstance(value, (list, tuple)):
                return str(value[0]) if len(value) > 0 else ""
            return str(value)
    return ""


@dataclass
class EpisodeData:
    rgb: np.ndarray  # [T, N, H, W, 3]
    actions: np.ndarray  # [T, N, action_dim]
    success: np.ndarray  # [N]
    instruction: np.ndarray  # [N]


def _load_episode(pkl_path: str, image_key: str) -> EpisodeData:
    with open(pkl_path, "rb") as f:
        ep = pickle.load(f)

    observations = ep.get("observations", [])
    actions = ep.get("actions", [])
    if len(actions) == 0:
        raise ValueError("Empty actions")
    if len(observations) == 0:
        raise ValueError("Empty observations")

    # The collector records reset obs first, so observations is often T+1.
    time_steps = min(len(actions), len(observations))
    observations = observations[:time_steps]
    actions = actions[:time_steps]

    frames = []
    for obs in observations:
        if not isinstance(obs, dict):
            raise ValueError("Observation item is not a dict")
        image = obs.get(image_key)
        if image is None:
            image = obs.get("image", obs.get("full_image", obs.get("main_images")))
        if image is None:
            raise ValueError(f"Cannot find image key in observation (requested `{image_key}`)")
        frames.append(_to_uint8_hwc(image))

    rgb = np.stack(frames, axis=0)[:, None, ...]  # [T, 1, H, W, 3]

    action_arr = np.stack([_to_numpy(a).astype(np.float32) for a in actions], axis=0)
    if action_arr.ndim == 1:
        action_arr = action_arr[:, None]
    actions_out = action_arr[:, None, :]  # [T, 1, action_dim]

    success_flag = bool(ep.get("success", False))
    success = np.array([int(success_flag)], dtype=np.int64)
    instruction = np.array([_extract_instruction(observations)], dtype=np.str_)

    return EpisodeData(rgb=rgb, actions=actions_out, success=success, instruction=instruction)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_episode(dst_seed_dir: str, data: EpisodeData, overwrite: bool) -> None:
    _ensure_dir(dst_seed_dir)
    targets = {
        "rgb.npy": data.rgb,
        "actions.npy": data.actions,
        "success.npy": data.success,
        "instruction.npy": data.instruction,
    }
    for name, arr in targets.items():
        out_path = os.path.join(dst_seed_dir, name)
        if (not overwrite) and os.path.exists(out_path):
            continue
        np.save(out_path, arr)


def _merge_episodes(episodes: list[EpisodeData]) -> EpisodeData:
    if len(episodes) == 0:
        raise ValueError("Cannot merge empty episodes list")

    base_hwc = episodes[0].rgb.shape[2:]
    base_action_dim = episodes[0].actions.shape[2]
    min_t = min(ep.rgb.shape[0] for ep in episodes)
    if min_t <= 0:
        raise ValueError("Merged episodes have non-positive time steps")

    rgb_list = []
    actions_list = []
    success_list = []
    instruction_list = []
    for ep in episodes:
        if ep.rgb.shape[2:] != base_hwc:
            raise ValueError(
                f"Inconsistent image shape in seed batch: {ep.rgb.shape[2:]} vs {base_hwc}"
            )
        if ep.actions.shape[2] != base_action_dim:
            raise ValueError(
                f"Inconsistent action dim in seed batch: {ep.actions.shape[2]} vs {base_action_dim}"
            )
        rgb_list.append(ep.rgb[:min_t])
        actions_list.append(ep.actions[:min_t])
        success_list.append(ep.success)
        instruction_list.append(ep.instruction)

    rgb = np.concatenate(rgb_list, axis=1)  # [T, N, H, W, 3]
    actions = np.concatenate(actions_list, axis=1)  # [T, N, action_dim]
    success = np.concatenate(success_list, axis=0)  # [N]
    instruction = np.concatenate(instruction_list, axis=0)  # [N]
    return EpisodeData(rgb=rgb, actions=actions, success=success, instruction=instruction)


def _collect_episode_files(input_dir: str, only_success: bool) -> list[str]:
    files = []
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".pkl"):
            continue
        if only_success and "_success.pkl" not in name:
            continue
        files.append(os.path.join(input_dir, name))
    return files


def _convert_list(
    episode_files: list[str],
    split_out_dir: str,
    start_seed_idx: int,
    image_key: str,
    overwrite: bool,
    episodes_per_seed: int,
    progress_desc: str,
) -> tuple[int, int]:
    step_dir = os.path.join(split_out_dir, "step_0")
    _ensure_dir(step_dir)

    ok = 0
    skipped = 0
    if episodes_per_seed <= 0:
        raise ValueError("--episodes_per_seed must be > 0.")

    total = len(episode_files)
    chunk_count = (total + episodes_per_seed - 1) // episodes_per_seed
    for chunk_idx in tqdm(range(chunk_count), desc=progress_desc):
        start = chunk_idx * episodes_per_seed
        end = min(start + episodes_per_seed, total)
        pkl_chunk = episode_files[start:end]
        seed_dir = os.path.join(step_dir, f"seed_{start_seed_idx + chunk_idx}")

        loaded_eps: list[EpisodeData] = []
        for pkl_path in pkl_chunk:
            try:
                loaded_eps.append(_load_episode(pkl_path, image_key=image_key))
                ok += 1
            except Exception as exc:  # pragma: no cover - best effort conversion
                skipped += 1
                print(f"[skip] {os.path.basename(pkl_path)}: {exc}")

        if len(loaded_eps) == 0:
            continue

        try:
            merged = _merge_episodes(loaded_eps)
            _write_episode(seed_dir, merged, overwrite=overwrite)
        except Exception as exc:  # pragma: no cover - best effort conversion
            skipped += len(loaded_eps)
            ok -= len(loaded_eps)
            print(f"[skip_seed] {os.path.basename(seed_dir)}: {exc}")
    return ok, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=str, required=True, help="Source directory containing episode_*.pkl files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument(
        "--val_input_dir",
        type=str,
        default=None,
        help="Optional separate source directory for validation episodes.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio when --val_input_dir is not provided.")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit total episodes from input for quick testing.")
    parser.add_argument("--image_key", type=str, default="main_images", help="Observation image key to use.")
    parser.add_argument("--only_success", action="store_true", help="Use only *_success.pkl episodes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing npy files.")
    parser.add_argument(
        "--episodes_per_seed",
        type=int,
        default=50,
        help="How many trajectories to pack into one seed directory.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")
    if args.val_input_dir is not None and not os.path.isdir(args.val_input_dir):
        raise FileNotFoundError(f"val_input_dir not found: {args.val_input_dir}")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val_ratio must be in [0, 1).")

    train_data_dir = os.path.join(args.output_dir, "train_data")
    val_data_dir = os.path.join(args.output_dir, "val_data")
    _ensure_dir(train_data_dir)
    _ensure_dir(val_data_dir)

    all_train_candidates = _collect_episode_files(args.input_dir, only_success=args.only_success)
    if args.max_episodes is not None:
        all_train_candidates = all_train_candidates[: args.max_episodes]

    if args.val_input_dir is not None:
        train_files = all_train_candidates
        val_files = _collect_episode_files(args.val_input_dir, only_success=args.only_success)
        if args.max_episodes is not None:
            val_files = val_files[: args.max_episodes]
    else:
        split_idx = int(len(all_train_candidates) * (1.0 - args.val_ratio))
        split_idx = max(split_idx, 1) if len(all_train_candidates) > 1 else split_idx
        train_files = all_train_candidates[:split_idx]
        val_files = all_train_candidates[split_idx:]

    print(f"train episodes: {len(train_files)}")
    print(f"val episodes:   {len(val_files)}")

    ok_train, skip_train = _convert_list(
        train_files,
        split_out_dir=train_data_dir,
        start_seed_idx=0,
        image_key=args.image_key,
        overwrite=args.overwrite,
        episodes_per_seed=args.episodes_per_seed,
        progress_desc="Converting train",
    )
    ok_val, skip_val = _convert_list(
        val_files,
        split_out_dir=val_data_dir,
        start_seed_idx=0,
        image_key=args.image_key,
        overwrite=args.overwrite,
        episodes_per_seed=args.episodes_per_seed,
        progress_desc="Converting val",
    )

    print("=== conversion done ===")
    print(f"train: ok={ok_train}, skipped={skip_train}")
    print(f"val:   ok={ok_val}, skipped={skip_val}")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
