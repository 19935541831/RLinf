# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import torch


class HoldoutSuccessDataset:
    """Held-out trajectories stored as per-episode pkl files.

    Compatible with the ``collected_data_libero_openvlaoft`` layout where each
    episode is a single pickle file:

        base_path/
            rank_{R}_env_{E}_episode_{EP}_{success|fail}.pkl

    Each pkl contains::

        {
            "rank": int,
            "env_idx": int,
            "episode_id": int,
            "success": bool,
            "observations": list[dict],   # len = T + 1
            "actions": list[np.ndarray],  # len = T, each shape (action_dim,)
            ...
        }

    ``observations[t]`` is a dict with keys:
        ``main_images``      – Tensor [H, W, C] uint8
        ``wrist_images``     – Tensor [H, W, C] uint8  (optional)
        ``states``           – Tensor [state_dim]
        ``task_descriptions`` – str
    """

    def __init__(
        self,
        data_dir: str,
        success_reward_threshold: float = 0.9,
        enable_kir: bool = True,
    ) -> None:
        del enable_kir, success_reward_threshold
        self.base_path = data_dir
        self.episode_paths: list[str] = []
        self._build_samples()

    def _build_samples(self) -> None:
        if not os.path.isdir(self.base_path):
            raise ValueError(f"Holdout data directory does not exist: {self.base_path}")

        for fname in sorted(os.listdir(self.base_path)):
            if not fname.endswith(".pkl"):
                continue
            # Only keep successful episodes.
            if "_success" not in fname:
                continue
            full_path = os.path.join(self.base_path, fname)
            if os.path.isfile(full_path):
                self.episode_paths.append(full_path)

        if len(self.episode_paths) == 0:
            raise ValueError(
                f"No successful episode pkl files found under {self.base_path}"
            )

    def __len__(self) -> int:
        return len(self.episode_paths)

    @staticmethod
    def _image_to_chw_float(img: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert an HWC uint8 image to CHW float32 in [0, 1]."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.float()
        if img.max() > 1.5:
            img = img / 255.0
        img = img.clamp(0.0, 1.0)
        if img.ndim == 3 and img.shape[-1] in (1, 3):
            img = img.permute(2, 0, 1)
        return img.contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        pkl_path = self.episode_paths[index]
        with open(pkl_path, "rb") as f:
            episode = pickle.load(f)

        observations: list[dict] = episode["observations"]
        actions: list = episode["actions"]
        T = len(actions)
        if T <= 0:
            raise ValueError(f"Empty action sequence in {pkl_path}")

        t = int(np.random.randint(0, T))
        obs_t = observations[t]

        main_image = obs_t.get("main_images")
        if main_image is None:
            raise ValueError(f"Missing 'main_images' in observation at t={t} of {pkl_path}")
        image = self._image_to_chw_float(main_image)

        wrist_image = obs_t.get("wrist_images", None)
        if wrist_image is not None:
            wrist_image = self._image_to_chw_float(wrist_image)

        state = obs_t.get("states", None)
        if state is not None:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            else:
                state = state.float()
        else:
            state = torch.zeros(8, dtype=torch.float32)

        action_raw = actions[t]
        if isinstance(action_raw, np.ndarray):
            action = torch.from_numpy(action_raw).float()
        elif isinstance(action_raw, torch.Tensor):
            action = action_raw.float()
        else:
            action = torch.tensor(action_raw, dtype=torch.float32)

        task = str(obs_t.get("task_descriptions", ""))

        result: dict[str, Any] = {
            "observation": {
                "image": image,
                "observation.state": state,
            },
            "action": action,
            "task": task,
            "episode_index": index,
            "data_path": pkl_path,
            "env_id": int(episode.get("env_idx", 0)),
            "timestep": t,
        }
        if wrist_image is not None:
            result["observation"]["wrist_image"] = wrist_image

        return result

    def sample_batch(self, batch_size: int) -> dict[str, Any]:
        if len(self) == 0:
            raise ValueError("HoldoutSuccessDataset is empty.")
        indices = torch.randint(low=0, high=len(self), size=(batch_size,))
        batch = [self[int(i)] for i in indices]
        actions = torch.stack([it["action"] for it in batch], dim=0)
        return {
            "items": batch,
            "actions": actions,
        }
