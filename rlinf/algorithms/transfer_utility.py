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

from typing import Optional

import torch


class TransferUtilityMonitor:
    """Track transfer-risk signals and compute stage-wise transfer utility."""

    METRIC_PREFIX = "transfer_risk"

    def __init__(self, transfer_utility_cfg):
        self.enabled = bool(getattr(transfer_utility_cfg, "enabled", False))
        self.log_interval = int(getattr(transfer_utility_cfg, "log_interval", 1))
        self.auto_stop = bool(getattr(transfer_utility_cfg, "auto_stop", False))

        self.c_m = float(getattr(transfer_utility_cfg, "c_m", 1.0))
        self.c_d = float(getattr(transfer_utility_cfg, "c_d", 0.1))
        self.c_h = float(getattr(transfer_utility_cfg, "c_h", 0.1))
        self.xi = float(getattr(transfer_utility_cfg, "xi", 0.0))

        self.lambda_out = float(getattr(transfer_utility_cfg, "lambda_out", 1.0))
        self.lambda_lat = float(getattr(transfer_utility_cfg, "lambda_lat", 0.0))
        self.f1_traj = float(getattr(transfer_utility_cfg, "f1_traj", 1.0))
        self.e_latent = float(getattr(transfer_utility_cfg, "e_latent", 0.0))
        # WAN WM is fixed during policy training, so epsilon is a constant.
        self.epsilon_wm = self.lambda_out * (1.0 - self.f1_traj) + self.lambda_lat * self.e_latent

        self.s_wm: float = 0.0
        self.delta_s_wm: float = 0.0
        self.d_beh: float = 0.0
        self.l_hold: float = 0.0
        self.delta_l_hold: float = 0.0
        self.transfer_risk: float = 0.0
        self.transfer_utility: float = 0.0
        self.should_stop: bool = False

        self._prev_success_rate: Optional[float] = None
        self._prev_l_hold: Optional[float] = None

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        if mask is None:
            return float(values.float().mean().item())
        valid = mask.to(dtype=torch.bool)
        if valid.shape != values.shape:
            while len(valid.shape) < len(values.shape):
                valid = valid.unsqueeze(-1)
            valid = valid.expand_as(values)
        valid_values = values[valid]
        if valid_values.numel() == 0:
            return 0.0
        return float(valid_values.float().mean().item())

    def should_log(self, step: int) -> bool:
        if self.log_interval <= 1:
            return True
        return step % self.log_interval == 0

    def update_success_rate(self, success_rate: float, step: int) -> dict[str, float]:
        del step
        self.s_wm = float(success_rate)
        if self._prev_success_rate is None:
            self.delta_s_wm = 0.0
        else:
            self.delta_s_wm = self.s_wm - self._prev_success_rate
        self._prev_success_rate = self.s_wm
        return {
            f"{self.METRIC_PREFIX}/s_wm": self.s_wm,
            f"{self.METRIC_PREFIX}/delta_s_wm": self.delta_s_wm,
        }

    def update_behavior_drift(
        self,
        curr_logprobs: Optional[torch.Tensor] = None,
        ref_logprobs: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        kl_value: Optional[float] = None,
    ) -> dict[str, float]:
        if kl_value is not None:
            self.d_beh = float(kl_value)
        elif curr_logprobs is not None and ref_logprobs is not None:
            self.d_beh = self._masked_mean(curr_logprobs - ref_logprobs, loss_mask)
        return {f"{self.METRIC_PREFIX}/d_beh": self.d_beh}

    def update_holdout_loss(self, holdout_loss: float) -> dict[str, float]:
        self.l_hold = float(holdout_loss)
        if self._prev_l_hold is None:
            self.delta_l_hold = 0.0
        else:
            self.delta_l_hold = self.l_hold - self._prev_l_hold
        self._prev_l_hold = self.l_hold
        return {
            f"{self.METRIC_PREFIX}/l_hold": self.l_hold,
            f"{self.METRIC_PREFIX}/delta_l_hold": self.delta_l_hold,
        }

    def compute_transfer_utility(self) -> dict[str, float]:
        self.transfer_risk = 2.0 * self.c_m * self.epsilon_wm + self.c_d * self.d_beh + self.c_h * self.l_hold
        self.transfer_utility = self.delta_s_wm - self.transfer_risk
        self.should_stop = self.transfer_utility <= self.xi
        return {
            f"{self.METRIC_PREFIX}/epsilon_wm": self.epsilon_wm,
            f"{self.METRIC_PREFIX}/transfer_penalty": self.transfer_risk,
            f"{self.METRIC_PREFIX}/transfer_utility": self.transfer_utility,
            f"{self.METRIC_PREFIX}/should_stop": float(self.should_stop),
        }

    def get_all_metrics(self) -> dict[str, float]:
        metrics = {}
        metrics.update(
            {
                f"{self.METRIC_PREFIX}/s_wm": self.s_wm,
                f"{self.METRIC_PREFIX}/delta_s_wm": self.delta_s_wm,
                f"{self.METRIC_PREFIX}/d_beh": self.d_beh,
                f"{self.METRIC_PREFIX}/l_hold": self.l_hold,
                f"{self.METRIC_PREFIX}/delta_l_hold": self.delta_l_hold,
                f"{self.METRIC_PREFIX}/epsilon_wm": self.epsilon_wm,
                f"{self.METRIC_PREFIX}/transfer_penalty": self.transfer_risk,
                f"{self.METRIC_PREFIX}/transfer_utility": self.transfer_utility,
                f"{self.METRIC_PREFIX}/should_stop": float(self.should_stop),
            }
        )
        return metrics
