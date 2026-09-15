"""Homoscedastic-uncertainty weighting for the multi-task objective.

Implements::

    heat: sum_i [ exp(-log_var_i) * MSE_i + softplus(log_var_i) ]
    seg:  CE(logits * exp(-log_var_k), target)

with one learnable scale per heatmap channel (21) plus one per segmentation
task (room, icon).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as loss_functional


@dataclass
class MultiTaskLossOutput:
    total: torch.Tensor
    heatmap_term: torch.Tensor
    segmentation_term: torch.Tensor
    room_cross_entropy: torch.Tensor
    icon_cross_entropy: torch.Tensor
    heatmap_mse: torch.Tensor


class UncertaintyWeightedLoss(nn.Module):
    # Parameter names double as checkpoint tensor keys for direct loading.
    def __init__(self, num_heatmap_channels: int = 21) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.log_vars_mse = nn.Parameter(torch.zeros(num_heatmap_channels, dtype=torch.float32))

    def forward(
        self,
        predicted_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        predicted_room_logits: torch.Tensor,
        target_room_labels: torch.Tensor,
        predicted_icon_logits: torch.Tensor,
        target_icon_labels: torch.Tensor,
    ) -> MultiTaskLossOutput:
        room_targets = target_room_labels.long()
        icon_targets = target_icon_labels.long()

        plain_room_ce = loss_functional.cross_entropy(predicted_room_logits, room_targets)
        plain_icon_ce = loss_functional.cross_entropy(predicted_icon_logits, icon_targets)
        scaled_room_ce = loss_functional.cross_entropy(
            predicted_room_logits * torch.exp(-self.log_vars[0]), room_targets
        )
        scaled_icon_ce = loss_functional.cross_entropy(
            predicted_icon_logits * torch.exp(-self.log_vars[1]), icon_targets
        )

        batch_count, _, height, width = predicted_heatmaps.shape
        flat_pred = predicted_heatmaps.permute(0, 2, 3, 1).reshape(-1, predicted_heatmaps.shape[1])
        flat_target = target_heatmaps.permute(0, 2, 3, 1).reshape(-1, target_heatmaps.shape[1])
        squared = (flat_pred - flat_target) ** 2
        per_task_mse = squared.sum(dim=0) / max(batch_count * height * width, 1)
        weighted_heatmap = (
            torch.exp(-self.log_vars_mse) * per_task_mse
            + torch.nn.functional.softplus(self.log_vars_mse)
        ).sum()

        segmentation_term = scaled_room_ce + scaled_icon_ce
        total = segmentation_term + weighted_heatmap
        return MultiTaskLossOutput(
            total=total,
            heatmap_term=weighted_heatmap,
            segmentation_term=segmentation_term,
            room_cross_entropy=plain_room_ce.detach(),
            icon_cross_entropy=plain_icon_ce.detach(),
            heatmap_mse=per_task_mse.detach().mean(),
        )

    def forward_stacked_tensors(
        self, raw_outputs: torch.Tensor, label_tensor: torch.Tensor
    ) -> MultiTaskLossOutput:
        """Accept 44ch outputs and 23ch labels ([21,1,1] split)."""
        heat_pred, room_logits, icon_logits = torch.split(raw_outputs, [21, 12, 11], dim=1)
        heat_target, room_target, icon_target = torch.split(label_tensor, [21, 1, 1], dim=1)
        return self.forward(
            heat_pred,
            heat_target,
            room_logits,
            room_target.squeeze(1),
            icon_logits,
            icon_target.squeeze(1),
        )
