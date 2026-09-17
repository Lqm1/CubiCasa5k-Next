"""Pre-activation hourglass with 44-channel output (21+12+11).

Single-head symmetric encoder-decoder: 21 junction heatmaps followed by 12
room logits and 11 icon logits, with the first 21 channels passed through a
sigmoid. The module layout below is written originally for this project.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cubicasa5k_next.config import InferenceConfig


class PreActivationBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = out_channels // 2
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True)
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shortcut = features
        activated = F.relu(self.bn(features), inplace=True)
        reduced = F.relu(self.bn1(self.conv1(activated)), inplace=True)
        refined = F.relu(self.bn2(self.conv2(reduced)), inplace=True)
        expanded = self.conv3(refined)
        if self.needs_projection:
            shortcut = self.conv4(features)  # type: ignore[attr-defined]
        return expanded + shortcut


class FloorplanHourglass(nn.Module):
    # Submodule attribute names below double as checkpoint tensor keys, so
    # saved weights load directly with ``load_state_dict``.
    def __init__(self, num_outputs: int = 44) -> None:
        super().__init__()
        self.conv1_ = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.r01 = PreActivationBottleneck(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r02 = PreActivationBottleneck(128, 128)
        self.r03 = PreActivationBottleneck(128, 128)
        self.r04 = PreActivationBottleneck(128, 256)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r11_a = PreActivationBottleneck(256, 256)
        self.r12_a = PreActivationBottleneck(256, 256)
        self.r13_a = PreActivationBottleneck(256, 256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r21_a = PreActivationBottleneck(256, 256)
        self.r22_a = PreActivationBottleneck(256, 256)
        self.r23_a = PreActivationBottleneck(256, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r31_a = PreActivationBottleneck(256, 256)
        self.r32_a = PreActivationBottleneck(256, 256)
        self.r33_a = PreActivationBottleneck(256, 256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r41_a = PreActivationBottleneck(256, 256)
        self.r42_a = PreActivationBottleneck(256, 256)
        self.r43_a = PreActivationBottleneck(256, 256)
        self.r44_a = PreActivationBottleneck(256, 512)
        self.r45_a = PreActivationBottleneck(512, 512)
        self.upsample4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, bias=True)
        self.r41_b = PreActivationBottleneck(256, 256)
        self.r42_b = PreActivationBottleneck(256, 256)
        self.r43_b = PreActivationBottleneck(256, 512)
        self.r4_ = PreActivationBottleneck(512, 512)
        self.upsample3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, bias=True)
        self.r31_b = PreActivationBottleneck(256, 256)
        self.r32_b = PreActivationBottleneck(256, 256)
        self.r33_b = PreActivationBottleneck(256, 512)
        self.r3_ = PreActivationBottleneck(512, 512)
        self.upsample2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, bias=True)
        self.r21_b = PreActivationBottleneck(256, 256)
        self.r22_b = PreActivationBottleneck(256, 256)
        self.r23_b = PreActivationBottleneck(256, 512)
        self.r2_ = PreActivationBottleneck(512, 512)
        self.upsample1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, bias=True)
        self.r11_b = PreActivationBottleneck(256, 256)
        self.r12_b = PreActivationBottleneck(256, 256)
        self.r13_b = PreActivationBottleneck(256, 512)
        self.conv2_ = nn.Conv2d(512, 512, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3_ = nn.Conv2d(512, 256, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4_ = nn.Conv2d(256, num_outputs, kernel_size=1, bias=True)
        self.upsample = nn.ConvTranspose2d(
            num_outputs, num_outputs, kernel_size=4, stride=4, bias=True
        )
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _add_upsampled(upper: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        _, _, target_h, target_w = lateral.shape
        if upper.shape != lateral.shape:
            upper = F.interpolate(
                upper, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        return upper + lateral

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.relu1(self.bn1(self.conv1_(image)))
        features = self.maxpool(features)
        features = self.r04(self.r03(self.r02(self.r01(features))))
        high_branch = features
        low = self.maxpool1(features)
        low = self.r13_a(self.r12_a(self.r11_a(low)))
        high_out = self.r13_b(self.r12_b(self.r11_b(high_branch)))
        mid_low = self.maxpool2(low)
        mid_low = self.r23_a(self.r22_a(self.r21_a(mid_low)))
        mid_high = self.r23_b(self.r22_b(self.r21_b(low)))
        deep_low = self.maxpool3(mid_low)
        deep_low = self.r33_a(self.r32_a(self.r31_a(deep_low)))
        deep_high = self.r33_b(self.r32_b(self.r31_b(mid_low)))
        bottom = self.maxpool4(deep_low)
        bottom = self.r45_a(self.r44_a(self.r43_a(self.r42_a(self.r41_a(bottom)))))
        side = self.r43_b(self.r42_b(self.r41_b(deep_low)))
        merged = self.r4_(self._add_upsampled(self.upsample4(bottom), side))
        merged = self.r3_(self._add_upsampled(self.upsample3(merged), deep_high))
        merged = self.r2_(self._add_upsampled(self.upsample2(merged), mid_high))
        merged = self._add_upsampled(self.upsample1(merged), high_out)
        merged = self.relu2(self.bn2(self.conv2_(merged)))
        merged = self.relu3(self.bn3(self.conv3_(merged)))
        merged = self.upsample(self.conv4_(merged))
        merged[:, :21] = self.sigmoid(merged[:, :21])
        return merged


def run_hourglass_inference(
    checkpoint: str | Path,
    image_path: str | Path,
    config: InferenceConfig | None = None,
    num_heatmap_channels: int = 21,
    num_room_classes: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from cubicasa5k_next.utils.checkpoint import load_checkpoint

    active = config or InferenceConfig()
    model, _ = load_checkpoint(checkpoint, device=torch.device("cpu"))
    model.eval()
    total = int(model.conv4_.out_channels)
    heatmaps_ch = min(num_heatmap_channels, total)
    rooms_ch = min(num_room_classes, max(total - heatmaps_ch, 0))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = active.image_size
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    scaled = 2.0 * (resized.astype(np.float32) / 255.0) - 1.0
    tensor = torch.from_numpy(scaled).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        raw = model(tensor)
        if raw.shape[-2:] != (size, size):
            raw = F.interpolate(raw, size=(size, size), mode="bilinear", align_corners=False)
    room_logits = raw[0, heatmaps_ch : heatmaps_ch + rooms_ch].numpy()
    icon_logits = raw[0, heatmaps_ch + rooms_ch :].numpy()
    heatmaps = raw[0, :heatmaps_ch].numpy()
    return room_logits, icon_logits, heatmaps
