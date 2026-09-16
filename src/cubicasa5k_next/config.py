"""Typed configuration containers for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Hourglass output layout aligned to past behavior (21+12+11=44)."""

    num_heatmap_channels: int = 21
    num_room_classes: int = 12
    num_icon_classes: int = 11

    @property
    def num_outputs(self) -> int:
        return self.num_heatmap_channels + self.num_room_classes + self.num_icon_classes


@dataclass(frozen=True)
class AugmentationConfig:
    image_size: int = 256
    use_rotation_augmentation: bool = True
    use_color_jitter: bool = True
    gaussian_radius: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 20
    max_epochs: int = 400
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    scheduler_patience: int = 20
    scheduler_factor: float = 0.1
    num_workers: int = 4
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    tensorboard_dir: str | None = None
    tensorboard_run_name: str = "cubicasa5k-next"
    tensorboard_image_every: int = 5
    tensorboard_max_images: int = 4
    tensorboard_flush_secs: int = 30


@dataclass(frozen=True)
class InferenceConfig:
    image_size: int = 256
    junction_threshold: float = 0.4
    alignment_tolerance_px: int = 10
    use_test_time_rotation: bool = False
    device: str = "cuda"
