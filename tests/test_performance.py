import math
import pickle

import cv2
import numpy as np
import torch

from cubicasa5k_next.config import AugmentationConfig, InferenceConfig, TrainingConfig
from cubicasa5k_next.data.dataset import SvgFloorplanDataset
from cubicasa5k_next.data.heatmaps import _gaussian_kernel
from cubicasa5k_next.data.transforms import to_normalized_tensor
from cubicasa5k_next.inference.predictor import FloorplanPredictor, split_raw_output
from cubicasa5k_next.target_cache import TargetCache
from cubicasa5k_next.training.trainer import train_model


def test_cache_is_bounded_and_returns_copies():
    cache = TargetCache(1)
    source = np.ones((512, 1024), dtype=np.uint8)
    cache.put((1,), (source,))
    source[:] = 7
    assert cache.get((1,))[0][0, 0] == 1
    cache.get((1,))[0][:] = 9
    assert cache.get((1,))[0][0, 0] == 1
    cache.put((2,), (source,))
    cache.put((3,), (source,))
    assert cache.get((1,)) is None
    assert cache.bytes <= cache.max_bytes
    assert not pickle.loads(pickle.dumps(cache)).entries
    disabled = TargetCache(0)
    disabled.put((1,), (source,))
    assert disabled.get((1,)) is None


def make_dataset(tmp_path, cache_mb=256):
    image = tmp_path / "image.png"
    annotation = tmp_path / "model.svg"
    cv2.imwrite(str(image), np.full((64, 64, 3), 200, dtype=np.uint8))
    annotation.write_text(
        '<svg width="64" height="64">'
        '<g class="Space LivingRoom"><polygon points="4,4 52,4 52,52 4,52"/></g>'
        '<g id="Wall"><polygon points="4,4 52,4 52,10 4,10"/></g>'
        '<g id="Door"><polygon points="20,4 30,4 30,10 20,10"/></g>'
        "</svg>"
    )
    return SvgFloorplanDataset(
        [image, image],
        [annotation, annotation],
        AugmentationConfig(image_size=64),
        training=False,
        target_cache_mb=cache_mb,
    )


def test_target_cache_parity_and_invalidation(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    first = dataset[0]
    from cubicasa5k_next.data import dataset as module

    original = module.parse_floorplan_svg
    monkeypatch.setattr(
        module,
        "parse_floorplan_svg",
        lambda *_: (_ for _ in ()).throw(AssertionError("cache missed")),
    )
    second = dataset[0]
    for name in ("image", "room_labels", "icon_labels", "heatmaps"):
        assert torch.equal(getattr(first, name), getattr(second, name))
    monkeypatch.setattr(module, "parse_floorplan_svg", original)
    dataset.annotation_paths[0].write_text('<svg width="64" height="64"></svg>')
    assert not torch.equal(dataset[0].room_labels, first.room_labels)


def test_gaussian_kernel_bitwise_parity():
    radius = 5
    reference = np.zeros((11, 11), dtype=np.float32)
    for y in range(11):
        for x in range(11):
            reference[y, x] = math.exp(
                -((x - radius) ** 2 + (y - radius) ** 2) / (2 * (radius / 2 + 1e-6) ** 2)
            )
    assert np.array_equal(_gaussian_kernel(radius), reference)
    assert _gaussian_kernel(radius) is _gaussian_kernel(radius)


def test_tta_matches_original_sequential_numpy():
    torch.manual_seed(5)
    model = torch.nn.Conv2d(3, 44, 1).eval()
    image = np.random.default_rng(2).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    reference = None
    with torch.no_grad():
        for k in range(4):
            tensor = to_normalized_tensor(np.rot90(image, k).copy()).unsqueeze(0)
            raw = np.rot90(model(tensor)[0].numpy(), -k, axes=(1, 2)).copy()
            reference = raw if reference is None else reference + raw
    for batch_size in (1, 2, 4):
        predictor = FloorplanPredictor(
            model,
            InferenceConfig(image_size=32, use_test_time_rotation=True, tta_batch_size=batch_size),
            torch.device("cpu"),
        )
        for actual, expected in zip(
            predictor.predict_arrays(image), split_raw_output(reference / 4), strict=True
        ):
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_real_hourglass_training_checkpoint_and_inference(tmp_path):
    torch.set_num_threads(2)
    dataset = make_dataset(tmp_path)
    config = TrainingConfig(
        batch_size=2,
        max_epochs=1,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        tensorboard_dir=None,
    )
    best = train_model(dataset, dataset, config, torch.device("cpu"))
    predictor = FloorplanPredictor.from_checkpoint(
        best, InferenceConfig(image_size=64), torch.device("cpu")
    )
    outputs = predictor.predict_arrays(np.zeros((64, 64, 3), dtype=np.uint8))
    assert [array.shape for array in outputs] == [(12, 64, 64), (11, 64, 64), (21, 64, 64)]
    assert all(np.isfinite(array).all() for array in outputs)


def test_persistent_workers_and_zero_workers(tmp_path):
    from dataclasses import replace

    from cubicasa5k_next.training.trainer import build_dataloaders

    dataset = make_dataset(tmp_path)
    config = TrainingConfig(batch_size=2, num_workers=1, persistent_workers=True)
    train, valid = build_dataloaders(dataset, dataset, config, torch.device("cpu"))
    first = next(iter(valid))
    second = next(iter(valid))
    assert torch.equal(first["heatmaps"], second["heatmaps"])
    assert valid.persistent_workers
    train_zero, _ = build_dataloaders(
        dataset, dataset, replace(config, num_workers=0), torch.device("cpu")
    )
    assert not train_zero.persistent_workers
    assert next(iter(train_zero))["image"].shape == (2, 3, 64, 64)
