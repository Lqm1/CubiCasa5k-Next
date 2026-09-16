import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from cubicasa5k_next.tensorboard import create_writer, should_log_images
from cubicasa5k_next.training.trainer import _confusion_scores


def test_confusion_metrics_include_false_positive_only_class():
    # Two correct class-0 pixels and two incorrectly predicted as class 1.
    miou, accuracy = _confusion_scores(torch.tensor([[2, 2], [0, 0]]))
    assert miou == 0.25
    assert accuracy == 0.5
    assert _confusion_scores(torch.zeros(2, 2)) == (0.0, 0.0)


def test_training_events_and_resume_are_separate(tmp_path, monkeypatch):
    from cubicasa5k_next.config import TrainingConfig
    from cubicasa5k_next.data.dataset import FloorplanSample
    from cubicasa5k_next.training import trainer

    monkeypatch.setattr(trainer, "FloorplanHourglass", lambda _: torch.nn.Conv2d(3, 44, 1))
    sample = FloorplanSample(
        torch.zeros(3, 8, 8),
        torch.zeros(8, 8, dtype=torch.long),
        torch.zeros(8, 8, dtype=torch.long),
        torch.zeros(21, 8, 8),
        "synthetic",
    )
    config = TrainingConfig(
        batch_size=1,
        max_epochs=1,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        tensorboard_dir=str(tmp_path / "logs"),
    )
    trainer.train_model([sample], [sample], config, torch.device("cpu"))
    first_run = next((tmp_path / "logs").iterdir())
    events = EventAccumulator(str(first_run)).Reload()
    assert events.Scalars("Loss/valid")[0].step == 1
    assert "Samples/valid/input" in events.Tags()["images"]
    from dataclasses import replace

    trainer.train_model(
        [sample],
        [sample],
        replace(config, max_epochs=2),
        torch.device("cpu"),
        tmp_path / "checkpoints" / "last.pt",
    )
    second_run = next(p for p in (tmp_path / "logs").iterdir() if p != first_run)
    assert EventAccumulator(str(second_run)).Reload().Scalars("Loss/valid")[0].step == 2


def test_heatmap_samples_are_not_sigmoided_again():
    from cubicasa5k_next.losses.uncertainty import UncertaintyWeightedLoss
    from cubicasa5k_next.training.trainer import evaluate_detailed

    model = torch.nn.Conv2d(3, 44, 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    batch = {
        "image": torch.zeros(2, 3, 4, 4),
        "room_labels": torch.zeros(2, 4, 4, dtype=torch.long),
        "icon_labels": torch.zeros(2, 4, 4, dtype=torch.long),
        "heatmaps": torch.zeros(2, 21, 4, 4),
    }
    snapshot = evaluate_detailed(
        model, UncertaintyWeightedLoss(), [batch], torch.device("cpu"), max_images=1
    )
    assert snapshot.sample_heatmaps.shape == (1, 21, 4, 4)
    assert snapshot.sample_heatmaps.max().item() == 0


def test_runs_are_isolated_and_events_readable(tmp_path):
    first = create_writer(tmp_path, "experiment")
    second = create_writer(tmp_path, "experiment")
    assert first is not None and second is not None
    try:
        assert first.log_dir != second.log_dir
        first.add_scalar("Loss/train", 0.25, 3)
        first.flush()
        events = EventAccumulator(first.log_dir).Reload()
        assert events.Scalars("Loss/train")[0].step == 3
        assert events.Scalars("Loss/train")[0].value == 0.25
    finally:
        first.close()
        second.close()


def test_disabled_and_invalid_options(tmp_path):
    assert create_writer(None, "experiment") is None
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        create_writer(tmp_path, "../escape")
    with pytest.raises(ValueError):
        create_writer("", "experiment")
    assert should_log_images(1, 5, 4)
    assert should_log_images(5, 5, 4)
    assert not should_log_images(2, 5, 4)
    assert not should_log_images(1, 0, 4)
    assert not should_log_images(1, 5, 0)
