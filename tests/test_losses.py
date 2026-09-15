import torch

from cubicasa5k_next.losses.uncertainty import UncertaintyWeightedLoss


def test_uncertainty_loss_runs() -> None:
    criterion = UncertaintyWeightedLoss()
    heat_pred = torch.rand(2, 21, 32, 32)
    heat_target = torch.rand(2, 21, 32, 32)
    room_logits = torch.randn(2, 12, 32, 32)
    icon_logits = torch.randn(2, 11, 32, 32)
    room_labels = torch.randint(0, 12, (2, 32, 32))
    icon_labels = torch.randint(0, 11, (2, 32, 32))
    output = criterion(heat_pred, heat_target, room_logits, room_labels, icon_logits, icon_labels)
    assert output.total.numel() == 1
    assert float(output.total.detach()) > 0
    output.total.backward()
    assert criterion.log_vars_mse.grad is not None
    assert criterion.log_vars.grad is not None
