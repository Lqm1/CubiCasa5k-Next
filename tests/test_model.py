import torch

from cubicasa5k_next.model.hourglass import FloorplanHourglass


def test_forward_shapes() -> None:
    model = FloorplanHourglass(44).eval()
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        raw = model(dummy)
    assert raw.shape == (1, 44, 256, 256)
    assert float(raw[:, :21].min()) >= 0.0
    assert float(raw[:, :21].max()) <= 1.0
