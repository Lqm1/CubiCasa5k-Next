import numpy as np

from cubicasa5k_next.postprocess.vectorize import vectorize_prediction


def test_vectorize_runs_on_synthetic_logits() -> None:
    height = width = 64
    num_room, num_icon, num_heat = 12, 11, 21
    room_logits = np.full((num_room, height, width), -10.0, dtype=np.float32)
    room_logits[4, 8:56, 8:56] = 10.0
    room_logits[2, 8:12, 8:56] = 12.0
    icon_logits = np.full((num_icon, height, width), -10.0, dtype=np.float32)
    icon_logits[0, :, :] = 10.0
    heatmaps = np.zeros((num_heat, height, width), dtype=np.float32)
    heatmaps[0, 8, 8] = 1.0
    heatmaps[0, 8, 55] = 1.0
    heatmaps[0, 55, 55] = 1.0
    heatmaps[0, 55, 8] = 1.0
    result = vectorize_prediction(room_logits, icon_logits, heatmaps)
    assert len(result.junctions) >= 4
    assert isinstance(result.walls, list)
    assert isinstance(result.rooms, list)
