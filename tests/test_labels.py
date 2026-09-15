from cubicasa5k_next.labels import (
    NUM_HEATMAP_CHANNELS,
    NUM_ICON_CLASSES,
    NUM_ROOM_CLASSES,
)


def test_class_counts_match_paper() -> None:
    assert NUM_ROOM_CLASSES == 12
    assert NUM_ICON_CLASSES == 11
    assert NUM_HEATMAP_CHANNELS == 21
