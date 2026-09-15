"""Sample ONNX inference (illustrative, not part of the package).

Usage:
    uv run --extra cu132 examples/onnx_infer.py model.onnx input.png outputs/
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def main(onnx_path: str, image_path: str, output_dir: str) -> None:
    import onnxruntime as ort

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from cubicasa5k_next.utils.image import save_colored_mask

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    tensor = (2.0 * (resized.astype(np.float32) / 255.0) - 1.0).transpose(2, 0, 1)
    tensor = np.expand_dims(tensor, 0).astype(np.float32)
    raw = session.run(["raw44"], {"image": tensor})[0][0]
    room_labels = np.argmax(raw[21:33], axis=0).astype("int64")
    icon_labels = np.argmax(raw[33:44], axis=0).astype("int64")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    save_colored_mask(room_labels, out / f"{stem}_rooms.png")
    save_colored_mask(icon_labels, out / f"{stem}_icons.png")
    print(f"saved masks to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
