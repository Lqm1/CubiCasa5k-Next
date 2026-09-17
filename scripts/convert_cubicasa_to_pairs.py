"""Convert a CubiCasa5k download into the images/annotations pair layout.

Source layout (Zenodo download)::

    <src>/
        train.txt / val.txt / test.txt      # lines like /high_quality_architectural/6044/
        high_quality/<id>/F1_scaled.png + model.svg
        high_quality_architectural/<id>/F1_scaled.png + model.svg
        colorful/<id>/F1_scaled.png + model.svg

Destination layout (what the training CLI loads)::

    <dst>/<split>/images/<stem>.png
    <dst>/<split>/annotations/<stem>.svg

where ``<stem>`` is ``<group>_<id>`` (e.g. ``high_quality_architectural_6044``).
File contents are copied unchanged; only the directory structure changes.

Usage::

    python scripts/convert_cubicasa_to_pairs.py D:\\Downloads\\cubicasa5k D:\\Downloads\\cubicasa5k-pairs
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SPLITS = ("train", "val", "test")


def stem_for_folder(folder: str) -> str:
    return folder.strip().strip("/\\").replace("/", "_").replace("\\", "_")


def convert(src: Path, dst: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        split_txt = src / f"{split}.txt"
        if not split_txt.exists():
            print(f"skip {split}: {split_txt} not found")
            continue
        folders = [
            ln.strip() for ln in split_txt.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        images_dir = dst / split / "images"
        annotations_dir = dst / split / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        done = 0
        for folder in folders:
            stem = stem_for_folder(folder)
            relative = folder.lstrip("/\\")
            image_src = src / relative / "F1_scaled.png"
            svg_src = src / relative / "model.svg"
            if not image_src.exists() or not svg_src.exists():
                print(f"  missing: {folder}")
                continue
            shutil.copyfile(image_src, images_dir / f"{stem}.png")
            shutil.copyfile(svg_src, annotations_dir / f"{stem}.svg")
            done += 1
        counts[split] = done
        print(f"{split}: {done}/{len(folders)} pairs -> {dst / split}")
    return counts


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    convert(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
