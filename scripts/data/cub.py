"""Loader for the CUB-200-2011 dataset's images, classes, and part-keypoint annotations.

Expects the standard CUB-200-2011 directory layout (as produced by
scripts/download_datasets.sh):

    CUB_200_2011/
        images/<class_folder>/<image>.jpg
        images.txt              # "<image_id> <relative_path>"
        image_class_labels.txt  # "<image_id> <class_id>"
        classes.txt              # "<class_id> <class_name>"
        bounding_boxes.txt        # "<image_id> <x> <y> <w> <h>"
        parts/parts.txt            # "<part_id> <part_name>"
        parts/part_locs.txt         # "<image_id> <part_id> <x> <y> <visible>"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CubImage:
    image_id: int
    path: Path
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]  # x, y, w, h
    keypoints: dict[str, tuple[float, float]] = field(default_factory=dict)  # only VISIBLE parts


def load_cub(root: str | Path) -> dict[int, CubImage]:
    root = Path(root)
    img_dir = root / "images"

    images_txt = _read_lines(root / "images.txt")
    id_to_path = {int(iid): img_dir / rel for iid, rel in (line.split(maxsplit=1) for line in images_txt)}

    class_labels = _read_lines(root / "image_class_labels.txt")
    id_to_class_id = {int(iid): int(cid) for iid, cid in (line.split() for line in class_labels)}

    classes_txt = _read_lines(root / "classes.txt")
    class_id_to_name = {int(cid): name for cid, name in (line.split(maxsplit=1) for line in classes_txt)}

    bboxes_txt = _read_lines(root / "bounding_boxes.txt")
    id_to_bbox = {}
    for line in bboxes_txt:
        iid, x, y, w, h = line.split()
        id_to_bbox[int(iid)] = (float(x), float(y), float(w), float(h))

    parts_txt = _read_lines(root / "parts" / "parts.txt")
    part_id_to_name = {int(pid): name for pid, name in (line.split(maxsplit=1) for line in parts_txt)}

    part_locs_txt = _read_lines(root / "parts" / "part_locs.txt")
    id_to_keypoints: dict[int, dict[str, tuple[float, float]]] = {}
    for line in part_locs_txt:
        iid, pid, x, y, visible = line.split()
        if int(visible) == 0:
            continue
        id_to_keypoints.setdefault(int(iid), {})[part_id_to_name[int(pid)]] = (float(x), float(y))

    images: dict[int, CubImage] = {}
    for image_id, path in id_to_path.items():
        class_id = id_to_class_id[image_id]
        images[image_id] = CubImage(
            image_id=image_id,
            path=path,
            class_id=class_id,
            class_name=class_id_to_name[class_id],
            bbox=id_to_bbox[image_id],
            keypoints=id_to_keypoints.get(image_id, {}),
        )
    return images


def images_by_class(images: dict[int, CubImage]) -> dict[int, list[CubImage]]:
    by_class: dict[int, list[CubImage]] = {}
    for img in images.values():
        by_class.setdefault(img.class_id, []).append(img)
    return by_class


def _read_lines(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]
