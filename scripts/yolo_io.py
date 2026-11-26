"""Read normalized YOLO detection labels and find their source images."""

import json
import math
import os
import re
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PREVIEW_SUBDIR = "_preview"
EDGE_TOLERANCE = 1e-6


def source_images(directory):
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.stem.lower().endswith("_bbox")
        and not re.fullmatch(r"bbox_overview(?:_[0-9]+)?", path.stem, re.IGNORECASE)
    )
    if not images:
        raise ValueError(f"no source images in {directory}")
    if len({p.stem.casefold() for p in images}) != len(images):
        raise ValueError("source images have duplicate basenames")

    return images


def read_boxes(path, class_ids=None):
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            class_id, *coordinates = line.split()
            if not re.fullmatch(r"[0-9]+", class_id):
                raise ValueError("expected a nonnegative integer class ID")
            class_id = int(class_id)
            if class_ids is not None and class_id not in class_ids:
                raise ValueError(f"class ID {class_id} has no class name")
            if len(coordinates) != 4:
                raise ValueError("expected x_center y_center width height")
            x, y, width, height = map(float, coordinates)
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, width, height)):
                raise ValueError("coordinates must be finite and within [0, 1]")
            if width <= 0 or height <= 0:
                raise ValueError("box dimensions must be positive")
            left, top = x - width / 2, y - height / 2
            right, bottom = x + width / 2, y + height / 2
            if min(left, top) < -EDGE_TOLERANCE or max(right, bottom) > 1 + EDGE_TOLERANCE:
                raise ValueError("box extends beyond the image")
            boxes.append((class_id, (max(0, left), max(0, top), min(1, right), min(1, bottom))))
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return boxes


def add_class_mapping_args(parser):
    mapping = parser.add_mutually_exclusive_group(required=True)
    mapping.add_argument("--class-names", nargs="+", help="class names in zero-based ID order")
    mapping.add_argument("--classes-file", type=Path, help="UTF-8 class names, one per line in ID order")
    mapping.add_argument("--class-map", type=Path, help="JSON object mapping class IDs to names")


def load_classes(args):
    if args.class_map:
        names = json.loads(args.class_map.read_text(encoding="utf-8-sig"))
        if not isinstance(names, dict) or any(
            not key.isascii() or not key.isdecimal() or key != str(int(key))
            for key in names
        ):
            raise ValueError("class map must be a JSON object with nonnegative integer ID keys")
        classes = {int(key): name for key, name in names.items()}
    else:
        names = args.class_names
        if args.classes_file:
            names = args.classes_file.read_text(encoding="utf-8-sig").splitlines()
        classes = dict(enumerate(names))
    if not classes or any(not isinstance(name, str) or not name.strip() for name in classes.values()):
        raise ValueError("provide a nonempty class name for each mapped ID")
    if len(set(classes.values())) != len(classes):
        raise ValueError("class names must be distinct to preserve class identity in JSON")
    return classes


def replace_text(path, content):
    """Replace a label at once, even if an existing label is empty."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"output must be a regular file: {path}")
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
        if mode is not None:
            temporary_path.chmod(mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
