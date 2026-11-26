"""Synchronize YOLO detection labels from manually edited X-AnyLabeling JSON."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from yolo_io import (
    IMAGE_EXTENSIONS,
    PREVIEW_SUBDIR,
    add_class_mapping_args,
    load_classes,
    read_boxes,
    replace_text,
)

IMAGE_FIELDS = {"imageWidth", "imageHeight"}


def normalized_box(shape, width, height):
    points = shape.get("points")
    if not isinstance(points, list) or len(points) not in (2, 4):
        raise ValueError("expected two opposite corners or four rectangle corners")
    for point in points:
        if not isinstance(point, list) or len(point) != 2 or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in point
        ):
            raise ValueError("each corner must contain finite numeric x and y coordinates")
    x, y = zip(*points)
    bounds = min(x), min(y), max(x), max(y)
    left, top, right, bottom = bounds
    if len(points) == 4 and set(map(tuple, points)) != {
        (left, top), (right, top), (right, bottom), (left, bottom)
    }:
        raise ValueError("four points must describe an axis-aligned rectangle")
    clipped = max(0, left), max(0, top), min(width, right), min(height, bottom)
    left, top, right, bottom = clipped
    if left >= right or top >= bottom:
        raise ValueError("rectangle must have positive area inside the image")
    return (
        ((left + right) / (2 * width), (top + bottom) / (2 * height),
         (right - left) / width, (bottom - top) / height),
        clipped != bounds,
    )


def read_annotation(path, class_ids):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(data, dict):
        return None
    if "imagePath" not in data and not ("shapes" in data and IMAGE_FIELDS & data.keys()):
        return None

    width, height = data.get("imageWidth"), data.get("imageHeight")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError(f"{path}: imageWidth and imageHeight must be positive integers")
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"{path}: shapes must be an array")

    lines = []
    clipped_count = 0
    for index, shape in enumerate(shapes, 1):
        try:
            if not isinstance(shape, dict) or shape.get("shape_type") != "rectangle":
                raise ValueError("expected a rectangle for YOLO detection export")
            name = shape.get("label")
            if not isinstance(name, str) or name not in class_ids:
                raise ValueError(f"unmapped class {name!r}")
            coordinates, clipped = normalized_box(shape, width, height)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{path}: shape {index}: {error}") from error
        lines.append(f"{class_ids[name]} " + " ".join(f"{value:.10g}" for value in coordinates))
        clipped_count += clipped
    return "".join(line + "\n" for line in lines), len(lines), clipped_count, data.get("imagePath")


def referenced_image(source, image_path):
    if image_path is None or image_path == "":
        return None
    if not isinstance(image_path, str):
        raise ValueError(f"{source}: imagePath must be a filename or path")
    image = source.parent / Path(image_path.replace('\\', "/"))
    if image.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"{source}: unsupported imagePath {image_path!r}; specify --output-dir")
    return image


def label_path(source, image_path):
    image = referenced_image(source, image_path)
    if image is None:
        return source.with_suffix(".txt")
    if image.parent.resolve() != source.parent.resolve() and not image.is_file():
        raise ValueError(f"{source}: referenced image not found: {image}; specify --output-dir")
    return image.with_suffix(".txt")


def preview_image(source, image):
    if image is not None:
        if not image.is_file():
            raise ValueError(f"referenced image not found: {image}")
        return image

    matches = [
        path for path in source.parent.iterdir()
        if path.stem == source.stem and path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    ]
    if not matches:
        raise ValueError("no imagePath or same-stem image")
    if len(matches) > 1:
        raise ValueError("more than one same-stem image")
    return matches[0]


def refresh_previews(args, annotations, classes):
    create_missing = args.preview or args.preview_dir is not None
    jobs = []
    for source, image_path, label in annotations:
        try:
            reference = referenced_image(source, image_path)
            name_source = reference or source
            preview_dir = name_source.parent / PREVIEW_SUBDIR
            if args.preview_dir is not None:
                preview_dir = args.preview_dir / source.relative_to(args.json_dir).parent
            output = preview_dir / f"{name_source.stem}_bbox.png"
            if not create_missing and not output.is_file():
                continue
            image = preview_image(source, reference)
        except (OSError, ValueError) as error:
            print(f"Skipped preview for {source}: {error}", file=sys.stderr)
            continue
        jobs.append((image, label, output))

    if not jobs:
        if create_missing:
            print("Updated 0 previews")
        return
    try:
        from render_yolo import check_preview_path, render, save_preview
    except ImportError as error:
        raise ValueError("Preview refresh requires Python with Pillow (10.1 or newer)") from error

    originals = {str(image.resolve()).casefold() for image, _, _ in jobs}
    outputs = Counter(str(output.resolve()).casefold() for _, _, output in jobs)
    rendered = 0
    for image, label, output in jobs:
        try:
            key = str(output.resolve()).casefold()
            if key in originals or outputs[key] > 1:
                raise ValueError(f"conflicting preview destination: {output}")
            check_preview_path(output, overwrite=True)
            boxes = read_boxes(label, classes)
            preview = render(image, boxes, classes)
            output.parent.mkdir(parents=True, exist_ok=True)
            save_preview(preview, output, overwrite=True)
            rendered += 1
        except (OSError, ValueError) as error:
            print(f"Skipped preview for {image}: {error}", file=sys.stderr)
    print(f"Updated {rendered} previews")


def run(args):
    if not args.json_dir.is_dir():
        raise ValueError(f"annotation directory does not exist: {args.json_dir}")
    classes = load_classes(args)
    class_ids = {name: class_id for class_id, name in classes.items()}
    mapping_file = args.classes_file.resolve() if args.classes_file else None
    pending = []
    annotations = []
    destinations = set()
    box_count = clipped_count = ignored = unchanged = 0
    for source in sorted(args.json_dir.rglob("*")):
        if source.suffix.lower() != ".json" or not source.is_file():
            continue
        result = read_annotation(source, class_ids)
        if result is None:
            ignored += 1
            continue
        content, count, clipped, image_path = result
        if args.output_dir is None:
            destination = label_path(source, image_path)
        else:
            destination = args.output_dir / source.relative_to(args.json_dir).with_suffix(".txt")
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError(f"output must be a regular file: {destination}")
        canonical = destination.resolve()
        if mapping_file == canonical:
            raise ValueError(f"output would overwrite the class list: {destination}")
        key = str(canonical).casefold()
        if key in destinations:
            raise ValueError(f"multiple JSON files map to {destination}")
        destinations.add(key)
        if not args.labels_only:
            annotations.append((source, image_path, destination))

        box_count += count
        clipped_count += clipped
        exists = destination.exists()
        if exists and destination.read_bytes() == content.encode("utf-8"):
            unchanged += 1
        else:
            pending.append((destination, content, exists))

    if not destinations:
        raise ValueError(f"no X-AnyLabeling annotations in {args.json_dir}")
    for destination, content, _ in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        replace_text(destination, content)
    updated = sum(existed for _, _, existed in pending)
    location = f" under {args.output_dir}" if args.output_dir else ""
    print(
        f"Synchronized {len(destinations)} labels with {box_count} boxes{location} "
        f"({len(pending) - updated} created, {updated} updated, {unchanged} unchanged)"
    )
    if clipped_count:
        print(f"Clipped {clipped_count} boxes to the image boundary")
    if ignored:
        print(f"Skipped {ignored} unrelated JSON files")
    if annotations:
        refresh_previews(args, annotations, classes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_dir", type=Path, help="directory containing X-AnyLabeling JSON files")
    parser.add_argument("--output-dir", type=Path, help="YOLO label root; defaults to each referenced image folder")
    previews = parser.add_mutually_exclusive_group()
    previews.add_argument(
        "--preview", action="store_true", help=f"refresh previews and create missing ones under {PREVIEW_SUBDIR}/"
    )
    previews.add_argument("--labels-only", action="store_true", help="sync labels without refreshing existing previews")
    parser.add_argument("--preview-dir", type=Path, help="refresh previews under this explicit root")
    add_class_mapping_args(parser)
    args = parser.parse_args()
    if args.labels_only and args.preview_dir is not None:
        parser.error("--labels-only cannot be combined with --preview-dir")
    try:
        run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
