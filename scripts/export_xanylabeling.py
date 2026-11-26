"""Export YOLO detection labels as X-AnyLabeling JSON files."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from yolo_io import add_class_mapping_args, load_classes, read_boxes, replace_text, source_images


def rectangle(name, bounds, width, height):
    left, top, right, bottom = bounds
    x1, y1 = left * width, top * height
    x2, y2 = right * width, bottom * height
    return {
        "label": name,
        "score": None,
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }


def run(args):
    classes = load_classes(args)
    source = args.images
    single_image = source.is_file()
    images_dir = source.parent if single_image else source
    labels_dir = args.labels_dir or images_dir
    output_dir = args.output_dir or images_dir
    exports = []
    box_count = 0
    images = [source] if single_image else source_images(images_dir)
    for path in images:
        destination = output_dir / f"{path.stem}.json"
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError(f"output must be a regular file: {destination}")
        if destination.exists() and not args.overwrite:
            raise ValueError(f"output already exists: {destination}; choose another --output-dir")
        boxes = read_boxes(labels_dir / f"{path.stem}.txt", classes)
        with Image.open(path) as image:
            width, height = image.size
        shapes = [rectangle(classes[class_id], bounds, width, height) for class_id, bounds in boxes]
        exports.append((destination, {
            "flags": {},
            "checked": False,
            "shapes": shapes,
            "imagePath": Path(os.path.relpath(path, output_dir)).as_posix(),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
            "description": "",
        }))
        box_count += len(shapes)

    output_dir.mkdir(parents=True, exist_ok=True)
    for destination, annotation in exports:
        content = json.dumps(annotation, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.overwrite:
            replace_text(destination, content)
        else:
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(content)
    print(f"Exported {len(exports)} JSON files with {box_count} boxes to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="one source image or a flat image directory")
    parser.add_argument("--labels-dir", type=Path, help="defaults to the image directory")
    parser.add_argument("--output-dir", type=Path, help="defaults to the image directory")
    parser.add_argument("--overwrite", action="store_true", help="replace existing X-AnyLabeling JSON")
    add_class_mapping_args(parser)
    try:
        run(parser.parse_args())
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
