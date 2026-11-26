"""Render YOLO detection labels onto copies of the source images."""

import argparse
import math
from pathlib import Path
from shutil import copymode
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont

from yolo_io import PREVIEW_SUBDIR, read_boxes, source_images

PAGE_SIZE = 24
COLORS = ("#FFCE49", "#56D8FF", "#8EED94", "#F9A9EE", "#FF9B72")


def render(path, boxes, names):
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, min(32, width // 60)))
    stroke = max(2, min(8, width // 450))
    for class_id, (left, top, right, bottom) in boxes:
        x1, y1 = int(left * width), int(top * height)
        x2 = min(width - 1, max(x1, math.ceil(right * width) - 1))
        y2 = min(height - 1, max(y1, math.ceil(bottom * height) - 1))
        color = COLORS[class_id % len(COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=stroke)
        tag = f"{class_id} {names[class_id]}" if names else str(class_id)
        _, _, label_width, label_height = draw.textbbox((0, 0), tag, font=font, anchor="lt")
        tag_width, tag_height = label_width + 8, label_height + 6
        tag_x = max(0, min(x1, width - tag_width))
        tag_y = max(0, y1 - tag_height)
        draw.rectangle((tag_x, tag_y, tag_x + tag_width, tag_y + tag_height), fill=color)
        draw.text((tag_x + 4, tag_y + 3), tag, font=font, anchor="lt", fill="#171717")
    return image


def overview(items):
    columns, cell_width, cell_height = 3, 336, 220
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#151A20")
    draw = ImageDraw.Draw(sheet)
    for index, (path, thumb) in enumerate(items):
        row, column = divmod(index, columns)
        x, y = column * cell_width + 8, row * cell_height + 6
        name = path.name if len(path.name) <= 48 else path.name[:45] + "..."
        draw.text((x, y), name, fill="white")
        sheet.paste(thumb, (x + (320 - thumb.width) // 2, y + 24))
    return sheet


def check_preview_path(path, overwrite):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"output must be a regular file: {path}")
    if path.exists() and not overwrite:
        raise ValueError(f"output already exists: {path}; use --overwrite to refresh it")


def save_preview(image, path, overwrite):
    if not overwrite:
        with path.open("xb") as handle:
            image.save(handle, format="PNG")
        return
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / path.name
        image.save(staged)
        if path.is_file():
            copymode(path, staged)
        staged.replace(path)


def run(args):
    source = args.images
    single_image = source.is_file()
    images_dir = source.parent if single_image else source
    labels_dir = args.labels_dir or images_dir
    output_dir = args.output_dir or (images_dir / PREVIEW_SUBDIR)
    images = [source] if single_image else source_images(images_dir)

    pages = math.ceil(len(images) / PAGE_SIZE)
    summaries = (
        [output_dir / ("bbox_overview.png" if pages == 1 else f"bbox_overview_{i + 1:03d}.png")
         for i in range(pages)] if args.overview else []
    )
    previews = [output_dir / f"{path.stem}_bbox.png" for path in images]
    for path in previews + summaries:
        check_preview_path(path, args.overwrite)
    class_ids = range(len(args.class_names)) if args.class_names else None
    annotations = [read_boxes(labels_dir / f"{path.stem}.txt", class_ids) for path in images]
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    for index, (path, boxes, preview) in enumerate(zip(images, annotations, previews)):
        image = render(path, boxes, args.class_names)
        save_preview(image, preview, args.overwrite)
        if args.overview:
            image.thumbnail((320, 180), Image.Resampling.LANCZOS)
            thumbnails.append((path, image))
            if len(thumbnails) == PAGE_SIZE or index + 1 == len(images):
                save_preview(overview(thumbnails), summaries[index // PAGE_SIZE], args.overwrite)
                thumbnails.clear()
    print(f"Rendered {len(images)} images and {sum(map(len, annotations))} boxes to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="one source image or a flat image directory")
    parser.add_argument("--labels-dir", type=Path, help="defaults to the image directory")
    parser.add_argument("--output-dir", type=Path, help=f"defaults to {PREVIEW_SUBDIR}/ in the image directory")
    parser.add_argument("--class-names", nargs="+", help="class names in zero-based ID order")
    parser.add_argument("--overview", action="store_true", help="write one or more contact sheets")
    parser.add_argument("--overwrite", action="store_true", help="replace existing preview PNGs")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
