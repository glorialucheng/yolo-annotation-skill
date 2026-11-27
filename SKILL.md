---
name: yolo-annotation
description: Create, revise, and review YOLO detection labels and previews. Sync labels between YOLO and X-AnyLabeling.
---

# YOLO object detection annotation

## Choose outputs

For new or revised annotations, deliver YOLO `.txt` files by default. Export X-AnyLabeling JSON only when the user requests it as an output. Recognize X-AnyLabeling across differences in case, spacing, and hyphenation. Common variants include `x any labeling`, `x label anything`, `xlabel`, `xlabelanything`, and `x-label-anything`. Generate new preview files only when the user requests previews. Sync from X-AnyLabeling to YOLO refreshes matching existing per-image previews. Apply these choices to new annotations, edits, reannotation, and direction-specific sync requests. Honor a request for only one format.

For preview or JSON export without an annotation change, start with existing same-stem YOLO labels. Use those boxes directly, including empty labels for negative images. Report missing labels for conversion-only or preview-only requests while processing the available pairs. Determine new boxes only for images the user asks to annotate or revise.

By default, save labels directly beside their original image and requested preview PNGs under that directory’s `_preview/` folder. Use an alternate location when the user explicitly specifies it. A dataset’s existing layout can help locate inputs and class mappings.

If a write outside the workspace needs permission, use the environment’s approval mechanism for the image directory. Temporary staging is fine. Confirm that the labels are present beside the originals before reporting completion. If the destination is blocked, report the write failure and any staging path.

Identify the input images, requested object classes, class IDs, output location, and any existing dataset conventions before labeling. Reuse an existing class mapping when present. If classes or their intended meaning cannot be inferred, ask the user. For a new single-class task, default to class ID 0 and report its name. Use the user's instructions and existing conventions for how to group objects, handle occlusion, and define boundaries.

## Annotation

- Annotate one image at a time. Open the original individually, inspect it, and record all its boxes or confirm an empty label before proceeding to the next image. Zoom in or crop that same image when targets or boundaries are hard to see, then map the coordinates back to the original. Base box decisions on that individual image, including when reviewing model proposals.
- Use collages, montages, and contact sheets only to review completed annotations. Track labeling progress by source image. Batch conversion and preview rendering can use the finalized labels.
- Draw a tight axis-aligned box around each visible target. Apply consistent grouping across images. Group fragments only when the requested target and visual evidence support treating them as one instance. Account for occlusions and disclose uncertain boundaries. Record an empty label for a confidently negative image; distinguish an image that could not be inspected from a negative.
- Write one UTF-8 text file per image using the same basename. Each line is `class_id x_center y_center width height`, with coordinates normalized by the original image width and height. Use nonnegative integer class IDs and positive box sizes, with all box edges within the image. This skill concerns detection boxes; follow the user's requested format if they explicitly ask for another YOLO task.
- Detect duplicate basenames and existing labels or previews before writing. Preserve existing work unless the user explicitly requests updates, and avoid treating generated previews as source images. Resolve naming or output conflicts without silently overwriting.

## Preview and verification

When previews are requested, use [scripts/render_yolo.py](scripts/render_yolo.py) with Python and Pillow (10.1 or newer). It reads and validates the existing or newly saved YOLO `.txt` and draws an original-size `_preview/*_bbox.png`. For a requested batch overview, add `--overview`. During sync, `--preview` also generates previews for images that lack them.

```bash
python3 scripts/render_yolo.py /path/to/images --overview --class-names class_zero class_one
```

The script accepts one image or one flat image directory, including images with empty label files. It uses the stored raster orientation. Pass class names in ID order to display names, or omit them to display IDs. Use `--labels-dir` when labels are stored elsewhere and `--output-dir` for an explicitly requested preview destination.

Check that every inspected image has a label, class IDs match the mapping, coordinates are finite and in bounds, and requested previews correspond to the saved labels. When previews are requested, inspect them for coverage and placement, especially uncertain or small targets. Report the number of images and boxes, class mapping, output location, and material uncertainties or images left unprocessed. Do not present estimates or unchecked proposals as verified ground truth.

## X-AnyLabeling JSON export

Use [scripts/export_xanylabeling.py](scripts/export_xanylabeling.py) when the user requests JSON. For an export-only request, read the existing YOLO labels directly. For new or revised annotations, export from the updated boxes. If the user requests only JSON for new annotations, stage the working YOLO labels and pass their directory with `--labels-dir`. Supply the dataset's actual class mapping using exactly one of `--class-names` (names in ID order), `--classes-file` (UTF-8, one name per line in ID order), or `--class-map` (a JSON object such as `{"0": "cup", "7": "bottle"}` for explicit IDs).

```bash
python3 scripts/export_xanylabeling.py /path/to/images --classes-file /path/to/classes.txt
```

The exporter reads a flat image directory or a single source image. It writes a same-stem JSON for each image, including empty `shapes` for negative images. Rectangles use four pixel corners in top-left, top-right, bottom-right, bottom-left order. `imagePath` is relative to the JSON directory, and `imageData` is null. Labels and JSON output default to the image directory. Use `--overwrite` only for explicitly requested JSON updates or synchronization to X-AnyLabeling. Report unresolved conflicts in the requested directory. Report the exported image and box counts.

## Synchronize manual X-AnyLabeling edits to YOLO

For requests such as “sync the labels,” “convert to YOLO format,” or “sync to YOLO format,” use [scripts/sync_xanylabeling_to_yolo.py](scripts/sync_xanylabeling_to_yolo.py) when the saved X-AnyLabeling JSON is the source. Inspect the requested folder and reuse the same label names and IDs used for export or training. Look for an existing mapping in the project, such as `classes.txt` or `data.yaml`. If it cannot be established, ask. Pass the same mapping options as the exporter.

```bash
python3 scripts/sync_xanylabeling_to_yolo.py /path/to/json --classes-file /path/to/classes.txt
```

Use the edited JSON as the source for this direction of sync. The script scans the folder and its subfolders, creating or updating YOLO `.txt` beside the original named by `imagePath`. Verify that any reference outside the JSON folder belongs to the intended image set. A standalone JSON without `imagePath` uses its own basename and folder. With an explicit `--output-dir /path/to/labels`, the script mirrors the JSON directory structure under that label root. An empty `shapes` list produces an empty label, including when all boxes were removed manually. The script uses the dimensions stored in the JSON and clips boxes to the image bounds when needed. It validates the batch before writing and reports unknown categories, invalid rectangles, and unsupported shapes with their file and shape index. After saving the YOLO labels, refresh each matching `_preview/*_bbox.png` that already exists using the synchronized `.txt`. Pass `--preview` to include missing previews when requested, `--preview-dir` for an explicit preview destination, or `--labels-only` for a labels-only request. Empty labels render images without boxes. Report which labels and previews were updated and any render failures.

## Edit or reannotate labels

Apply specific box edits to the existing labels or inspect the original to reannotate it, as requested. Save the revised YOLO label by default. When the user also requests JSON or previews, generate those outputs from the revised boxes and verify they match. Use the single-image path and `--overwrite` on the applicable script when replacing a requested JSON or preview.
