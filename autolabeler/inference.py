"""Ultralytics prediction and YOLO label serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _number(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _enabled_keypoints_for(enabled_keypoints, cls: int):
    """Return the enabled keypoint indices for ``cls`` (``None`` = all enabled)."""
    if enabled_keypoints is None:
        return None
    if isinstance(enabled_keypoints, dict):
        if cls in enabled_keypoints:
            return set(enabled_keypoints[cls])
        return None
    return set(enabled_keypoints)


def serialize_result(result: Any, label_format: str, enabled_keypoints=None) -> str:
    """Convert one Ultralytics Results object to normalized YOLO label rows.

    ``enabled_keypoints`` optionally restricts which pose keypoints are kept;
    disabled keypoints are written as zero padding (0 0 0).
    """
    width, height = result.orig_shape[1], result.orig_shape[0]
    rows: list[str] = []
    if label_format == "yolo":
        boxes = result.boxes
        if boxes is None:
            return ""
        for box in boxes:
            x1, y1, x2, y2 = (_number(value) for value in box.xyxy[0])
            cls = int(_number(box.cls[0]))
            rows.append(
                f"{cls} {(x1 + x2) / (2 * width):.6f} {(y1 + y2) / (2 * height):.6f} "
                f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
            )
    elif label_format == "yolo-obb":
        obb = result.obb
        if obb is None:
            return ""
        for corners, cls_value in zip(obb.xyxyxyxy, obb.cls):
            coordinates = [
                coordinate
                for point in corners
                for coordinate in (
                    max(0.0, min(1.0, _number(point[0]) / width)),
                    max(0.0, min(1.0, _number(point[1]) / height)),
                )
            ]
            rows.append(f"{int(_number(cls_value))} " + " ".join(f"{value:.6f}" for value in coordinates))
    elif label_format == "yolo-pose":
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None:
            return ""
        if keypoints is None or len(keypoints.data) != len(boxes):
            raise ValueError("Pose model returned boxes without matching keypoints")
        for box, points in zip(boxes, keypoints.data):
            x1, y1, x2, y2 = (_number(value) for value in box.xyxy[0])
            cls = int(_number(box.cls[0]))
            enabled = _enabled_keypoints_for(enabled_keypoints, cls)
            values = [
                str(cls),
                f"{(x1 + x2) / (2 * width):.6f}",
                f"{(y1 + y2) / (2 * height):.6f}",
                f"{(x2 - x1) / width:.6f}",
                f"{(y2 - y1) / height:.6f}",
            ]
            for point_index, point in enumerate(points):
                x = max(0.0, min(1.0, _number(point[0]) / width))
                y = max(0.0, min(1.0, _number(point[1]) / height))
                visibility = _number(point[2]) if len(point) > 2 else 2.0
                if enabled is not None and point_index not in enabled:
                    x, y, visibility = 0.0, 0.0, 0.0
                values.extend((f"{x:.6f}", f"{y:.6f}", f"{visibility:g}"))
            rows.append(" ".join(values))
    else:
        raise ValueError(f"Unsupported label format: {label_format}")
    return "\n".join(rows) + ("\n" if rows else "")


def run_inference(
    model_path: str,
    images: Iterable[Path],
    label_dir: Path,
    image_dir: Path,
    label_format: str,
    progress: Callable[[dict[str, Any]], None],
    is_cancelled: Callable[[], bool],
    enabled_keypoints=None,
) -> None:
    """Predict selected images and write labels while preserving folder structure."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the Python environment running autolabeler"
        ) from exc

    model = YOLO(model_path)
    expected_task = {"yolo": "detect", "yolo-obb": "obb", "yolo-pose": "pose"}[label_format]
    if model.task != expected_task:
        raise ValueError(
            f"Selected {model.task} model is incompatible with {label_format} labels"
        )
    selected = list(images)
    for index, image_path in enumerate(selected, start=1):
        if is_cancelled():
            break
        results = model.predict(source=str(image_path), stream=False, verbose=False)
        if len(results) != 1:
            raise RuntimeError(f"Expected one prediction result for {image_path}, got {len(results)}")
        content = serialize_result(results[0], label_format, enabled_keypoints)
        relative = image_path.relative_to(image_dir)
        destination = (label_dir / relative).with_suffix(".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".autolabeler-tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        progress({"completed": index, "total": len(selected), "image": relative.as_posix()})
