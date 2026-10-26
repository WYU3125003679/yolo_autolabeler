"""Ultralytics training adapter and labelled dataset staging."""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MODEL_TASKS = {"yolo": "detect", "yolo-obb": "obb", "yolo-pose": "pose"}
EXPECTED_FIELDS = {"yolo": 5, "yolo-obb": 9}


def natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Sort digit runs numerically while keeping text runs case-insensitive."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
    )


def iter_labeled_pairs(image_dir: Path, label_dir: Path) -> Iterable[tuple[Path, Path]]:
    """Yield ``(image, label)`` for images that already have a non-empty label file."""
    for image in sorted(
        image_dir.rglob("*"),
        key=lambda item: natural_sort_key(item.relative_to(image_dir).as_posix()),
    ):
        if image.suffix.lower() not in IMAGE_EXTENSIONS or not image.is_file():
            continue
        label = (label_dir / image.relative_to(image_dir)).with_suffix(".txt")
        if not label.is_file():
            continue
        if not label.read_text(encoding="utf-8").strip():
            continue
        yield image, label


def load_dataset_config(dataset_config: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the training dataset YAML") from exc
    config = yaml.safe_load(dataset_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping with class names")
    return config


def parse_class_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names")
    if isinstance(names, list):
        class_names = {index: str(name) for index, name in enumerate(names)}
    elif isinstance(names, dict):
        try:
            class_names = {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset class names must use integer IDs") from exc
    else:
        raise ValueError("Dataset YAML must define class names as a list or mapping")
    if not class_names or sorted(class_names) != list(range(len(class_names))):
        raise ValueError("Dataset class IDs must be contiguous and start at zero")
    return class_names


def parse_keypoint_count(config: dict[str, Any], label_format: str) -> int:
    if label_format != "yolo-pose":
        return 0
    shape = config.get("kpt_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not isinstance(shape[0], int)
        or shape[0] < 1
        or shape[1] != 3
    ):
        raise ValueError("Pose training requires kpt_shape: [keypoint_count, 3] in the dataset YAML")
    return shape[0]


def derive_metadata(
    samples: list[tuple[Path, Path]], label_format: str
) -> tuple[dict[int, str], int]:
    """Auto-generate class names and pose keypoint count from the labels."""
    used_ids: set[int] = set()
    keypoint_counts: set[int] = set()
    for _image, label in samples:
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError as exc:
                raise ValueError(f"Invalid class ID in {label}") from exc
            used_ids.add(class_id)
            if label_format == "yolo-pose":
                if (len(fields) - 5) % 3 != 0:
                    raise ValueError(f"Invalid pose label field count in {label}")
                keypoint_counts.add((len(fields) - 5) // 3)
    if not used_ids:
        raise ValueError("No labelled objects were found in the selected label directory")
    if sorted(used_ids) != list(range(len(used_ids))):
        raise ValueError(
            "Class IDs must be contiguous and start at zero; provide a dataset YAML with explicit names"
        )
    class_names = {class_id: f"class_{class_id}" for class_id in sorted(used_ids)}
    keypoint_count = 0
    if label_format == "yolo-pose":
        if len(keypoint_counts) != 1:
            raise ValueError(
                "Pose labels use inconsistent keypoint counts; provide a dataset YAML with kpt_shape"
            )
        keypoint_count = next(iter(keypoint_counts))
    return class_names, keypoint_count


def validate_samples(
    samples: list[tuple[Path, Path]],
    label_format: str,
    class_names: dict[int, str],
    keypoint_count: int,
) -> None:
    expected_fields = {
        "yolo": 5,
        "yolo-obb": 9,
        "yolo-pose": 5 + keypoint_count * 3,
    }[label_format]
    for _image, label in samples:
        for row_number, row in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            fields = row.split()
            if not fields:
                continue
            if len(fields) != expected_fields:
                raise ValueError(
                    f"Expected {expected_fields} fields in {label} line {row_number} for {label_format}"
                )
            try:
                class_id = int(fields[0])
            except ValueError as exc:
                raise ValueError(f"Invalid class ID in {label} line {row_number}") from exc
            if class_id not in class_names:
                raise ValueError(
                    f"Class ID {class_id} in {label} is not present in dataset YAML names"
                )
            try:
                values = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"Invalid coordinate in {label} line {row_number}") from exc
            coordinate_values = values
            if label_format == "yolo-pose":
                coordinate_values = list(values[:4])
                visibility_values = []
                for keypoint_index in range(keypoint_count):
                    coordinate_values.append(values[4 + keypoint_index * 3])
                    coordinate_values.append(values[5 + keypoint_index * 3])
                    visibility_values.append(values[6 + keypoint_index * 3])
                if any(value not in {0, 1, 2} for value in visibility_values):
                    raise ValueError(f"Invalid keypoint visibility in {label} line {row_number}")
            if any(not 0 <= value <= 1 for value in coordinate_values):
                raise ValueError(
                    f"Coordinates must be normalized to [0, 1] in {label} line {row_number}"
                )
            if label_format in {"yolo", "yolo-pose"} and (values[2] <= 0 or values[3] <= 0):
                raise ValueError(f"Bounding-box size must be positive in {label} line {row_number}")


def stage_dataset(
    samples: list[tuple[Path, Path]],
    class_names: dict[int, str],
    run_dir: Path,
    seed: int,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    """Shuffle, split and stage the samples into a run-local Ultralytics dataset."""
    if not samples:
        raise ValueError("No labelled images were found in the selected image directory")

    random.Random(seed).shuffle(samples)
    if len(samples) == 1:
        train_samples = val_samples = samples
    else:
        validation_count = min(max(1, round(len(samples) * 0.2)), len(samples) - 1)
        val_samples = samples[:validation_count]
        train_samples = samples[validation_count:]

    staging = run_dir / "dataset"
    for split, split_samples in (("train", train_samples), ("val", val_samples)):
        images_dir = staging / "images" / split
        labels_dir = staging / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for index, (image, label) in enumerate(split_samples):
            image_target = images_dir / f"{index:08d}{image.suffix.lower()}"
            label_target = labels_dir / f"{index:08d}.txt"
            try:
                os.link(image, image_target)
            except OSError:
                shutil.copy2(image, image_target)
            shutil.copy2(label, label_target)

    staged_config: dict[str, Any] = {
        "path": str(staging),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    for key in ("kpt_shape", "flip_idx"):
        if extra_config and key in extra_config:
            staged_config[key] = extra_config[key]

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write the staged dataset YAML") from exc
    staged_yaml = staging / "dataset.yaml"
    staged_yaml.write_text(
        yaml.safe_dump(staged_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return staged_yaml


def prepare_training_dataset(
    image_dir: Path,
    label_dir: Path,
    dataset_config: Path | None,
    run_dir: Path,
    label_format: str,
    seed: int,
) -> Path:
    """Collect labelled pairs from ``image_dir`` and stage an Ultralytics dataset.

    When ``dataset_config`` is ``None``, class names and the pose keypoint count are
    derived automatically from the labels and a temporary YAML is written to ``run_dir``.
    """
    samples = list(iter_labeled_pairs(image_dir, label_dir))
    if not samples:
        raise ValueError("No labelled images were found in the selected image directory")

    extra_config: dict[str, Any] = {}
    if dataset_config is not None:
        config = load_dataset_config(dataset_config)
        class_names = parse_class_names(config)
        keypoint_count = parse_keypoint_count(config, label_format)
        if label_format == "yolo-pose":
            extra_config["kpt_shape"] = config.get("kpt_shape")
        if "flip_idx" in config:
            extra_config["flip_idx"] = config["flip_idx"]
    else:
        class_names, keypoint_count = derive_metadata(samples, label_format)
        run_dir.mkdir(parents=True, exist_ok=True)
        generated_config: dict[str, Any] = {"names": list(class_names.values())}
        if label_format == "yolo-pose":
            extra_config["kpt_shape"] = [keypoint_count, 3]
            generated_config["kpt_shape"] = [keypoint_count, 3]

        generated_yaml = run_dir / "generated.yaml"
        generated_yaml.write_text(_dump_yaml(generated_config), encoding="utf-8")

    validate_samples(samples, label_format, class_names, keypoint_count)
    return stage_dataset(samples, class_names, run_dir, seed, extra_config)


def _dump_yaml(config: dict[str, Any]) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write the generated dataset YAML") from exc
    return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)


def run_training(
    parameters: dict[str, Any],
    run_dir: Path,
    image_dir: Path,
    label_dir: Path,
    dataset_config: Path | None,
    label_format: str,
    progress: Callable[[dict[str, Any]], None],
    is_cancelled: Callable[[], bool],
) -> None:
    """Run one Ultralytics training call and stream epoch progress."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the Python environment running autolabeler"
        ) from exc

    staged_data = prepare_training_dataset(
        image_dir,
        label_dir,
        dataset_config,
        run_dir,
        label_format,
        parameters["seed"],
    )
    parameters["staged_data"] = str(staged_data)
    (run_dir / "request.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model = YOLO(parameters["model"] or "yolo11n.pt")
    expected_task = MODEL_TASKS[label_format]
    if model.task != expected_task:
        raise ValueError(
            f"Selected {model.task} model is incompatible with {label_format} training labels"
        )

    def on_epoch_end(trainer: Any) -> None:
        if is_cancelled():
            trainer.stop = True
        progress(
            {
                "epoch": int(getattr(trainer, "epoch", -1)) + 1,
                "epochs": int(getattr(trainer, "epochs", parameters["epochs"])),
                "metrics": {
                    str(key): float(value)
                    for key, value in getattr(trainer, "metrics", {}).items()
                    if isinstance(value, (int, float))
                },
            }
        )

    aug_crop = bool(parameters.get("aug_crop", True))
    aug_occlusion = bool(parameters.get("aug_occlusion", True))
    parameters["aug_crop"] = aug_crop
    parameters["aug_occlusion"] = aug_occlusion
    arguments = {
        "data": str(staged_data),
        "epochs": parameters["epochs"],
        "imgsz": parameters["imgsz"],
        "batch": parameters["batch"],
        "device": parameters["device"] or None,
        "workers": parameters["workers"],
        "optimizer": parameters["optimizer"],
        "lr0": parameters["lr0"],
        "patience": parameters["patience"],
        "seed": parameters["seed"],
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        "pretrained": parameters["pretrained"],
        "cache": parameters["cache"],
        "amp": parameters["amp"],
        "close_mosaic": parameters["close_mosaic"],
        "degrees": parameters["degrees"],
        "fliplr": parameters["fliplr"],
        "hsv_h": parameters["hsv_h"],
        "hsv_s": parameters["hsv_s"],
        "hsv_v": parameters["hsv_v"],
        "plots": True,
        "verbose": True,
    }

    if expected_task in {"pose", "detect"}:
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.models.yolo.pose import PoseTrainer

        from .augmentations import apply_detection_augmentations, apply_pose_augmentations

        class _AugmentedDatasetMixin:
            augment_task = None

            def build_dataset(self, img_path, mode="train", batch=None):
                dataset = super().build_dataset(img_path, mode, batch)
                if mode == "train":
                    crop = bool(getattr(self, "aug_crop", True))
                    occlusion = bool(getattr(self, "aug_occlusion", True))
                    if self.augment_task == "pose":
                        apply_pose_augmentations(dataset, crop, occlusion)
                    elif self.augment_task == "detect":
                        apply_detection_augmentations(dataset, crop, occlusion)
                return dataset

        class _AugmentedPoseTrainer(_AugmentedDatasetMixin, PoseTrainer):
            augment_task = "pose"

        class _AugmentedDetectionTrainer(_AugmentedDatasetMixin, DetectionTrainer):
            augment_task = "detect"

        trainer_class = (
            _AugmentedPoseTrainer if expected_task == "pose" else _AugmentedDetectionTrainer
        )
        trainer = trainer_class(overrides={**arguments, "model": parameters["model"] or "yolo11n.pt"})
        trainer.aug_crop = aug_crop
        trainer.aug_occlusion = aug_occlusion
        trainer.add_callback("on_train_epoch_end", on_epoch_end)
        trainer.train()
    else:
        model.add_callback("on_train_epoch_end", on_epoch_end)
        model.train(**arguments)

def recommend_training_parameters(sample_count: int) -> dict[str, Any]:
    """Return sensible training hyper-parameters based on the labelled sample count."""
    count = max(1, int(sample_count))
    epochs = max(30, min(300, round(2000 / math.sqrt(count))))
    if count < 64:
        batch = 8
    elif count < 512:
        batch = 16
    else:
        batch = 32
    return {
        "epochs": epochs,
        "batch": batch,
        "imgsz": 640,
        "workers": 8,
        "patience": 100,
        "close_mosaic": 10,
    }
