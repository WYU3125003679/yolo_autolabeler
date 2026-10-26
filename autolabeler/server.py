"""Small, local-only HTTP API and static web server."""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import time
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MODEL_EXTENSIONS = {".pt", ".onnx", ".engine"}
LABEL_FORMATS = {"yolo", "yolo-obb", "yolo-pose"}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024
def _write_text_lf(path: Path, content: str) -> None:
    """Write text with LF newlines without requiring Python 3.10's ``newline=``."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


DEFAULT_KEYS = {
    "add": "a",
    "previous": "ArrowLeft",
    "next": "ArrowRight",
    "save": "Enter",
    "delete": "Delete",
    "keypoint": "k",
    "move_keypoint": "m",
    "class_1": "e",
    "class_2": "r",
    "class_3": "",
    "class_4": "",
    "class_5": "",
    "class_6": "",
    "class_7": "",
    "class_8": "",
    "class_9": "",
    "keypoint_visible": "v",
    "keypoint_occluded": "o",
    "keypoint_invisible": "i",
}


def natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Sort digit runs numerically while keeping text runs case-insensitive."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
    )


class WorkspaceError(ValueError):
    """An invalid or inaccessible path within the workspace."""


class JobBusyError(RuntimeError):
    """A training or inference task is already using the worker."""


class AutolabelerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], workspace: Path):
        self.workspace = workspace.resolve()
        self.state_dir = self.workspace / ".autolabeler"
        self.run_dir = self.state_dir / "run"
        self.models_dir = self.state_dir / "models"
        self.manifest_path = self.state_dir / "manually_labeled.txt"
        self.keys_path = Path(__file__).resolve().parent / "key.yaml"
        self.base_models_dir = (Path(__file__).resolve().parent.parent / "baseModels").resolve()
        self.settings_path = self.workspace / "autolabel-setting.json"
        self.settings_lock = threading.Lock()
        self.manifest_lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.active_job: str | None = None
        self._ensure_settings_file()
        super().__init__(address, AutolabelerHandler)

    @staticmethod
    def _default_settings() -> dict[str, Any]:
        return {
            "image_dir": "",
            "label_dir": ".autolabeler/labels",
            "format": "yolo",
            "auto_copy_labels": False,
            "auto_save": False,
            "nav_filter": "none",
            "auto_count": 10,
            "data": "",
            "model": "",
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "device": "cpu",
            "workers": 8,
            "patience": 100,
            "optimizer": "auto",
            "lr0": 0.01,
            "seed": 0,
            "degrees": 0.0,
            "fliplr": 0.5,
            "close_mosaic": 10,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "pretrained": True,
            "cache": False,
            "amp": True,
            "generate_data": False,
            "aug_crop": True,
            "aug_occlusion": True,
            "selected_image": "",
            "label_states": {},
        }

    def _ensure_settings_file(self) -> None:
        """Create the workspace settings file on startup when it is missing."""
        try:
            if not self.settings_path.exists():
                self._write_settings(self._default_settings())
        except OSError:
            pass

    def _sanitize_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        defaults = self._default_settings()
        sanitized: dict[str, Any] = {}
        for key, default in defaults.items():
            value = data.get(key, default)
            if isinstance(default, dict):
                if not isinstance(value, dict):
                    value = default
                elif key == "label_states":
                    cleaned: dict[str, str] = {}
                    for path, status in value.items():
                        if isinstance(status, str) and status in {"unannotated", "buffer", "complete"}:
                            cleaned[str(path)] = status
                    value = cleaned
            elif isinstance(default, bool):
                value = value if isinstance(value, bool) else default
            elif isinstance(default, int):
                value = value if isinstance(value, int) and not isinstance(value, bool) else default
            elif isinstance(default, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    value = default
                else:
                    value = float(value)
            else:
                value = value if isinstance(value, str) else default
            sanitized[key] = value
        return sanitized

    def _load_settings(self) -> dict[str, Any]:
        try:
            if self.settings_path.is_file():
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return self._sanitize_settings(data)
        except (OSError, ValueError):
            pass
        return self._default_settings()

    def _write_settings(self, settings: dict[str, Any]) -> None:
        with self.settings_lock:
            self.settings_path.write_text(
                json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


class AutolabelerHandler(BaseHTTPRequestHandler):
    server: AutolabelerServer

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/":
            self._serve_web_file("index.html")
        elif parsed.path.startswith("/web/"):
            self._serve_web_file(parsed.path.removeprefix("/web/"))
        elif parsed.path == "/api/browse":
            self._browse(query)
        elif parsed.path == "/api/image":
            self._serve_image(query)
        elif parsed.path == "/api/labels":
            self._get_labels(query)
        elif parsed.path == "/api/label-states":
            self._get_label_states(query)
        elif parsed.path == "/api/models":
            self._list_models()
        elif parsed.path == "/api/base-models":
            self._list_base_models()
        elif parsed.path == "/api/model-info":
            self._model_info(query)
        elif parsed.path == "/api/keys":
            self._get_keys()
        elif parsed.path == "/api/settings":
            self._get_settings()
        elif parsed.path == "/api/train-params":
            self._recommend_training(query)
        elif parsed.path == "/api/jobs":
            self._get_job(urllib.parse.parse_qs(parsed.query))
        else:
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/api/labels":
            self._save_labels()
        elif parsed.path == "/api/labels/review":
            self._review_labels()
        elif parsed.path == "/api/train":
            self._train()
        elif parsed.path == "/api/auto-label":
            self._auto_label()
        elif parsed.path == "/api/models/upload":
            self._upload_model(urllib.parse.parse_qs(parsed.query))
        elif parsed.path == "/api/keys":
            self._save_keys()
        elif parsed.path == "/api/settings":
            self._save_settings()
        elif parsed.path == "/api/jobs/cancel":
            self._cancel_job()
        else:
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[autolabeler] {self.address_string()} - {fmt % args}")

    def _json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "")
        if not raw_length.isdigit():
            raise ValueError("A valid Content-Length header is required")
        length = int(raw_length)
        if length > MAX_JSON_BYTES:
            raise OverflowError("Request body is too large")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON request body must be an object")
        return data

    def _serve_web_file(self, name: str) -> None:
        web_root = (Path(__file__).resolve().parent / "web").resolve()
        candidate = (web_root / name).resolve()
        if not candidate.is_relative_to(web_root) or not candidate.is_file():
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _workspace_path(self, value: Any) -> Path:
        if not isinstance(value, str):
            raise WorkspaceError("A workspace-relative path is required")
        relative = Path(value.replace("\\", "/"))
        if relative.is_absolute() or any(part == ".." for part in relative.parts):
            raise WorkspaceError("Path must stay inside the workspace")
        target = (self.server.workspace / relative).resolve()
        if not target.is_relative_to(self.server.workspace):
            raise WorkspaceError("Path must stay inside the workspace")
        return target

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self.server.workspace).as_posix()

    @staticmethod
    def _query_value(query: dict[str, list[str]], key: str, default: str = "") -> str:
        return query.get(key, [default])[0]

    def _browse(self, query: dict[str, list[str]]) -> None:
        path = self._workspace_path(self._query_value(query, "path"))
        if not path.exists():
            path = self.server.workspace
        if not path.is_dir():
            raise WorkspaceError("The selected path is not a directory")
        directories = []
        images = []
        for child in sorted(
            path.iterdir(),
            key=lambda item: (not item.is_dir(), natural_sort_key(item.name)),
        ):
            try:
                resolved = child.resolve()
                if not resolved.is_relative_to(self.server.workspace):
                    continue
            except OSError:
                continue
            if child.is_dir():
                directories.append({"name": child.name, "path": self._relative_path(child)})
            elif child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                images.append({"name": child.name, "path": self._relative_path(child)})
        self._json_response(
            HTTPStatus.OK,
            {
                "path": self._relative_path(path),
                "parent": self._relative_path(path.parent) if path != self.server.workspace else None,
                "directories": directories,
                "images": images,
            },
        )

    def _serve_image(self, query: dict[str, list[str]]) -> None:
        path = self._workspace_path(self._query_value(query, "path"))
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise WorkspaceError("The selected path is not a supported image")
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _label_path(self, image: Path, image_dir: Path, label_dir: Path) -> Path:
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            raise WorkspaceError("The selected path is not a supported image")
        if not image_dir.is_dir():
            raise WorkspaceError("The selected image directory does not exist")
        if label_dir.exists() and not label_dir.is_dir():
            raise WorkspaceError("The selected label path is not a directory")
        try:
            relative = image.relative_to(image_dir)
        except ValueError as exc:
            raise WorkspaceError("Image must be inside the selected image directory") from exc
        label_path = (label_dir / relative).with_suffix(".txt")
        if not label_path.resolve().is_relative_to(self.server.workspace):
            raise WorkspaceError("Label path must stay inside the workspace")
        return label_path

    def _buffer_dir(self, image_dir: Path) -> Path:
        relative_dir = image_dir.relative_to(self.server.workspace)
        return self.server.state_dir / "buffer" / (relative_dir if relative_dir.parts else Path("_root"))

    def _buffer_label_path(self, image: Path, image_dir: Path) -> Path:
        relative_image = image.relative_to(image_dir)
        buffer_dir = self._buffer_dir(image_dir)
        label_path = (buffer_dir / relative_image).with_suffix(".txt")
        if not label_path.resolve().is_relative_to(self.server.state_dir.resolve()):
            raise WorkspaceError("Buffer label path must stay inside the workspace state directory")
        return label_path

    def _buffer_pending_path(self, buffer_label_path: Path) -> Path:
        return buffer_label_path.with_suffix(".pending")

    def _label_status(self, image: Path, image_dir: Path, label_dir: Path) -> str:
        final_path = self._label_path(image, image_dir, label_dir)
        if final_path.is_file():
            return "complete"
        return "buffer" if self._buffer_label_path(image, image_dir).is_file() else "unannotated"

    def _get_labels(self, query: dict[str, list[str]]) -> None:
        image = self._workspace_path(self._query_value(query, "image"))
        image_dir = self._workspace_path(self._query_value(query, "image_dir"))
        label_dir = self._workspace_path(self._query_value(query, "label_dir"))
        label_path = self._label_path(image, image_dir, label_dir)
        buffer_path = self._buffer_label_path(image, image_dir)
        pending = self._buffer_pending_path(buffer_path).is_file()
        status = self._label_status(image, image_dir, label_dir)
        source_path = label_path if status == "complete" else buffer_path
        content = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
        self._json_response(
            HTTPStatus.OK,
            {
                "content": content,
                "label_path": self._relative_path(label_path),
                "buffer_path": self._relative_path(buffer_path),
                "status": status,
                "pending": pending,
            },
        )

    def _get_label_states(self, query: dict[str, list[str]]) -> None:
        image_dir = self._workspace_path(self._query_value(query, "image_dir"))
        label_dir = self._workspace_path(self._query_value(query, "label_dir"))
        if not image_dir.is_dir():
            raise WorkspaceError("The selected image directory does not exist")
        states = {}
        for image in image_dir.rglob("*"):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                resolved = image.resolve()
                if resolved.is_relative_to(self.server.workspace):
                    states[self._relative_path(image)] = self._label_status(image, image_dir, label_dir)
        self._json_response(HTTPStatus.OK, {"states": states})

    def _save_labels(self) -> None:
        data = self._read_json()
        image = self._workspace_path(data.get("image"))
        image_dir = self._workspace_path(data.get("image_dir"))
        label_dir = self._workspace_path(data.get("label_dir"))
        label_path = self._label_path(image, image_dir, label_dir)
        content = data.get("content")
        if not isinstance(content, str) or len(content.encode("utf-8")) > 1_000_000:
            raise ValueError("Label content must be text smaller than 1 MB")
        label_format = data.get("format", "yolo")
        if not isinstance(label_format, str) or label_format not in LABEL_FORMATS:
            raise ValueError(f"format must be one of {', '.join(sorted(LABEL_FORMATS))}")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_lf(label_path, content)
        self._buffer_label_path(image, image_dir).unlink(missing_ok=True)
        self._record_manual_label(image, image_dir, label_dir, label_format)
        self._json_response(
            HTTPStatus.OK,
            {"saved": True, "label_path": self._relative_path(label_path)},
        )

    def _review_labels(self) -> None:
        data = self._read_json()
        image = self._workspace_path(data.get("image"))
        image_dir = self._workspace_path(data.get("image_dir"))
        label_dir = self._workspace_path(data.get("label_dir"))
        final_path = self._label_path(image, image_dir, label_dir)
        buffer_path = self._buffer_label_path(image, image_dir)
        if final_path.exists():
            self._json_response(HTTPStatus.CONFLICT, {"error": "This image is already marked complete"})
            return
        if not buffer_path.is_file():
            raise WorkspaceError("No buffered annotation exists for this image")
        content = data.get("content")
        if not isinstance(content, str) or len(content.encode("utf-8")) > 1_000_000:
            raise ValueError("Label content must be text smaller than 1 MB")
        label_format = data.get("format", "yolo")
        if not isinstance(label_format, str) or label_format not in LABEL_FORMATS:
            raise ValueError(f"format must be one of {', '.join(sorted(LABEL_FORMATS))}")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = final_path.with_name(final_path.name + ".autolabeler-tmp")
        try:
            _write_text_lf(temporary, content)
            temporary.replace(final_path)
        finally:
            temporary.unlink(missing_ok=True)
        buffer_path.unlink(missing_ok=True)
        self._record_manual_label(image, image_dir, label_dir, label_format)
        self._json_response(
            HTTPStatus.OK,
            {"saved": True, "label_path": self._relative_path(final_path), "status": "complete"},
        )

    def _record_manual_label(
        self,
        image: Path,
        image_dir: Path,
        label_dir: Path,
        label_format: str,
    ) -> None:
        self.server.state_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "image": self._relative_path(image),
            "image_dir": self._relative_path(image_dir),
            "label_dir": self._relative_path(label_dir),
            "format": label_format,
        }
        with self.server.manifest_lock:
            records: dict[str, dict[str, str]] = {}
            if self.server.manifest_path.is_file():
                for line in self.server.manifest_path.read_text(encoding="utf-8").splitlines():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        record = {"image": line}
                    if isinstance(record, dict) and isinstance(record.get("image"), str):
                        records[record["image"]] = record
            records[entry["image"]] = entry
            self.server.manifest_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records.values()),
                encoding="utf-8",
            )

    def _model_entries(self) -> list[dict[str, str]]:
        entries = []
        for entry in self._base_model_entries():
            entries.append(
                {
                    "name": entry["name"],
                    "path": f"base:{entry['name']}",
                    "source": f"baseModels/{entry['name']}",
                }
            )
        for root in (self.server.models_dir, self.server.run_dir):
            if not root.is_dir():
                continue
            for model in sorted(root.rglob("*")):
                if model.is_file() and model.suffix.lower() in MODEL_EXTENSIONS:
                    entries.append(
                        {"name": model.name, "path": self._relative_path(model), "source": self._relative_path(model)}
                    )
        return entries

    def _list_models(self) -> None:
        self._json_response(HTTPStatus.OK, {"models": self._model_entries()})

    def _base_model_entries(self) -> list[dict[str, str]]:
        entries = []
        root = self.server.base_models_dir
        if not root.is_dir():
            return entries
        for model in sorted(root.iterdir()):
            if model.is_file() and model.suffix.lower() in MODEL_EXTENSIONS:
                entries.append({"name": model.name, "path": model.name})
        return entries

    def _list_base_models(self) -> None:
        self._json_response(HTTPStatus.OK, {"models": self._base_model_entries()})

    def _model_info(self, query: dict[str, list[str]]) -> None:
        value = self._query_value(query, "model")
        if not value:
            raise ValueError("Select a model first")
        model_path = self._resolve_model(value)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed in the Python environment running autolabeler"
            ) from exc
        model = YOLO(model_path)
        task = getattr(model, "task", "")
        keypoints = 0
        if task == "pose":
            shape = getattr(getattr(model, "model", None), "kpt_shape", None)
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                keypoints = int(shape[0])
        names = getattr(model, "names", None)
        name_list = []
        if isinstance(names, dict):
            name_list = [str(names.get(index, str(index))) for index in range(len(names))]
        self._json_response(
            HTTPStatus.OK,
            {"task": task, "keypoints": keypoints, "names": name_list},
        )

    def _resolve_model(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Select a model or enter a model name")
        model_name = value.strip()
        if model_name.startswith("base:"):
            base_name = Path(model_name[5:].replace("\\", "/")).name
            base_path = self.server.base_models_dir / base_name
            if not base_path.is_file() or base_path.suffix.lower() not in MODEL_EXTENSIONS:
                raise WorkspaceError("The selected base model does not exist")
            return str(base_path)
        if Path(model_name).is_absolute():
            model_path = Path(model_name)
            if not model_path.is_file() or model_path.suffix.lower() not in MODEL_EXTENSIONS:
                raise WorkspaceError("The selected local model file does not exist")
            return str(model_path)
        if "/" not in model_name and "\\" not in model_name:
            return model_name
        model_path = self._workspace_path(model_name)
        if not model_path.is_file() or model_path.suffix.lower() not in MODEL_EXTENSIONS:
            raise WorkspaceError("The selected model file does not exist in the workspace")
        return str(model_path)

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self.server.jobs_lock:
            job = self.server.jobs[job_id]
            job.update(updates)
            snapshot = {key: value for key, value in job.items() if key != "cancel_event"}
            run_dir = Path(job["run_dir"])
        status_path = run_dir / "status.json"
        status_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    def _start_job(
        self,
        kind: str,
        run_dir: Path,
        worker: Any,
    ) -> str:
        with self.server.jobs_lock:
            if self.server.active_job is not None:
                active = self.server.jobs[self.server.active_job]
                if active["status"] in {"queued", "running", "cancelling"}:
                    raise JobBusyError("Another training or inference job is already active")
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "kind": kind,
                "status": "queued",
                "progress": {},
                "error": None,
                "run": self._relative_path(run_dir),
                "run_dir": str(run_dir),
                "cancel_event": threading.Event(),
            }
            self.server.jobs[job_id] = job
            self.server.active_job = job_id

        self._update_job(job_id)

        def run() -> None:
            self._update_job(job_id, status="running", started_at=time.time())
            try:
                worker(
                    lambda progress: self._update_job(job_id, progress=progress),
                    lambda: self.server.jobs[job_id]["cancel_event"].is_set(),
                )
                cancelled = self.server.jobs[job_id]["cancel_event"].is_set()
                self._update_job(
                    job_id,
                    status="cancelled" if cancelled else "completed",
                    completed_at=time.time(),
                )
            except Exception as exc:
                self._update_job(
                    job_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    completed_at=time.time(),
                )
                print(f"[autolabeler] {kind} job {job_id} failed: {exc}")
            finally:
                with self.server.jobs_lock:
                    if self.server.active_job == job_id:
                        self.server.active_job = None

        threading.Thread(target=run, name=f"autolabeler-{kind}-{job_id[:8]}", daemon=True).start()
        return job_id

    def _get_job(self, query: dict[str, list[str]]) -> None:
        job_id = self._query_value(query, "id")
        with self.server.jobs_lock:
            job = self.server.jobs.get(job_id)
            if job is None:
                raise WorkspaceError("Job not found")
            result = {key: value for key, value in job.items() if key not in {"cancel_event", "run_dir"}}
        self._json_response(HTTPStatus.OK, result)

    def _cancel_job(self) -> None:
        job_id = self._read_json().get("id")
        if not isinstance(job_id, str):
            raise ValueError("Job id is required")
        with self.server.jobs_lock:
            job = self.server.jobs.get(job_id)
            if job is None:
                raise WorkspaceError("Job not found")
            if job["status"] not in {"queued", "running", "cancelling"}:
                raise ValueError("This job is no longer active")
            job["cancel_event"].set()
            job["status"] = "cancelling"
        self._update_job(job_id)
        self._json_response(HTTPStatus.OK, {"id": job_id, "status": "cancelling"})

    def _upload_model(self, query: dict[str, list[str]]) -> None:
        raw_length = self.headers.get("Content-Length", "")
        if not raw_length.isdigit():
            raise ValueError("A valid Content-Length header is required")
        length = int(raw_length)
        if length < 1 or length > MAX_MODEL_BYTES:
            raise OverflowError("Model file must be between 1 byte and 2 GB")
        name = urllib.parse.unquote(self._query_value(query, "name", ""))
        safe_name = Path(name.replace("\\", "/")).name
        if not safe_name or Path(safe_name).suffix.lower() not in MODEL_EXTENSIONS:
            raise ValueError("Choose a .pt, .onnx, or .engine model file")

        self.server.models_dir.mkdir(parents=True, exist_ok=True)
        destination = self.server.models_dir / safe_name
        suffix = destination.suffix
        stem = destination.stem
        index = 1
        while destination.exists():
            destination = self.server.models_dir / f"{stem}-{index}{suffix}"
            index += 1
        temporary = destination.with_name(destination.name + f".{threading.get_ident()}.upload")
        try:
            with temporary.open("wb") as output:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Model upload ended before all bytes were received")
                    output.write(chunk)
                    remaining -= len(chunk)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        self._json_response(HTTPStatus.OK, {"uploaded": True, "path": self._relative_path(destination)})

    def _read_keys(self) -> dict[str, str]:
        if not self.server.keys_path.is_file():
            return DEFAULT_KEYS.copy()
        result: dict[str, str] = {}
        in_shortcuts = False
        for line in self.server.keys_path.read_text(encoding="utf-8").splitlines():
            if line.strip() == "shortcuts:":
                in_shortcuts = True
            elif in_shortcuts:
                match = re.fullmatch(r'\s+([a-z_]+):\s*"([^"]*)"\s*', line)
                if match:
                    result[match.group(1)] = match.group(2)
        return {**DEFAULT_KEYS, **result}

    def _get_settings(self) -> None:
        self._json_response(HTTPStatus.OK, {"settings": self.server._load_settings()})

    def _save_settings(self) -> None:
        data = self._read_json().get("settings")
        if not isinstance(data, dict):
            raise ValueError("Settings must be an object")
        settings = self.server._sanitize_settings(data)
        self.server._write_settings(settings)
        self._json_response(HTTPStatus.OK, {"settings": settings})

    def _get_keys(self) -> None:
        self._json_response(HTTPStatus.OK, {"shortcuts": self._read_keys()})

    def _save_keys(self) -> None:
        data = self._read_json().get("shortcuts")
        if not isinstance(data, dict) or not set(data).issubset(DEFAULT_KEYS):
            raise ValueError("Unknown shortcut action")
        shortcuts = DEFAULT_KEYS.copy()
        for action, key in data.items():
            if not isinstance(key, str) or len(key) > 32 or any(char in key for char in '\r\n"'):
                raise ValueError("Shortcut values must be short key names")
            shortcuts[action] = key
        content = "shortcuts:\n" + "".join(
            f'  {action}: "{key}"\n' for action, key in shortcuts.items()
        )
        self.server.keys_path.write_text(content, encoding="utf-8")
        self._json_response(HTTPStatus.OK, {"shortcuts": shortcuts})

    def _recommend_training(self, query: dict[str, list[str]]) -> None:
        image_dir_value = self._query_value(query, "image_dir")
        label_dir_value = self._query_value(query, "label_dir")
        label_format = self._query_value(query, "format", "yolo")
        if not image_dir_value:
            raise WorkspaceError("Select an image directory first")
        if not label_dir_value:
            raise WorkspaceError("Select a label directory first")
        if label_format not in LABEL_FORMATS:
            raise ValueError(f"format must be one of {', '.join(sorted(LABEL_FORMATS))}")
        image_dir = self._workspace_path(image_dir_value)
        label_dir = self._workspace_path(label_dir_value)
        if not image_dir.is_dir():
            raise WorkspaceError("The selected image directory does not exist")
        if label_dir.exists() and not label_dir.is_dir():
            raise WorkspaceError("The selected label path is not a directory")
        from .training import iter_labeled_pairs, recommend_training_parameters

        count = sum(1 for _ in iter_labeled_pairs(image_dir, label_dir))
        self._json_response(
            HTTPStatus.OK,
            {"count": count, "parameters": recommend_training_parameters(count)},
        )

    def _train(self) -> None:
        data = self._read_json()
        parameters = data.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("Training parameters are required")
        for field, minimum, maximum in (("epochs", 1, 10000), ("imgsz", 32, 4096), ("batch", 1, 2048)):
            value = parameters.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{field} must be between {minimum} and {maximum}")

        label_format = data.get("format", "yolo")
        if not isinstance(label_format, str) or label_format not in LABEL_FORMATS:
            raise ValueError(f"format must be one of {', '.join(sorted(LABEL_FORMATS))}")

        if not isinstance(parameters.get("generate_data", False), bool):
            raise ValueError("generate_data must be true or false")
        generate_data = parameters["generate_data"]
        image_dir = self._workspace_path(parameters.get("image_dir"))
        label_dir = self._workspace_path(parameters.get("label_dir"))
        if not image_dir.is_dir():
            raise WorkspaceError("The selected image directory does not exist")
        if label_dir.exists() and not label_dir.is_dir():
            raise WorkspaceError("The selected label path is not a directory")

        if generate_data:
            dataset_config = None
            parameters["data"] = ""
        else:
            if not isinstance(parameters.get("data"), str) or not parameters["data"]:
                raise ValueError("Select a dataset YAML file or enable automatic generation")
            dataset_config = self._workspace_path(parameters["data"])
            if not dataset_config.is_file() or dataset_config.suffix.lower() not in {".yaml", ".yml"}:
                raise WorkspaceError("Select an existing dataset YAML file inside the workspace")
            parameters["data"] = str(dataset_config)

        parameters["model"] = self._resolve_model(parameters.get("model") or "yolo11n.pt")
        parameters["workspace"] = str(self.server.workspace)
        parameters["image_dir"] = str(image_dir)
        parameters["label_dir"] = str(label_dir)
        parameters["generate_data"] = generate_data

        for field, default, low, high in (
            ("workers", 8, 0, 256),
            ("patience", 100, 0, 10000),
            ("seed", 0, 0, 2**31 - 1),
            ("close_mosaic", 10, 0, 10000),
        ):
            value = parameters.get(field, default)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{field} must be an integer between {low} and {high}")
            parameters[field] = value
        for field, default, low, high in (
            ("lr0", 0.01, 0, 1),
            ("degrees", 0.0, 0, 180),
            ("fliplr", 0.5, 0, 1),
            ("hsv_h", 0.015, 0, 1),
            ("hsv_s", 0.7, 0, 1),
            ("hsv_v", 0.4, 0, 1),
        ):
            value = parameters.get(field, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"{field} must be a number between {low} and {high}")
            parameters[field] = float(value)
        optimizer = parameters.get("optimizer", "auto")
        if not isinstance(optimizer, str) or optimizer not in {
            "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"
        }:
            raise ValueError("Unsupported optimizer")
        parameters["optimizer"] = optimizer
        for field, default in (
            ("pretrained", True),
            ("cache", False),
            ("amp", True),
            ("aug_crop", True),
            ("aug_occlusion", True),
        ):
            value = parameters.get(field, default)
            if not isinstance(value, bool):
                raise ValueError(f"{field} must be true or false")
            parameters[field] = value
        device = parameters.get("device", "")
        if not isinstance(device, str) or not re.fullmatch(r"(?:cpu|mps|\d+(?:,\d+)*)?", device.strip()):
            raise ValueError("device must be cpu, mps, a GPU index such as 0, or a list such as 0,1")
        parameters["device"] = device.strip()
        run_name = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
        run_dir = self.server.run_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "request.json").write_text(
            json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        from .training import run_training

        job_id = self._start_job(
            "train",
            run_dir,
            lambda progress, cancelled: run_training(
                parameters,
                run_dir,
                image_dir,
                label_dir,
                dataset_config,
                label_format,
                progress,
                cancelled,
            ),
        )
        self._json_response(
            HTTPStatus.ACCEPTED,
            {"id": job_id, "status": "queued", "run": self._relative_path(run_dir)},
        )

    def _auto_label(self) -> None:
        data = self._read_json()
        count = data.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100000:
            raise ValueError("count must be an integer between 1 and 100000")
        if not isinstance(data.get("model"), str) or not data["model"]:
            raise ValueError("Select a model before starting automatic annotation")
        image_dir = self._workspace_path(data.get("image_dir"))
        label_dir = self._workspace_path(data.get("label_dir"))
        if not image_dir.is_dir():
            raise WorkspaceError("The selected image directory does not exist")
        if label_dir.exists() and not label_dir.is_dir():
            raise WorkspaceError("The selected label path is not a directory")
        label_format = data.get("format", "yolo")
        if not isinstance(label_format, str) or label_format not in LABEL_FORMATS:
            raise ValueError(f"format must be one of {', '.join(sorted(LABEL_FORMATS))}")
        enabled_keypoints = data.get("enabled_keypoints")
        if enabled_keypoints is not None:
            if isinstance(enabled_keypoints, dict):
                parsed = {}
                for key, value in enabled_keypoints.items():
                    try:
                        cls = int(key)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("enabled_keypoints class keys must be integers") from exc
                    if not isinstance(value, list) or not all(
                        isinstance(index, int) and not isinstance(index, bool) and index >= 0
                        for index in value
                    ):
                        raise ValueError("enabled_keypoints values must be lists of non-negative integers")
                    parsed[cls] = sorted(set(value))
                enabled_keypoints = parsed
            elif isinstance(enabled_keypoints, list):
                if not all(
                    isinstance(index, int) and not isinstance(index, bool) and index >= 0
                    for index in enabled_keypoints
                ):
                    raise ValueError("enabled_keypoints must be a list of non-negative integers")
                enabled_keypoints = sorted(set(enabled_keypoints))
            else:
                raise ValueError("enabled_keypoints must be a mapping of classes to keypoint indices")
        model_path = self._resolve_model(data["model"])
        images = []
        buffer_dir = self._buffer_dir(image_dir)
        for image in sorted(image_dir.rglob("*"), key=lambda item: natural_sort_key(item.relative_to(image_dir).as_posix())):
            if image.suffix.lower() not in IMAGE_EXTENSIONS or not image.is_file():
                continue
            resolved = image.resolve()
            if not resolved.is_relative_to(self.server.workspace):
                continue
            relative = image.relative_to(image_dir)
            label_path = (label_dir / relative).with_suffix(".txt")
            buffer_path = (buffer_dir / relative).with_suffix(".txt")
            pending_path = self._buffer_pending_path(buffer_path)
            if not label_path.exists() and (
                not buffer_path.is_file() or pending_path.is_file()
            ):
                images.append(image)
            if len(images) == count:
                break
        if not images:
            raise ValueError("No unlabelled images were found in the selected image directory")

        run_dir = self.server.run_dir / ("predict-" + time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:6]}")
        run_dir.mkdir(parents=True, exist_ok=False)
        request = {
            "model": model_path,
            "count": count,
            "format": label_format,
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "buffer_dir": str(buffer_dir),
            "selected_images": [self._relative_path(image) for image in images],
        }
        (run_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        from .inference import run_inference

        created_buffer_paths = []
        created_pending_paths = []
        for image in images:
            buffer_path = self._buffer_label_path(image, image_dir)
            buffer_path.parent.mkdir(parents=True, exist_ok=True)
            if not buffer_path.exists():
                buffer_path.touch()
                created_buffer_paths.append(buffer_path)
            pending_path = self._buffer_pending_path(buffer_path)
            if not pending_path.exists():
                pending_path.touch()
                created_pending_paths.append(pending_path)

        def run_auto_label(progress: Any, cancelled: Any) -> None:
            def report_progress(progress_data: dict[str, Any]) -> None:
                relative_image = progress_data.get("image")
                if isinstance(relative_image, str):
                    completed_buffer = (buffer_dir / Path(relative_image)).with_suffix(".txt")
                    self._buffer_pending_path(completed_buffer).unlink(missing_ok=True)
                progress(progress_data)

            run_inference(
                model_path,
                images,
                buffer_dir,
                image_dir,
                label_format,
                report_progress,
                cancelled,
                enabled_keypoints,
            )

        try:
            job_id = self._start_job(
                "auto-label",
                run_dir,
                run_auto_label,
            )
        except JobBusyError:
            for buffer_path in created_buffer_paths:
                buffer_path.unlink(missing_ok=True)
            for pending_path in created_pending_paths:
                pending_path.unlink(missing_ok=True)
            raise
        self._json_response(
            HTTPStatus.ACCEPTED,
            {
                "id": job_id,
                "status": "queued",
                "run": self._relative_path(run_dir),
                "count": len(images),
                "images": [self._relative_path(image) for image in images],
            },
        )

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except WorkspaceError as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except OverflowError as exc:
            self._json_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": str(exc)})
        except JobBusyError as exc:
            self._json_response(HTTPStatus.CONFLICT, {"error": str(exc)})
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except OSError as exc:
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        except Exception as exc:
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Request failed: {exc}"})


def make_server(address: tuple[str, int], workspace: Path) -> AutolabelerServer:
    return AutolabelerServer(address, workspace)
