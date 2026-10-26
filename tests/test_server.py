import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autolabeler.inference import serialize_result
from autolabeler.server import make_server
from autolabeler.training import prepare_training_dataset, recommend_training_parameters


class ServerApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.image_dir = self.workspace / "images"
        self.image_dir.mkdir()
        (self.image_dir / "sample.jpg").write_bytes(b"image fixture")
        self.server = make_server(("127.0.0.1", 0), self.workspace)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if body is not None else {}
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def wait_for_job(self, job_id):
        deadline = time.time() + 5
        while time.time() < deadline:
            status, data = self.request(f"/api/jobs?id={job_id}")
            self.assertEqual(status, 200)
            if data["status"] in {"completed", "failed", "cancelled"}:
                return data
            time.sleep(0.02)
        self.fail("Background job did not finish in time")

    def test_browse_save_and_reload_labels(self):
        for name in ("frame10.jpg", "frame2.jpg", "frame1.jpg"):
            (self.image_dir / name).write_bytes(b"image fixture")
        status, data = self.request("/api/browse?path=images")
        self.assertEqual(status, 200)
        self.assertEqual(
            [image["name"] for image in data["images"]],
            ["frame1.jpg", "frame2.jpg", "frame10.jpg", "sample.jpg"],
        )

        payload = {
            "image": "images/sample.jpg",
            "image_dir": "images",
            "label_dir": "labels",
            "format": "yolo",
            "content": "2 0.5 0.5 0.4 0.3\n",
        }
        status, data = self.request("/api/labels", payload)
        self.assertEqual(status, 200)
        self.assertEqual((self.workspace / "labels/sample.txt").read_text(), payload["content"])
        manifest_entry = json.loads(
            (self.workspace / ".autolabeler/manually_labeled.txt").read_text()
        )
        self.assertEqual(manifest_entry["image"], "images/sample.jpg")
        self.assertEqual(manifest_entry["label_dir"], "labels")

        status, data = self.request(
            "/api/labels?image=images%2Fsample.jpg&image_dir=images&label_dir=labels"
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["content"], payload["content"])

    def test_buffered_labels_are_reviewed_before_becoming_complete(self):
        buffer_path = self.workspace / ".autolabeler/buffer/images/sample.txt"
        buffer_path.parent.mkdir(parents=True)
        buffer_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        status, data = self.request(
            "/api/labels?image=images%2Fsample.jpg&image_dir=images&label_dir=labels"
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "buffer")
        self.assertEqual(data["content"], buffer_path.read_text(encoding="utf-8"))

        status, data = self.request("/api/label-states?image_dir=images&label_dir=labels")
        self.assertEqual(status, 200)
        self.assertEqual(data["states"]["images/sample.jpg"], "buffer")

        reviewed_content = "0 0.4 0.4 0.3 0.3\n"
        status, data = self.request(
            "/api/labels/review",
            {
                "image": "images/sample.jpg",
                "image_dir": "images",
                "label_dir": "labels",
                "format": "yolo",
                "content": reviewed_content,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual((self.workspace / "labels/sample.txt").read_text(), reviewed_content)
        self.assertFalse(buffer_path.exists())

        status, data = self.request("/api/label-states?image_dir=images&label_dir=labels")
        self.assertEqual(status, 200)
        self.assertEqual(data["states"]["images/sample.jpg"], "complete")

    def test_rejects_path_escape(self):
        status, data = self.request("/api/browse?path=..")
        self.assertEqual(status, 400)
        self.assertIn("inside the workspace", data["error"])

    def test_settings_are_created_and_persisted(self):
        settings_path = self.workspace / "autolabel-setting.json"
        self.assertTrue(settings_path.is_file())

        status, data = self.request("/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(data["settings"]["format"], "yolo")
        self.assertEqual(data["settings"]["label_dir"], ".autolabeler/labels")

        status, data = self.request(
            "/api/settings",
            {
                "settings": {
                    "image_dir": "images",
                    "label_dir": "labels",
                    "format": "yolo-obb",
                    "epochs": 50,
                    "pretrained": False,
                    "generate_data": True,
                    "selected_image": "images/sample.jpg",
                    "label_states": {"images/sample.jpg": "complete", "images/other.jpg": "buffer"},
                }
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["settings"]["image_dir"], "images")
        self.assertEqual(data["settings"]["epochs"], 50)
        self.assertEqual(data["settings"]["selected_image"], "images/sample.jpg")
        self.assertEqual(data["settings"]["label_states"]["images/other.jpg"], "buffer")

        status, data = self.request("/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(data["settings"]["format"], "yolo-obb")
        self.assertEqual(data["settings"]["generate_data"], True)
        self.assertEqual(data["settings"]["pretrained"], False)
        self.assertEqual(data["settings"]["label_states"]["images/sample.jpg"], "complete")

    def test_base_models_and_model_resolution(self):
        import autolabeler.server as server_module

        base_dir = self.workspace / "_base_models"
        base_dir.mkdir()
        (base_dir / "preset.pt").write_bytes(b"preset weights")
        self.server.base_models_dir = base_dir

        status, data = self.request("/api/base-models")
        self.assertEqual(status, 200)
        self.assertEqual([model["name"] for model in data["models"]], ["preset.pt"])

        handler = server_module.AutolabelerHandler.__new__(server_module.AutolabelerHandler)
        handler.server = self.server
        self.assertEqual(handler._resolve_model("base:preset.pt"), str(base_dir / "preset.pt"))
        self.assertEqual(handler._resolve_model("yolo11n.pt"), "yolo11n.pt")

        abs_model = self.workspace / "abs.pt"
        abs_model.write_bytes(b"weights")
        self.assertEqual(handler._resolve_model(str(abs_model)), str(abs_model))

    def test_training_requires_dataset_yaml_unless_auto_generate(self):
        status, data = self.request(
            "/api/train",
            {
                "parameters": {
                    "epochs": 3,
                    "imgsz": 640,
                    "batch": 8,
                    "data": "dataset.yaml",
                    "model": "yolo.pt",
                    "image_dir": "images",
                    "label_dir": "labels",
                    "generate_data": False,
                }
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("dataset YAML", data["error"])
        self.assertFalse((self.workspace / ".autolabeler/run").exists())

    def test_training_dataset_stages_labelled_images(self):
        import yaml

        dataset_yaml = self.workspace / "data.yaml"
        dataset_yaml.write_text(yaml.safe_dump({"names": ["cat", "dog"]}), encoding="utf-8")
        label = self.workspace / "labels" / "sample.txt"
        label.parent.mkdir()
        label.write_text("1 0.5 0.5 0.2 0.3\n", encoding="utf-8")
        run_dir = self.workspace / ".autolabeler" / "run" / "unit"
        staged_yaml = prepare_training_dataset(
            self.image_dir, self.workspace / "labels", dataset_yaml, run_dir, "yolo", 0
        )
        staged_data = yaml.safe_load(staged_yaml.read_text(encoding="utf-8"))
        self.assertEqual(staged_data["names"], {0: "cat", 1: "dog"})
        self.assertTrue((run_dir / "dataset/images/train/00000000.jpg").is_file())
        self.assertTrue((run_dir / "dataset/labels/train/00000000.txt").is_file())

    def test_pose_labels_with_visibility_values_validate(self):
        import yaml

        label = self.workspace / "labels" / "sample.txt"
        label.parent.mkdir()
        keypoints = " ".join(f"0.5 0.5 {index % 3}" for index in range(12))
        label.write_text(f"0 0.5 0.5 0.2 0.2 {keypoints}\n", encoding="utf-8")
        run_dir = self.workspace / ".autolabeler" / "run" / "pose"
        staged_yaml = prepare_training_dataset(
            self.image_dir, self.workspace / "labels", None, run_dir, "yolo-pose", 0
        )
        staged_data = yaml.safe_load(staged_yaml.read_text(encoding="utf-8"))
        self.assertEqual(staged_data["kpt_shape"], [12, 3])
        self.assertEqual(staged_data["names"], {0: "class_0"})
        self.assertTrue((run_dir / "dataset/images/train/00000000.jpg").is_file())

    def test_training_dataset_auto_generates_yaml_from_labels(self):
        import yaml

        label = self.workspace / "labels" / "sample.txt"
        label.parent.mkdir()
        label.write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
        run_dir = self.workspace / ".autolabeler" / "run" / "auto"
        staged_yaml = prepare_training_dataset(
            self.image_dir, self.workspace / "labels", None, run_dir, "yolo", 0
        )
        staged_data = yaml.safe_load(staged_yaml.read_text(encoding="utf-8"))
        self.assertEqual(staged_data["names"], {0: "class_0"})
        generated = yaml.safe_load((run_dir / "generated.yaml").read_text(encoding="utf-8"))
        self.assertEqual(generated["names"], ["class_0"])
        self.assertTrue((run_dir / "dataset/images/train/00000000.jpg").is_file())

    def test_training_auto_generate_passes_no_dataset_config(self):
        self.request(
            "/api/labels",
            {
                "image": "images/sample.jpg",
                "image_dir": "images",
                "label_dir": "labels",
                "format": "yolo",
                "content": "0 0.5 0.5 0.3 0.3\n",
            },
        )

        def fake_train(parameters, run_dir, image_dir, label_dir, dataset_config, label_format, progress, cancelled):
            self.assertIsNone(dataset_config)
            self.assertEqual(label_format, "yolo")
            progress({"epoch": 1, "epochs": parameters["epochs"]})

        with patch("autolabeler.training.run_training", side_effect=fake_train):
            status, response = self.request(
                "/api/train",
                {
                    "format": "yolo",
                    "parameters": {
                        "epochs": 1,
                        "imgsz": 64,
                        "batch": 1,
                        "model": "yolo11n.pt",
                        "image_dir": "images",
                        "label_dir": "labels",
                        "generate_data": True,
                    },
                },
            )
            self.assertEqual(status, 202)
            result = self.wait_for_job(response["id"])
        self.assertEqual(result["status"], "completed")

    def test_training_job_reports_progress_and_completion(self):
        import yaml

        self.request(
            "/api/labels",
            {
                "image": "images/sample.jpg",
                "image_dir": "images",
                "label_dir": "labels",
                "format": "yolo",
                "content": "0 0.5 0.5 0.3 0.3\n",
            },
        )
        dataset_yaml = self.workspace / "data.yaml"
        dataset_yaml.write_text(yaml.safe_dump({"names": ["object"]}), encoding="utf-8")

        def fake_train(parameters, run_dir, image_dir, label_dir, dataset_config, label_format, progress, cancelled):
            self.assertEqual(str(image_dir), str(self.image_dir))
            self.assertEqual(str(label_dir), str(self.workspace / "labels"))
            self.assertIsNotNone(dataset_config)
            self.assertEqual(label_format, "yolo")
            progress({"epoch": 1, "epochs": parameters["epochs"]})

        with patch("autolabeler.training.run_training", side_effect=fake_train):
            status, response = self.request(
                "/api/train",
                {
                    "format": "yolo",
                    "parameters": {
                        "epochs": 1,
                        "imgsz": 64,
                        "batch": 1,
                        "data": "data.yaml",
                        "model": "yolo11n.pt",
                        "image_dir": "images",
                        "label_dir": "labels",
                        "generate_data": False,
                    },
                },
            )
            self.assertEqual(status, 202)
            result = self.wait_for_job(response["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["progress"]["epoch"], 1)

    def test_auto_label_job_uses_selected_image_count(self):
        (self.image_dir / "second.jpg").write_bytes(b"second image")
        existing_label = self.workspace / "labels" / "sample.txt"
        existing_label.parent.mkdir()
        existing_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        model_path = self.workspace / ".autolabeler" / "models" / "model.pt"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"test weights")

        def fake_inference(model, images, label_dir, image_dir, label_format, progress, cancelled, enabled_keypoints=None):
            selected = list(images)
            self.assertEqual(len(selected), 1)
            self.assertEqual(label_format, "yolo")
            label = label_dir / selected[0].relative_to(image_dir).with_suffix(".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            progress({
                "completed": 1,
                "total": 1,
                "image": selected[0].relative_to(image_dir).as_posix(),
            })

        with patch("autolabeler.inference.run_inference", side_effect=fake_inference):
            status, response = self.request(
                "/api/auto-label",
                {
                    "count": 1,
                    "model": ".autolabeler/models/model.pt",
                    "image_dir": "images",
                    "label_dir": "labels",
                    "format": "yolo",
                },
            )
            self.assertEqual(status, 202)
            self.assertEqual(response["images"], ["images/second.jpg"])
            result = self.wait_for_job(response["id"])
        self.assertEqual(result["status"], "completed")
        self.assertFalse((self.workspace / "labels/second.txt").exists())
        self.assertTrue((self.workspace / ".autolabeler/buffer/images/second.txt").is_file())
        self.assertFalse((self.workspace / ".autolabeler/buffer/images/second.pending").exists())

    def test_training_parameter_recommendation_scales_with_count(self):
        small = recommend_training_parameters(10)
        large = recommend_training_parameters(10000)
        self.assertGreater(small["epochs"], large["epochs"])
        self.assertEqual(small["imgsz"], 640)
        self.assertGreaterEqual(large["batch"], small["batch"])

    def test_train_params_endpoint_counts_labelled_images(self):
        label = self.workspace / "labels" / "sample.txt"
        label.parent.mkdir()
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        status, data = self.request(
            "/api/train-params?image_dir=images&label_dir=labels&format=yolo"
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["count"], 1)
        self.assertIn("epochs", data["parameters"])
        self.assertIn("batch", data["parameters"])

    def test_augmentation_pipeline_injection(self):
        from ultralytics.data.augment import Compose, Format, LetterBox

        from autolabeler.augmentations import (
            RandomCrop,
            RandomOcclusion,
            apply_detection_augmentations,
            apply_pose_augmentations,
        )

        class FakeDataset:
            def __init__(self):
                self.transforms = Compose([LetterBox(new_shape=(640, 640)), Format()])

        pose = FakeDataset()
        apply_pose_augmentations(pose, crop=True, occlusion=True)
        names = [type(transform).__name__ for transform in pose.transforms.transforms]
        self.assertEqual(names, ["RandomCrop", "LetterBox", "RandomOcclusion", "Format"])

        detection = FakeDataset()
        apply_detection_augmentations(detection, crop=True, occlusion=True)
        names = [type(transform).__name__ for transform in detection.transforms.transforms]
        self.assertEqual(names[1], "LetterBox")
        self.assertEqual(names[2], "RandomOcclusion")
        self.assertEqual(names[3], "Format")

    def test_prediction_label_serialization_for_all_supported_formats(self):
        boxes = [
            SimpleNamespace(xyxy=[[10, 20, 30, 60]], cls=[1]),
        ]
        result = SimpleNamespace(
            orig_shape=(100, 200),
            boxes=boxes,
            obb=SimpleNamespace(
                xyxyxyxy=[[[10, 20], [30, 20], [30, 60], [10, 60]]],
                cls=[1],
            ),
            keypoints=SimpleNamespace(data=[[[20, 30, 2], [40, 50, 1]]]),
        )
        self.assertEqual(
            serialize_result(result, "yolo"),
            "1 0.100000 0.400000 0.100000 0.400000\n",
        )
        self.assertEqual(
            serialize_result(result, "yolo-obb"),
            "1 0.050000 0.200000 0.150000 0.200000 0.150000 0.600000 0.050000 0.600000\n",
        )
        self.assertEqual(
            serialize_result(result, "yolo-pose"),
            "1 0.100000 0.400000 0.100000 0.400000 0.100000 0.300000 2 0.200000 0.500000 1\n",
        )
        self.assertEqual(
            serialize_result(result, "yolo-pose", {0}),
            "1 0.100000 0.400000 0.100000 0.400000 0.100000 0.300000 2 0.000000 0.000000 0\n",
        )


    def test_pose_serialize_per_class_enabled_keypoints(self):
        boxes = [
            SimpleNamespace(xyxy=[[10, 20, 30, 60]], cls=[0]),
            SimpleNamespace(xyxy=[[50, 60, 70, 100]], cls=[1]),
        ]
        result = SimpleNamespace(
            orig_shape=(100, 200),
            boxes=boxes,
            keypoints=SimpleNamespace(
                data=[
                    [[20, 30, 2], [40, 50, 1]],
                    [[60, 70, 2], [80, 90, 2]],
                ]
            ),
        )
        self.assertEqual(
            serialize_result(result, "yolo-pose", {0: [0], 1: [1]}),
            "0 0.100000 0.400000 0.100000 0.400000 0.100000 0.300000 2 0.000000 0.000000 0\n"
            "1 0.300000 0.800000 0.100000 0.400000 0.000000 0.000000 0 0.400000 0.900000 2\n",
        )


if __name__ == "__main__":
    unittest.main()
