from __future__ import annotations

from pathlib import Path

from key_action_indexer.model_inventory import discover_lab_assets, resolve_best_model_path


def test_model_inventory_discovers_runtime_model_dataset_and_capabilities(tmp_path: Path) -> None:
    root = tmp_path
    (root / "src" / "key_action_indexer").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (root / "configs" / "model").mkdir(parents=True)
    (root / "configs" / "data").mkdir(parents=True)
    weights = root / "outputs" / "training" / "yolo26s_pose_lab_v4_focus_auto" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"fake")
    (root / "configs" / "model" / "detection_runtime.yaml").write_text(
        "model: outputs/training/yolo26s_pose_lab_v4_focus_auto/weights/best.pt\n"
        "model_fallbacks:\n"
        "  - yolo26s.pt\n",
        encoding="utf-8",
    )
    (root / "configs" / "data" / "class_schema.yaml").write_text(
        "classes:\n"
        "  - id: 0\n"
        "    name: gloved_hand\n"
        "  - id: 1\n"
        "    name: tube-cap\n"
        "  - id: 2\n"
        "    name: pipette\n",
        encoding="utf-8",
    )
    (root / "configs" / "data" / "pose_keypoints_schema.yaml").write_text(
        "default_keypoints: 3\n"
        "keypoint_names:\n"
        "  titration_tool: [tool_tip, tool_body_center, tool_grasp_point]\n",
        encoding="utf-8",
    )
    dataset = root / "data" / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "images" / "train" / "frame.jpg").write_bytes(b"jpg")
    (dataset / "labels" / "train" / "frame.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (dataset / "dataset.yaml").write_text(
        f"path: {dataset.as_posix()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: gloved_hand\n"
        "  1: tube-cap\n"
        "  2: pipette\n",
        encoding="utf-8",
    )

    inventory = discover_lab_assets(root)

    assert inventory["primary_model"]["path"].endswith("best.pt")
    assert inventory["dataset_count"] == 1
    assert inventory["datasets"][0]["total_image_count"] == 2
    assert inventory["datasets"][0]["total_label_count"] == 2
    assert inventory["capabilities"]["cap_lid_detection"]["available"] is True
    assert inventory["capabilities"]["pipette_tool_detection"]["available"] is True
    assert resolve_best_model_path(project_root=root) == weights / "best.pt"
