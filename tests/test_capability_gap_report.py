from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.capability_gap_report import build_capability_gap_report
from key_action_indexer.model_inventory import discover_lab_assets


def test_capability_gap_report_counts_required_label_foundations(tmp_path: Path) -> None:
    root = tmp_path
    dataset = root / "data" / "dataset"
    (root / "src" / "key_action_indexer").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (root / "configs" / "data").mkdir(parents=True)
    (root / "outputs").mkdir(parents=True)
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "images" / "train" / "frame_001.jpg").write_bytes(b"jpg")
    (dataset / "labels" / "train" / "frame_001.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n"
        "2 0.4 0.4 0.1 0.1\n"
        "1 0.3 0.3 0.1 0.1\n",
        encoding="utf-8",
    )
    (root / "configs" / "data" / "class_schema.yaml").write_text(
        "classes:\n"
        "  - id: 0\n"
        "    name: gloved_hand\n"
        "  - id: 1\n"
        "    name: pipette\n"
        "  - id: 2\n"
        "    name: button\n"
        "  - id: 3\n"
        "    name: display\n",
        encoding="utf-8",
    )
    (dataset / "dataset.yaml").write_text(
        f"path: {dataset.as_posix()}\n"
        "train: images/train\n"
        "names:\n"
        "  0: gloved_hand\n"
        "  1: pipette\n"
        "  2: button\n"
        "  3: display\n",
        encoding="utf-8",
    )

    inventory = discover_lab_assets(project_root=root)
    output = tmp_path / "capability_gap_report.json"
    report = build_capability_gap_report(
        project_root=root,
        model_inventory=inventory,
        dataset_root=dataset,
        class_schema_path=root / "configs" / "data" / "class_schema.yaml",
        output_path=output,
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["metadata_version"] == "key_action_capability_gap_report.v1"
    assert report["capabilities"]["button"]["has_label_foundation"] is True
    assert report["capabilities"]["button"]["sample_count"] == 1
    assert report["capabilities"]["display"]["has_class_definition"] is True
    assert report["capabilities"]["display"]["has_label_foundation"] is False
    assert report["capabilities"]["display"]["status"] == "schema_without_samples"
    assert report["capabilities"]["liquid"]["has_class_definition"] is False
    assert report["capabilities"]["liquid"]["has_label_foundation"] is False
    assert "liquid_region" in report["recommended_new_classes"]
    assert "container_open" in report["recommended_new_classes"]
    assert report["summary"]["total_yolo_label_instances"] == 3
    assert "liquid" in report["summary"]["capabilities_missing_label_foundation"]
