"""Resplit dataset into train/val/test with stratification by class presence."""
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 42
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
NUM_CLASSES = 13
DATASET_ROOT = Path(os.environ.get("LAB_DATASET_ROOT", r"D:\LabDatasets\LabEmbodied\dataset"))

def get_classes_in_label(label_path: Path) -> set[int]:
    classes = set()
    for line in label_path.read_text().strip().splitlines():
        cls_id = int(line.split()[0])
        classes.add(cls_id)
    return classes

def main():
    img_dir = DATASET_ROOT / "images"
    lbl_dir = DATASET_ROOT / "labels"

    all_pairs = []
    for split in ["train", "val", "test"]:
        split_img = img_dir / split
        split_lbl = lbl_dir / split
        if not split_img.exists():
            continue
        for img_file in sorted(split_img.iterdir()):
            stem = img_file.stem
            lbl_file = split_lbl / f"{stem}.txt"
            if lbl_file.exists():
                all_pairs.append((img_file, lbl_file))

    print(f"Total image-label pairs: {len(all_pairs)}")

    class_to_indices = defaultdict(list)
    for i, (_, lbl) in enumerate(all_pairs):
        for cls_id in get_classes_in_label(lbl):
            class_to_indices[cls_id].append(i)

    print("\nClass distribution before split:")
    for cls_id in range(NUM_CLASSES):
        print(f"  class {cls_id:2d}: {len(class_to_indices[cls_id])} images")

    random.seed(SEED)

    assigned = {}  # index -> split
    for cls_id in range(NUM_CLASSES):
        indices = class_to_indices[cls_id]
        unassigned = [i for i in indices if i not in assigned]
        if not unassigned:
            continue
        random.shuffle(unassigned)
        n = len(unassigned)
        n_val = max(1, round(n * RATIOS["val"]))
        n_test = max(1, round(n * RATIOS["test"]))
        n_train = n - n_val - n_test
        for i in unassigned[:n_train]:
            if i not in assigned:
                assigned[i] = "train"
        for i in unassigned[n_train:n_train + n_val]:
            if i not in assigned:
                assigned[i] = "val"
        for i in unassigned[n_train + n_val:]:
            if i not in assigned:
                assigned[i] = "test"

    for i in range(len(all_pairs)):
        if i not in assigned:
            assigned[i] = "train"

    split_counts = defaultdict(int)
    for s in assigned.values():
        split_counts[s] += 1
    print(f"\nSplit counts: {dict(split_counts)}")

    split_class_counts = {s: defaultdict(int) for s in ["train", "val", "test"]}
    for i, (_, lbl) in enumerate(all_pairs):
        s = assigned[i]
        for cls_id in get_classes_in_label(lbl):
            split_class_counts[s][cls_id] += 1

    print("\nPer-class per-split image counts:")
    header = f"{'class':>20s}  {'train':>6s}  {'val':>6s}  {'test':>6s}"
    print(header)
    for cls_id in range(NUM_CLASSES):
        tr = split_class_counts["train"].get(cls_id, 0)
        va = split_class_counts["val"].get(cls_id, 0)
        te = split_class_counts["test"].get(cls_id, 0)
        print(f"  {cls_id:>18d}  {tr:>6d}  {va:>6d}  {te:>6d}")

    confirm = input("\nProceed with resplit? [y/N] ")
    if confirm.lower() != "y":
        print("Aborted.")
        return

    tmp_dir = DATASET_ROOT / "_tmp_resplit"
    for split in ["train", "val", "test"]:
        (tmp_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for i, (img_file, lbl_file) in enumerate(all_pairs):
        split = assigned[i]
        shutil.copy2(img_file, tmp_dir / "images" / split / img_file.name)
        shutil.copy2(lbl_file, tmp_dir / "labels" / split / lbl_file.name)

    for split in ["train", "val", "test"]:
        old_img = img_dir / split
        old_lbl = lbl_dir / split
        if old_img.exists():
            shutil.rmtree(old_img)
        if old_lbl.exists():
            shutil.rmtree(old_lbl)

    for split in ["train", "val", "test"]:
        shutil.move(str(tmp_dir / "images" / split), str(img_dir / split))
        shutil.move(str(tmp_dir / "labels" / split), str(lbl_dir / split))

    shutil.rmtree(tmp_dir)

    report = {
        "seed": SEED,
        "ratios": RATIOS,
        "total_pairs": len(all_pairs),
        "split_counts": dict(split_counts),
    }
    (DATASET_ROOT / "split_report.json").write_text(json.dumps(report, indent=2))

    print("\nDone. Dataset resplit complete.")

if __name__ == "__main__":
    main()
