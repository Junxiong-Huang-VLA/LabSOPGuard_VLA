"""
将 D:/cvdemo/lab_detection/data/unified_dataset 的标注数据
重映射类别 ID 后合并到 LabSOPGuard 主数据集。

cvdemo class → LabSOPGuard class:
  0(tube) → 9, 1(tube-cap) → 10, 2(beaker) → 1, 3(spearhead) → 11, 4(pipette) → 12
"""

import shutil
from pathlib import Path

SRC = Path("D:/cvdemo/lab_detection/data/unified_dataset")
DST = Path("D:/LabEmbodiedVLA/LabSOPGuard/data/dataset")

CLASS_MAP = {0: 9, 1: 10, 2: 1, 3: 11, 4: 12}


def remap_label(src_path: Path, dst_path: Path):
    lines = src_path.read_text(encoding="utf-8").strip().splitlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        old_cls = int(parts[0])
        if old_cls not in CLASS_MAP:
            continue
        parts[0] = str(CLASS_MAP[old_cls])
        new_lines.append(" ".join(parts))
    dst_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main():
    stats = {v: 0 for v in CLASS_MAP.values()}
    copied = 0
    skipped = 0

    for split in ["train", "val"]:
        src_img_dir = SRC / "images" / split
        src_lbl_dir = SRC / "labels" / split
        dst_img_dir = DST / "images" / split
        dst_lbl_dir = DST / "labels" / split

        if not src_img_dir.exists():
            print(f"[skip] {src_img_dir} not found")
            continue

        for img_path in sorted(src_img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            dst_img = dst_img_dir / f"cvdemo_{img_path.name}"
            dst_lbl = dst_lbl_dir / f"cvdemo_{lbl_path.name}"

            if dst_img.exists():
                skipped += 1
                continue

            shutil.copy2(img_path, dst_img)
            remap_label(lbl_path, dst_lbl)
            copied += 1

            for line in dst_lbl.read_text(encoding="utf-8").strip().splitlines():
                cls = int(line.split()[0])
                stats[cls] = stats.get(cls, 0) + 1

    name_map = {1: "beaker", 9: "tube", 10: "tube-cap", 11: "spearhead", 12: "pipette"}
    print(f"\n合并完成: {copied} 张图片, {skipped} 张跳过(已存在)")
    print("\n新增标注数量:")
    for cls, count in sorted(stats.items()):
        print(f"  class {cls:2d} ({name_map.get(cls, '?'):12s}): {count}")


if __name__ == "__main__":
    main()
