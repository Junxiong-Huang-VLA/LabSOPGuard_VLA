"""
可视化自动标注结果
在图像上绘制标注框，方便检查标注质量

用法：python scripts/visualize_labels.py
"""

import os
import cv2
import random
from pathlib import Path

# ============ 配置 ============
IMAGE_DIR = r"D:\cvdemo\lab_detection\data\raw\color"
LABEL_DIR = r"D:\cvdemo\lab_detection\data\raw\labels"
OUTPUT_DIR = r"D:\cvdemo\lab_detection\data\raw\visualized"
# 只可视化前N张（设为0则全部可视化）
MAX_IMAGES = 20
# ==============================

CLASS_NAMES = {
    0: 'beaker',
    1: 'pipette',
    2: 'spearhead',
    3: 'tube',
    4: 'tube-cap'
}

# 生成随机颜色
random.seed(42)
CLASS_COLORS = {
    cls_id: (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255))
    for cls_id in CLASS_NAMES.keys()
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = sorted(Path(IMAGE_DIR).glob("*.jpg"))
if MAX_IMAGES > 0:
    image_files = image_files[:MAX_IMAGES]

print(f"可视化 {len(image_files)} 张图像")

for img_path in image_files:
    label_path = Path(LABEL_DIR) / (img_path.stem + ".txt")

    if not label_path.exists():
        print(f"跳过（无标注）: {img_path.name}")
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # 读取标注
    with open(label_path, "r") as f:
        lines = f.readlines()

    obj_count = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        box_w = float(parts[3]) * w
        box_h = float(parts[4]) * h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))

        # 绘制框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = cls_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        obj_count += 1

    # 左上角显示统计
    cv2.putText(img, f"Objects: {obj_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 保存
    output_path = Path(OUTPUT_DIR) / img_path.name
    cv2.imwrite(str(output_path), img)

    print(f"OK {img_path.name} ({obj_count} objects)")

print(f"\n完成！可视化结果保存在: {OUTPUT_DIR}")
