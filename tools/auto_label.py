"""
使用已训练模型对 raw/color 目录下的图像进行自动标注
生成 YOLO 格式的 txt 标注文件

用法：python scripts/auto_label.py
"""

import os
from pathlib import Path
from ultralytics import YOLO

# ============ 配置 ============
WEIGHTS = r"D:\cvdemo\runs\detect\models\runs\lab_detection_v125\weights\best.pt"
IMAGE_DIR = r"D:\cvdemo\lab_detection\data\raw\color"
LABEL_DIR = r"D:\cvdemo\lab_detection\data\raw\labels"
CONF_THRESH = 0.5
IOU_THRESH = 0.45
# ==============================

os.makedirs(LABEL_DIR, exist_ok=True)

print(f"加载模型: {WEIGHTS}")
model = YOLO(WEIGHTS)
print(f"模型类别: {model.names}")

# 获取第一张图片的尺寸作为图像大小
sample_img = next(Path(IMAGE_DIR).glob("*.jpg"))
import cv2
sample = cv2.imread(str(sample_img))
img_h, img_w = sample.shape[:2]
print(f"图像尺寸: {img_w}x{img_h}")

image_files = sorted(Path(IMAGE_DIR).glob("*.jpg"))
print(f"找到 {len(image_files)} 张图像")

total_objects = 0
labeled_count = 0

for img_path in image_files:
    results = model.predict(
        source=str(img_path),
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False,
    )

    label_path = Path(LABEL_DIR) / (img_path.stem + ".txt")

    with open(label_path, "w") as f:
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                # xyxy -> xywhn (归一化到[0,1])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                conf = float(box.conf[0].item())

                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_objects += 1

    labeled_count += 1
    if labeled_count % 20 == 0 or labeled_count == len(image_files):
        print(f"进度: {labeled_count}/{len(image_files)}")

print(f"\n完成！共处理 {labeled_count} 张图像，标注了 {total_objects} 个目标")
print(f"标注文件保存在: {LABEL_DIR}")
