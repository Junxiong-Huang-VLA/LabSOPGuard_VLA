"""
对测试集跑推理，为每个类别挑出置信度最高的一张，裁剪出检测框区域并保存带框的完整图。
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL = "outputs/training/yolo26s_autodl_8_1_1/weights/best.pt"
IMG_DIR = "data/dataset/images/test"
OUT_DIR = "outputs/val/best_per_class"
CONF_THRESH = 0.10  # 低阈值确保每类都能抓到

CLASS_NAMES = {
    0: "balance", 1: "beaker", 2: "gloved_hand", 3: "lab_coat", 4: "paper",
    5: "reagent_bottle", 6: "sample_bottle", 7: "sample_bottle_blue",
    8: "spatula",
}

COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52),
]

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL)
img_paths = list(Path(IMG_DIR).glob("*.jpg")) + list(Path(IMG_DIR).glob("*.png"))

# best_det[cls] = (conf, img_path, box_xyxy, all_results_for_img)
best_det = {}

print(f"Running inference on {len(img_paths)} images ...")
results = model.predict(img_paths, conf=CONF_THRESH, iou=0.45, imgsz=640,
                        verbose=False, stream=True)

for r in results:
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        continue
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls not in CLASS_NAMES:
            continue
        if cls not in best_det or conf > best_det[cls][0]:
            best_det[cls] = (conf, r.orig_img.copy(), box.xyxy[0].cpu().numpy(), r)

print(f"Found best detections for classes: {[CLASS_NAMES[c] for c in sorted(best_det)]}")

for cls, (conf, img, xyxy, r) in sorted(best_det.items()):
    name = CLASS_NAMES[cls]
    color = COLORS[cls % len(COLORS)]

    # 画所有框（同图中所有类）
    vis = img.copy()
    for box in r.boxes:
        c = int(box.cls[0])
        cf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        bc = COLORS[c % len(COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), bc, 2)
        label = f"{CLASS_NAMES.get(c, str(c))} {cf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), bc, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 高亮目标框（加粗）
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 4)
    title = f"BEST: {name}  conf={conf:.3f}"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(vis, title, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out_path = os.path.join(OUT_DIR, f"{cls:02d}_{name}_conf{conf:.3f}.jpg")
    cv2.imwrite(out_path, vis)
    print(f"  [{cls}] {name:20s}  conf={conf:.3f}  -> {out_path}")

print(f"\nDone. Saved {len(best_det)} images to {OUT_DIR}/")
