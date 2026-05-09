"""
简单的图像标注工具
支持加载现有标注、修改、添加、删除标注框

用法：python scripts/simple_annotator.py

快捷键：
- 鼠标左键拖动：绘制新框
- 鼠标右键点击框：删除框
- 数字键 0-4：选择类别
- S：保存当前标注
- D：下一张图
- A：上一张图
- Q/ESC：退出
- R：重置当前图像标注
- H：显示/隐藏帮助
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# ============ 配置 ============
IMAGE_DIR = r"D:\cvdemo\lab_detection\data\raw\color"
LABEL_DIR = r"D:\cvdemo\lab_detection\data\raw\labels"
# ==============================

CLASS_NAMES = ['tube', 'tube-cap', 'beaker', 'spearhead', 'pipette']
CLASS_COLORS = [
    (255, 255, 100),  # tube - 黄色
    (255, 100, 255),  # tube-cap - 粉红色
    (255, 100, 100),  # beaker - 红色
    (100, 100, 255),  # spearhead - 蓝色
    (100, 255, 100),  # pipette - 绿色
]

class Annotator:
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_dir.mkdir(exist_ok=True)

        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        if not self.image_files:
            print("错误：未找到图像文件")
            sys.exit(1)

        self.current_idx = 0
        self.current_class = 0  # 默认选择 tube
        self.boxes = []  # [(cls_id, x1, y1, x2, y2), ...]
        self.drawing = False
        self.start_point = None
        self.temp_box = None
        self.show_help = True
        self.modified = False

        self.load_image()

    def load_image(self):
        """加载当前图像和标注"""
        img_path = self.image_files[self.current_idx]
        self.img = cv2.imread(str(img_path))
        self.img_display = self.img.copy()
        self.h, self.w = self.img.shape[:2]

        # 加载标注
        label_path = self.label_dir / (img_path.stem + ".txt")
        self.boxes = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1]) * self.w
                        y_center = float(parts[2]) * self.h
                        box_w = float(parts[3]) * self.w
                        box_h = float(parts[4]) * self.h

                        x1 = int(x_center - box_w / 2)
                        y1 = int(y_center - box_h / 2)
                        x2 = int(x_center + box_w / 2)
                        y2 = int(y_center + box_h / 2)

                        self.boxes.append((cls_id, x1, y1, x2, y2))

        self.modified = False
        print(f"\n[{self.current_idx + 1}/{len(self.image_files)}] {img_path.name} - {len(self.boxes)} objects")

    def save_labels(self):
        """保存当前标注"""
        img_path = self.image_files[self.current_idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        with open(label_path, "w") as f:
            for cls_id, x1, y1, x2, y2 in self.boxes:
                x_center = ((x1 + x2) / 2) / self.w
                y_center = ((y1 + y2) / 2) / self.h
                width = (x2 - x1) / self.w
                height = (y2 - y1) / self.h
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"已保存: {label_path.name}")
        self.modified = False

    def draw_boxes(self):
        """绘制所有标注框"""
        self.img_display = self.img.copy()

        # 绘制已有的框
        for cls_id, x1, y1, x2, y2 in self.boxes:
            color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (255, 255, 255)
            cv2.rectangle(self.img_display, (x1, y1), (x2, y2), color, 2)

            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(self.img_display, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(self.img_display, label, (x1 + 2, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制正在绘制的框
        if self.temp_box:
            x1, y1, x2, y2 = self.temp_box
            color = CLASS_COLORS[self.current_class]
            cv2.rectangle(self.img_display, (x1, y1), (x2, y2), color, 2)

        # 显示当前类别
        class_text = f"Class: [{self.current_class}] {CLASS_NAMES[self.current_class]}"
        cv2.putText(self.img_display, class_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示图像信息
        info_text = f"Image: {self.current_idx + 1}/{len(self.image_files)} | Objects: {len(self.boxes)}"
        if self.modified:
            info_text += " [Modified]"
        cv2.putText(self.img_display, info_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示帮助
        if self.show_help:
            help_lines = [
                "=== Help (H to hide) ===",
                "Left Click+Drag: Draw box",
                "Right Click: Delete box",
                "0-4: Select class",
                "S: Save | D: Next | A: Prev",
                "R: Reset | Q/ESC: Quit"
            ]
            y_offset = 90
            for line in help_lines:
                cv2.putText(self.img_display, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件处理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                x1, y1 = self.start_point
                self.temp_box = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
                self.draw_boxes()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y

                # 确保框有效（宽高至0像素）
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
              box = (self.current_class, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    self.boxes.append(box)
                    self.modified = True
                    print(f"添加框: {CLASS_NAMES[self.current_class]}")

                self.drawing = False
                self.start_point = None
                self.temp_box = None
                self.draw_boxes()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键删除框
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    removed = self.boxes.pop(i)
                    self.modified = True
                    print(f"删除框: {CLASS_NAMES[removed[0]]}")
                    self.draw_boxes()
                    break

    def run(self):
        """主循环"""
        window_name = "Simple Annotator"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        self.draw_boxes()

        while True:
            cv2.imshow(window_name, self.img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                if self.modified:
                    print("警告：当前图像有未保存的修改")
                    print("按 S 保存，或再次按 Q 退出")
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord('s'):
                        self.save_labels()
                        continue
                    elif key2 == ord('q') or key2 == 27:
                        break
                else:
                    break

            elif key == ord('s'):  # Save
                self.save_labels()

            elif key == ord('d'):  # Next
                if self.modified:
                    print("自动保存...")
                    self.save_labels()
                if self.current_idx < len(self.image_files) - 1:
                    self.current_idx += 1
                    self.load_image()
                    self.draw_boxes()
                else:
                    print("已经是最后一张")

            elif key == ord('a'):  # Previous
                if self.modified:
                    print("自动保存...")
                    self.save_labels()
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.load_image()
                    self.draw_boxes()
                else:
                    print("已经是第一张")

            elif key == ord('r'):  # Reset
                self.boxes = []
                self.modified = True
                print("已清空当前标注")
                self.draw_boxes()

            elif key == ord('h'):  # Toggle help
                self.show_help = not self.show_help
                self.draw_boxes()

            elif ord('0') <= key <= ord('4'):  # Select class
                self.current_class = key - ord('0')
                print(f"选择类别: [{self.current_class}] {CLASS_NAMES[self.current_class]}")
                self.draw_boxes()

        cv2.destroyAllWindows()
        print("\n标注工具已退出")


if __name__ == "__main__":
    print("=" * 50)
    print("简单标注工具")
    print("=" * 50)
    print(f"图像目录: {IMAGE_DIR}")
    print(f"标注目录: {LABEL_DIR}")
    print("\n类别列表:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  [{i}] {name}")
    print("\n
    annotator = Annotator(IMAGE_DIR, LABEL_DIR)
    annotator.run()
