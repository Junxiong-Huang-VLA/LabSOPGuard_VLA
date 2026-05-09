from ultralytics import YOLO
import argparse
import cv2
import os
from pathlib import Path
from glob import glob
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="图像/视频/目录路径")
    parser.add_argument("--weights", default="models/runs/lab_detection_v125\\"
        ""
        ""
        "/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--save", action="store_true", help="保存结果图像")
    parser.add_argument("--fps", type=int, default=5, help="视频帧率")
    parser.add_argument("--video_output", default="output_video.mp4", help="保存视频的路径")
    parser.add_argument("--duration", type=int, default=60, help="视频时长（秒）")
    return parser.parse_args()

def generate_class_colors(names_dict):
    random.seed(42)
    colors = {}
    for _, class_name in names_dict.items():
        colors[class_name] = (
            random.randint(60, 255),
            random.randint(60, 255),
            random.randint(60, 255)
        )
    return colors

def create_video(images, output_path, fps, target_duration_sec, model, conf_thres=0.5):
    if len(images) == 0:
        raise ValueError("未找到任何图片!")

    repeat_each_frame = max(1, round(target_duration_sec * fps / len(images)))
    first_img = cv2.imread(images[0])
    h, w = first_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use "mp4v" codec for mp4 file
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    class_labels_displayed = set()  # Set to track classes that have already been displayed
    class_colors = generate_class_colors(model.names)

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue

        results = model.predict(source=img, conf=conf_thres, iou=0.45, verbose=False)
        vis = img.copy()

        # Visualize bounding boxes and class labels on the image
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = model.names[cls_id]
                conf = float(box.conf[0].cpu().numpy())

                # Draw bounding box with color for each class
                color = class_colors[cls_name]
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Display label and confidence score for the first box of each class
                if cls_name not in class_labels_displayed:
                    cv2.putText(vis, f"{cls_name} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    class_labels_displayed.add(cls_name)

        # Display the class names and confidence scores on the left side of the image
        y_offset = 25
        for cls_name, color in class_colors.items():
            if cls_name in class_labels_displayed:
                cv2.putText(vis, f"{cls_name}: {round(conf, 2)}", (12, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 30  # Move to the next line for the next class

        # Repeat the frame for the required number of times
        for _ in range(repeat_each_frame):
            writer.write(vis)

        if idx % 20 == 0:
            print(f"已处理 {idx}/{len(images)}")

    writer.release()
    print(f"视频已保存到 {output_path}")

def main():
    args = parse_args()
    
    # Collect all image paths
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(glob(os.path.join(args.source, ext)))
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f"未找到图片: {args.source}")

    print(f"找到 {len(image_paths)} 张图片.")
    
    model = YOLO(args.weights)
    
    # Create video
    create_video(image_paths, args.video_output, args.fps, args.duration, model, args.conf)

if __name__ == "__main__":
    main()