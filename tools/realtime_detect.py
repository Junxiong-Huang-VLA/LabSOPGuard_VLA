"""
RealSense D435i 实时目标检测
结合深度信息输出每个检测目标的距离
用法：python scripts/realtime_detect.py --weights models/runs/lab_detection_v1/weights/best.pt
"""

import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


# 每个类别显示颜色(BGR)
CLASS_COLORS = {
    "beaker": (0, 255, 100),
    "pipette": (255, 160, 0),
    "spearhead": (180, 0, 255),
    "tube": (0, 180, 255),
    "tube-cap": (255, 100, 100),
}

# Tube 误检抑制配置
TUBE_LABELS = {"tube", "test_tube"}
TUBE_CAP_LABELS = {"tube-cap", "tube_cap", "test_tube_cap"}
TUBE_MIN_CONF = 0.60
TUBE_CAP_MIN_CONF = 0.62
TUBE_MIN_AREA = 800.0
TUBE_CAP_MIN_AREA = 180.0
TUBE_MIN_ASPECT = 1.65  # h / w
TUBE_MAX_ASPECT = 8.50
TUBE_MIN_HEIGHT_PX = 35.0
TUBE_MAX_WIDTH_PX = 120.0
TUBE_CAP_MAX_AREA = 3200.0
TUBE_CAP_MIN_ASPECT = 0.45
TUBE_CAP_MAX_ASPECT = 1.75
TUBE_CAP_MIN_SIDE_PX = 14.0
TUBE_DEPTH_RANGE_M = (0.10, 2.50)
TUBE_CAP_DEPTH_RANGE_M = (0.10, 2.50)
DEDUP_IOU_THRES = 0.40
DEDUP_CENTER_DIST_PX = 26.0
DEDUP_DEPTH_DIST_M = 0.07

CLASS_MIN_CONF = {
    "beaker": 0.46,
    "pipette": 0.70,
    "spearhead": 0.68,
    "tube": TUBE_MIN_CONF,
    "test-tube": TUBE_MIN_CONF,
    "tube-cap": TUBE_CAP_MIN_CONF,
    "test-tube-cap": TUBE_CAP_MIN_CONF,
}
DEFAULT_MIN_CONF = 0.50
BEAKER_MIN_AREA = 700.0
BEAKER_ASPECT_RANGE = (0.45, 2.60)
PIPETTE_MIN_AREA = 380.0
PIPETTE_MAX_AREA = 22000.0
PIPETTE_MIN_ELONGATION = 2.30
PIPETTE_DEPTH_RANGE_M = (0.10, 1.30)
SPEARHEAD_MIN_AREA = 90.0
SPEARHEAD_MAX_AREA = 5200.0
SPEARHEAD_ASPECT_RANGE = (0.35, 2.80)
MIN_CONFIRM_FRAMES = 2
MAX_TRACK_MISS = 6
TRACK_CENTER_DIST_PX = 42.0
TRACK_DEPTH_DIST_M = 0.12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.set_defaults(all_classes=True)
    parser.add_argument("--all-classes", dest="all_classes", action="store_true")
    parser.add_argument("--tube-only", dest="all_classes", action="store_false")
    return parser.parse_args()


def get_median_depth(depth_frame, x1, y1, x2, y2, margin=10):
    """获取 BBox 中心区域的中位数深度（更稳定）"""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    m = margin
    region = np.array(
        [
            [
                depth_frame.get_distance(max(cx - m, 0), max(cy - m, 0)),
                depth_frame.get_distance(min(cx + m, depth_frame.width - 1), max(cy - m, 0)),
            ],
            [
                depth_frame.get_distance(max(cx - m, 0), min(cy + m, depth_frame.height - 1)),
                depth_frame.get_distance(min(cx + m, depth_frame.width - 1), min(cy + m, depth_frame.height - 1)),
            ],
        ]
    )
    valid = region[region > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def normalize_label(name):
    return str(name).strip().lower().replace("_", "-")


def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def suppress_near_duplicates(box_infos, iou_thres=0.40, center_dist_px=26.0, depth_dist_m=0.07):
    if len(box_infos) <= 1:
        return box_infos

    keep = []
    for cand in sorted(box_infos, key=lambda b: b["conf"], reverse=True):
        duplicate = False
        for kept in keep:
            iou = compute_iou(cand, kept)
            center_dist = float(np.hypot(cand["cx"] - kept["cx"], cand["cy"] - kept["cy"]))
            depth_dist = abs(float(cand["depth_m"]) - float(kept["depth_m"]))
            if iou >= iou_thres or (center_dist <= center_dist_px and depth_dist <= depth_dist_m):
                duplicate = True
                break
        if not duplicate:
            keep.append(cand)
    return keep


def filter_temporally_confirmed(detections, tracks):
    for t in tracks:
        t["matched"] = False

    for det in sorted(detections, key=lambda d: d["conf"], reverse=True):
        best_idx = -1
        best_score = 1e18
        for idx, t in enumerate(tracks):
            if t["label"] != det["cls_name"]:
                continue
            center_dist = float(np.hypot(det["cx"] - t["cx"], det["cy"] - t["cy"]))
            depth_dist = abs(float(det["depth_m"]) - float(t["depth_m"]))
            iou = compute_iou(det, t)
            if iou < 0.15 and (center_dist > TRACK_CENTER_DIST_PX or depth_dist > TRACK_DEPTH_DIST_M):
                continue
            score = center_dist + 100.0 * depth_dist
            if score < best_score:
                best_idx = idx
                best_score = score

        if best_idx >= 0:
            t = tracks[best_idx]
            t["cx"] = det["cx"]
            t["cy"] = det["cy"]
            t["depth_m"] = det["depth_m"]
            t["x1"] = det["x1"]
            t["y1"] = det["y1"]
            t["x2"] = det["x2"]
            t["y2"] = det["y2"]
            t["det"] = det
            t["miss"] = 0
            t["hits"] = min(t["hits"] + 1, 1000)
            t["matched"] = True
        else:
            tracks.append(
                {
                    "label": det["cls_name"],
                    "cx": det["cx"],
                    "cy": det["cy"],
                    "depth_m": det["depth_m"],
                    "x1": det["x1"],
                    "y1": det["y1"],
                    "x2": det["x2"],
                    "y2": det["y2"],
                    "det": det,
                    "hits": 1,
                    "miss": 0,
                    "matched": True,
                }
            )

    kept_tracks = []
    confirmed_dets = []
    for t in tracks:
        if not t["matched"]:
            t["miss"] += 1
        if t["miss"] <= MAX_TRACK_MISS:
            kept_tracks.append(t)
        if t["matched"] and t["hits"] >= MIN_CONFIRM_FRAMES:
            confirmed_dets.append(t["det"])

    return confirmed_dets, kept_tracks


def should_keep_detection(cls_name, conf, x1, y1, x2, y2, depth_m):
    norm_label = normalize_label(cls_name)
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    area = w * h
    aspect = h / w
    elongation = max(w, h) / max(1.0, min(w, h))
    min_conf = CLASS_MIN_CONF.get(norm_label, DEFAULT_MIN_CONF)
    if conf < min_conf:
        return False

    if norm_label in {"tube", "test-tube"}:
        if conf < TUBE_MIN_CONF:
            return False
        if area < TUBE_MIN_AREA:
            return False
        if aspect < TUBE_MIN_ASPECT:
            return False
        if aspect > TUBE_MAX_ASPECT:
            return False
        if h < TUBE_MIN_HEIGHT_PX:
            return False
        if w > TUBE_MAX_WIDTH_PX:
            return False
        if depth_m > 0 and not (TUBE_DEPTH_RANGE_M[0] <= depth_m <= TUBE_DEPTH_RANGE_M[1]):
            return False
        return True

    if norm_label in {"tube-cap", "test-tube-cap"}:
        if conf < TUBE_CAP_MIN_CONF:
            return False
        if area < TUBE_CAP_MIN_AREA:
            return False
        if area > TUBE_CAP_MAX_AREA:
            return False
        if aspect < TUBE_CAP_MIN_ASPECT or aspect > TUBE_CAP_MAX_ASPECT:
            return False
        if min(w, h) < TUBE_CAP_MIN_SIDE_PX:
            return False
        if depth_m > 0 and not (TUBE_CAP_DEPTH_RANGE_M[0] <= depth_m <= TUBE_CAP_DEPTH_RANGE_M[1]):
            return False
        return True

    if norm_label == "beaker":
        if area < BEAKER_MIN_AREA:
            return False
        if not (BEAKER_ASPECT_RANGE[0] <= aspect <= BEAKER_ASPECT_RANGE[1]):
            return False
        return True

    if norm_label == "pipette":
        if area < PIPETTE_MIN_AREA or area > PIPETTE_MAX_AREA:
            return False
        if elongation < PIPETTE_MIN_ELONGATION:
            return False
        if depth_m > 0 and not (PIPETTE_DEPTH_RANGE_M[0] <= depth_m <= PIPETTE_DEPTH_RANGE_M[1]):
            return False
        return True

    if norm_label == "spearhead":
        if area < SPEARHEAD_MIN_AREA or area > SPEARHEAD_MAX_AREA:
            return False
        if not (SPEARHEAD_ASPECT_RANGE[0] <= aspect <= SPEARHEAD_ASPECT_RANGE[1]):
            return False
        return True

    return True


def draw_detection(image, xyxy, cls_name, conf, depth_m):
    """绘制检测结果"""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    color = CLASS_COLORS.get(normalize_label(cls_name), (0, 255, 255))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    label = f"{cls_name} {conf:.2f}"
    if depth_m > 0:
        label += f" {depth_m:.2f}m"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(image, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    args = parse_args()

    model = YOLO(args.weights)
    target_labels = {"tube", "test-tube", "tube-cap", "test-tube-cap"}
    print(f"[INFO] 模型加载完成: {args.weights}")
    print(f"[INFO] 检测类别: {model.names}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)

    print("\n[INFO] 实时检测已启动，按 Q 退出\n")

    import time

    fps_counter = 0
    fps_start = time.time()
    fps = 0
    temporal_tracks = []

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            results = model.predict(
                color_image,
                conf=args.conf,
                verbose=False,
            )

            output = color_image.copy()
            detections = []
            grouped = {}

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    if (not args.all_classes) and (normalize_label(cls_name) not in target_labels):
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    depth_m = 0.0
                    if depth_frame:
                        sx = 640 / args.width
                        sy = 480 / args.height
                        depth_m = get_median_depth(
                            depth_frame,
                            int(x1 * sx),
                            int(y1 * sy),
                            int(x2 * sx),
                            int(y2 * sy),
                        )

                    if not should_keep_detection(cls_name, conf, x1, y1, x2, y2, depth_m):
                        continue

                    info = {
                        "cls_name": cls_name,
                        "conf": conf,
                        "depth_m": depth_m,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": int((x1 + x2) / 2),
                        "cy": int((y1 + y2) / 2),
                        "xyxy": [x1, y1, x2, y2],
                    }
                    grouped.setdefault(cls_name, []).append(info)

            final_detections = []
            for cls_name, box_infos in grouped.items():
                kept = suppress_near_duplicates(
                    box_infos,
                    iou_thres=DEDUP_IOU_THRES,
                    center_dist_px=DEDUP_CENTER_DIST_PX,
                    depth_dist_m=DEDUP_DEPTH_DIST_M,
                )
                final_detections.extend(kept)

            confirmed_detections, temporal_tracks = filter_temporally_confirmed(final_detections, temporal_tracks)

            for det in sorted(confirmed_detections, key=lambda d: d["conf"], reverse=True):
                draw_detection(output, det["xyxy"], det["cls_name"], det["conf"], det["depth_m"])
                detections.append((det["cls_name"], det["conf"], det["depth_m"]))

            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()

            cv2.putText(output, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output, f"Det: {len(detections)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("化学实验室目标检测 - RealSense D435i", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 程序已退出")


if __name__ == "__main__":
    main()
