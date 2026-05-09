import os
import csv
import cv2
import random
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

# =========================
# 浠ｇ悊璁剧疆
# =========================
os.environ["YOLO_OFFLINE"] = "1"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# =========================
# 閰嶇疆鍖?
# =========================
weights_path = r"D:\cvdemo\runs\detect\models\runs\lab_detection_v123\weights\best.pt"
save_dir = r"D:\cvdemo\output"

video_output_path = os.path.join(save_dir, "realsense_show_all_and_export_all_tube_tubecap.mp4")
csv_output_path = os.path.join(save_dir, "realsense_show_all_and_export_all_tube_tubecap.csv")

conf_thres = 0.30
iou_thres = 0.45
max_det = 30

camera_width = 640
camera_height = 480
camera_fps = 30
video_save_fps = 10

# 姣忕被鏈€浣崇洰鏍囩瓥鐣?
best_strategy = "conf"   # "conf" / "area" / "center"

# 娣卞害绐楀彛
depth_win = 2

# 宸︿笂瑙掍俊鎭?
show_top_left_info = True

# 鍏ㄩ儴鏄剧ず銆佸叏閮ㄥ鍑虹殑绫诲埆
SHOW_AND_EXPORT_ALL_CLASSES = ["beaker", "pipette", "spearhead", "tube", "test-tube", "tube-cap", "test-tube-cap"]
ONLY_PROCESS_TARGET_CLASSES = False

# 鏄惁缁欐墍鏈夋樉绀烘鐢讳腑蹇冪偣
show_center_for_all_boxes = True

# Tube 璇鎶戝埗锛氬 tube / tube-cap 鍋氭洿涓ユ牸浜屾杩囨护
TUBE_LABELS = {"tube", "test_tube"}
TUBE_CAP_LABELS = {"tube-cap", "tube_cap", "test_tube_cap"}
tube_min_conf = 0.60
tube_cap_min_conf = 0.62
tube_min_area = 1300.0
tube_cap_min_area = 300.0
tube_min_aspect = 1.65

tube_max_aspect = 8.50
tube_min_height_px = 45.0
tube_max_width_px = 82.0
tube_cap_max_area = 3200.0
tube_cap_min_aspect = 0.45
tube_cap_max_aspect = 1.75
tube_cap_min_side_px = 14.0
tube_depth_range_m = (0.14, 0.95)
tube_cap_depth_range_m = (0.10, 0.95)
dedup_iou_thres = 0.40
dedup_center_dist_px = 26.0
dedup_depth_dist_m = 0.07

CLASS_MIN_CONF = {
    "beaker": 0.46,
    "pipette": 0.70,
    "spearhead": 0.68,
    "tube": tube_min_conf,
    "test-tube": tube_min_conf,
    "tube-cap": tube_cap_min_conf,
    "test-tube-cap": tube_cap_min_conf,
}
default_min_conf = 0.50
beaker_min_area = 750.0
beaker_aspect_range = (0.45, 2.60)
pipette_min_area = 420.0
pipette_max_area = 18000.0
pipette_min_elongation = 2.30
pipette_depth_range_m = (0.10, 1.20)
spearhead_min_area = 100.0
spearhead_max_area = 5200.0
spearhead_aspect_range = (0.35, 2.80)
min_confirm_frames = 2

# =========================
# 鎵嬬溂鏍囧畾鍙傛暟
# =========================
T_robot_from_cam = np.array([ -0.736315, -0.0478544, 1.14669], dtype=np.float64)
q_x, q_y, q_z, q_w = 0.999387, -0.0147962, -0.00924636, -0.0303465

def quat_to_rot_matrix(qx, qy, qz, qw):
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)

R_robot_from_cam = quat_to_rot_matrix(q_x, q_y, q_z, q_w)

def cam_to_robot(X_cam, Y_cam, Z_cam):
    p_cam = np.array([X_cam, Y_cam, Z_cam], dtype=np.float64)
    p_robot = R_robot_from_cam @ p_cam + T_robot_from_cam
    return float(p_robot[0]), float(p_robot[1]), float(p_robot[2])

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

def choose_best_box(box_infos, width, height, strategy="conf"):
    if len(box_infos) == 1:
        return box_infos[0]

    if strategy == "conf":
        return max(box_infos, key=lambda b: b["conf"])
    if strategy == "area":
        return max(box_infos, key=lambda b: b["area"])
    if strategy == "center":
        cx0, cy0 = width / 2, height / 2
        return min(box_infos, key=lambda b: (b["cx"] - cx0) ** 2 + (b["cy"] - cy0) ** 2)

    return box_infos[0]

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
            depth_dist = abs(float(cand["depth"]) - float(kept["depth"]))
            if iou >= iou_thres or (center_dist <= center_dist_px and depth_dist <= depth_dist_m):
                duplicate = True
                break

        if not duplicate:
            keep.append(cand)

    return keep

def normalize_label(name):
    return str(name).strip().lower().replace("_", "-")

def get_stable_depth(depth_frame, cx, cy, width, height, win=2):
    depths = []
    for dx in range(-win, win + 1):
        for dy in range(-win, win + 1):
            px = cx + dx
            py = cy + dy
            if 0 <= px < width and 0 <= py < height:
                d = depth_frame.get_distance(px, py)
                if d > 0:
                    depths.append(d)

    if len(depths) == 0:
        return 0.0

    return float(np.median(depths))

def should_keep_detection(label, conf, x1, y1, x2, y2, depth):
    norm_label = normalize_label(label)
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    area = w * h
    aspect = h / w
    elongation = max(w, h) / max(1.0, min(w, h))
    min_conf = CLASS_MIN_CONF.get(norm_label, default_min_conf)
    if conf < min_conf:
        return False

    if norm_label in {"tube", "test-tube"}:
        if conf < tube_min_conf:
            return False
        if area < tube_min_area:
            return False
        if aspect < tube_min_aspect:
            return False
        if aspect > tube_max_aspect:
            return False
        if h < tube_min_height_px:
            return False
        if w > tube_max_width_px:
            return False
        if not (tube_depth_range_m[0] <= depth <= tube_depth_range_m[1]):
            return False
        return True

    if norm_label in {"tube-cap", "test-tube-cap"}:
        if conf < tube_cap_min_conf:
            return False
        if area < tube_cap_min_area:
            return False
        if area > tube_cap_max_area:
            return False
        if aspect < tube_cap_min_aspect or aspect > tube_cap_max_aspect:
            return False
        if min(w, h) < tube_cap_min_side_px:
            return False
        if not (tube_cap_depth_range_m[0] <= depth <= tube_cap_depth_range_m[1]):
            return False
        return True

    if norm_label == "beaker":
        if area < beaker_min_area:
            return False
        if not (beaker_aspect_range[0] <= aspect <= beaker_aspect_range[1]):
            return False
        return True

    if norm_label == "pipette":
        if area < pipette_min_area or area > pipette_max_area:
            return False
        if elongation < pipette_min_elongation:
            return False
        if not (pipette_depth_range_m[0] <= depth <= pipette_depth_range_m[1]):
            return False
        return True

    if norm_label == "spearhead":
        if area < spearhead_min_area or area > spearhead_max_area:
            return False
        if not (spearhead_aspect_range[0] <= aspect <= spearhead_aspect_range[1]):
            return False
        return True

    return True

def main():
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"妯″瀷涓嶅瓨鍦? {weights_path}")

    os.makedirs(save_dir, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU =", torch.cuda.get_device_name(0))
    print("device =", device)

    model = YOLO(weights_path)
    class_colors = generate_class_colors(model.names)
    target_labels = {"tube", "test-tube", "tube-cap", "test-tube-cap"}

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, camera_width, camera_height, rs.format.bgr8, camera_fps)
    config.enable_stream(rs.stream.depth, camera_width, camera_height, rs.format.z16, camera_fps)

    print("姝ｅ湪鍚姩 RealSense ...")
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_output_path, fourcc, video_save_fps, (camera_width, camera_height))

    frame_id = 0
    class_streaks = {}

    with open(csv_output_path, "w", newline="", encoding="utf-8-sig") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "frame_id",
            "class_name",
            "is_best",
            "conf",
            "x1", "y1", "x2", "y2",
            "cx", "cy",
            "depth_m",
            "X_cam_m", "Y_cam_m", "Z_cam_m",
            "X_robot_m", "Y_robot_m", "Z_robot_m"
        ])

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                vis = color_image.copy()

                results = model.predict(
                    source=color_image,
                    device=device,
                    conf=conf_thres,
                    iou=iou_thres,
                    max_det=max_det,
                    verbose=False
                )

                intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                grouped = {}

                for r in results:
                    if r.boxes is None:
                        continue

                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
                        cls_id = int(box.cls[0].detach().cpu().numpy()) if box.cls is not None else -1
                        conf = float(box.conf[0].detach().cpu().numpy()) if box.conf is not None else 0.0

                        label = model.names[cls_id] if cls_id in model.names else str(cls_id)
                        if ONLY_PROCESS_TARGET_CLASSES and normalize_label(label) not in target_labels:
                            continue

                        x1_i = int(x1)
                        y1_i = int(y1)
                        x2_i = int(x2)
                        y2_i = int(y2)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

                        if not (0 <= cx < camera_width and 0 <= cy < camera_height):
                            continue

                        depth = get_stable_depth(depth_frame, cx, cy, camera_width, camera_height, win=depth_win)
                        if depth <= 0:
                            continue

                        if not should_keep_detection(label, conf, x1, y1, x2, y2, depth):
                            continue

                        X_cam, Y_cam, Z_cam = rs.rs2_deproject_pixel_to_point(intrin, [cx, cy], depth)
                        X_robot, Y_robot, Z_robot = cam_to_robot(X_cam, Y_cam, Z_cam)

                        info = {
                            "label": label,
                            "conf": conf,
                            "x1": x1_i,
                            "y1": y1_i,
                            "x2": x2_i,
                            "y2": y2_i,
                            "cx": cx,
                            "cy": cy,
                            "area": area,
                            "depth": float(depth),
                            "X_cam": float(X_cam),
                            "Y_cam": float(Y_cam),
                            "Z_cam": float(Z_cam),
                            "X_robot": float(X_robot),
                            "Y_robot": float(Y_robot),
                            "Z_robot": float(Z_robot)
                        }

                        grouped.setdefault(label, []).append(info)

                # 姣忕被鏈€浣虫
                for label in list(grouped.keys()):
                    grouped[label] = suppress_near_duplicates(
                        grouped[label],
                        iou_thres=dedup_iou_thres,
                        center_dist_px=dedup_center_dist_px,
                        depth_dist_m=dedup_depth_dist_m,
                    )
                    if len(grouped[label]) == 0:
                        del grouped[label]

                present_labels = set(grouped.keys())
                for label in list(class_streaks.keys()):
                    if label not in present_labels:
                        class_streaks[label] = max(0, class_streaks[label] - 1)
                for label in present_labels:
                    class_streaks[label] = class_streaks.get(label, 0) + 1

                grouped = {
                    label: boxes
                    for label, boxes in grouped.items()
                    if class_streaks.get(label, 0) >= min_confirm_frames
                }

                best_map = {}
                for label, box_infos in grouped.items():
                    best_map[label] = choose_best_box(box_infos, camera_width, camera_height, strategy=best_strategy)

                # =========================
                # 鐢诲浘锛歵ube / tube-cap 鍏ㄧ敾锛涘叾瀹冨彧鐢绘渶浣?
                # =========================
                for label, box_infos in grouped.items():
                    color = class_colors.get(label, (0, 255, 0))
                    best_b = best_map[label]

                    if normalize_label(label) in SHOW_AND_EXPORT_ALL_CLASSES:
                        draw_list = box_infos
                    else:
                        draw_list = [best_b]

                    for b in draw_list:
                        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)

                        if show_center_for_all_boxes:
                            cv2.circle(vis, (b["cx"], b["cy"]), 5, (0, 0, 255), -1)

                    # 鍚屼竴绫诲埆鍙湪鏈€浣虫涓婂啓涓€娆℃爣绛?
                    cv2.putText(
                        vis,
                        best_b["label"],
                        (best_b["x1"], max(20, best_b["y1"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

                    robot_text = f"R=({best_b['X_robot']:.3f},{best_b['Y_robot']:.3f},{best_b['Z_robot']:.3f})"
                    cv2.putText(
                        vis,
                        robot_text,
                        (best_b["x1"], min(camera_height - 10, best_b["y2"] + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                # =========================
                # CSV锛歵ube / tube-cap 鍏ㄥ锛涘叾瀹冨彧瀵兼渶浣?
                # =========================
                for label, box_infos in grouped.items():
                    best_b = best_map[label]

                    if normalize_label(label) in SHOW_AND_EXPORT_ALL_CLASSES:
                        export_list = box_infos
                    else:
                        export_list = [best_b]

                    for b in export_list:
                        is_best = int(
                            b["x1"] == best_b["x1"] and
                            b["y1"] == best_b["y1"] and
                            b["x2"] == best_b["x2"] and
                            b["y2"] == best_b["y2"]
                        )

                        csv_writer.writerow([ 
                            frame_id,
                            b["label"],
                            is_best,
                            round(b["conf"], 4),
                            b["x1"], b["y1"], b["x2"], b["y2"],
                            b["cx"], b["cy"],
                            round(b["depth"], 4),
                            round(b["X_cam"], 4), round(b["Y_cam"], 4), round(b["Z_cam"], 4),
                            round(b["X_robot"], 4), round(b["Y_robot"], 4), round(b["Z_robot"], 4)
                        ])

                # =========================
                # 宸︿笂瑙掞細鍙樉绀烘瘡绫绘渶浣虫憳瑕?
                # =========================
                if show_top_left_info:
                    cv2.putText(
                        vis,
                        f"frame: {frame_id}",
                        (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    y = 50
                    for label in sorted(best_map.keys()):
                        b = best_map[label]
                        color = class_colors.get(label, (255, 255, 255))
                        line = (
                            f"{b['label']} "
                            f"conf={b['conf']:.2f} "
                            f"px=({b['cx']},{b['cy']}) "
                            f"Z={b['depth']:.3f}m"
                        )
                        cv2.putText(
                            vis,
                            line,
                            (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2
                        )
                        y += 24

                writer.write(vis)
                cv2.imshow("RealSense Show All + Export All Tube TubeCap", vis)

                print(f"\n===== frame {frame_id} =====")
                for label in sorted(best_map.keys()):
                    b = best_map[label]
                    extra = ""
                    if normalize_label(label) in SHOW_AND_EXPORT_ALL_CLASSES:
                        extra = f", all_count={len(grouped[label])}"
                    print(
                        f"{b['label']}: "
                        f"conf={b['conf']:.3f}, "
                        f"pixel=({b['cx']},{b['cy']}), "
                        f"depth={b['depth']:.3f} m, "
                        f"cam=({b['X_cam']:.3f}, {b['Y_cam']:.3f}, {b['Z_cam']:.3f}), "
                        f"robot=({b['X_robot']:.3f}, {b['Y_robot']:.3f}, {b['Z_robot']:.3f})"
                        f"{extra}"
                    )

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

                frame_id += 1

        finally:
            writer.release()
            pipeline.stop()
            cv2.destroyAllWindows()

    print("\n瀹屾垚")
    print(f"瑙嗛杈撳嚭: {video_output_path}")
    print(f"CSV杈撳嚭: {csv_output_path}")

if __name__ == "__main__":
    main()

