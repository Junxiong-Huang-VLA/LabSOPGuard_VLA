#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取带 YOLO 检测的视频帧查看效果
"""

import cv2
import sys
from pathlib import Path

video_path = "D:/LabEmbodiedVLA/LabSOPGuard/outputs/video_analysis/test_with_yolo.mp4"
output_dir = Path("D:/LabEmbodiedVLA/LabSOPGuard/outputs/video_analysis/sample_frames")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Extracting frames from: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}, Total frames: {total_frames}")

# 提取关键帧（0s, 5s, 10s）
frame_times = [0, 5, 10]

for t in frame_times:
    frame_idx = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if ret:
        output_path = output_dir / f"frame_{t}s_with_yolo.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

cap.release()

print(f"\nFrames saved to: {output_dir}")
print("\nYou can now view these frames to see:")
print("- YOLO detection boxes (green/cyan/yellow/purple)")
print("- Class labels and confidence scores")
print("- VLM analysis (activities, objects)")
print("- PPE status indicators")
print("- Safety alerts")
