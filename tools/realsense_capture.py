"""
RealSense D435i 数据采集脚本
用法：python scripts/capture.py --output data/raw --num 100
按空格键保存当前帧，按 Q 退出
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense D435i 数据采集")
    parser.add_argument("--output", type=str, default="data/raw", help="输出目录")
    parser.add_argument("--num", type=int, default=100, help="目标采集数量")
    parser.add_argument("--width", type=int, default=1280, help="图像宽度")
    parser.add_argument("--height", type=int, default=720, help="图像高度")
    parser.add_argument("--fps", type=int, default=30, help="帧率")
    parser.add_argument("--save_depth", action="store_true", help="同时保存深度图")
    return parser.parse_args()


def setup_pipeline(width, height, fps):
    """初始化 RealSense 管道"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置彩色流
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    # 配置深度流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    
    # 启动管道
    profile = pipeline.start(config)
    
    # 获取深度传感器并设置高精度预设
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 3)  # 3 = High Accuracy
    
    print(f"[INFO] 相机已启动: {width}x{height} @ {fps}fps")
    return pipeline, profile


def main():
    args = parse_args()
    
    # 创建输出目录
    color_dir = Path(args.output) / "color"
    depth_dir = Path(args.output) / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    if args.save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化管道
    pipeline, profile = setup_pipeline(args.width, args.height, args.fps)
    
    # 对齐器：将深度图对齐到彩色图
    align = rs.align(rs.stream.color)
    
    # 深度图后处理滤波器
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    
    saved_count = 0
    frame_count = 0
    
    print(f"\n{'='*50}")
    print(f"目标采集: {args.num} 张")
    print(f"操作说明:")
    print(f"  [空格] 保存当前帧")
    print(f"  [A]    自动连续采集（每0.5秒保存一张）")
    print(f"  [Q]    退出")
    print(f"{'='*50}\n")
    
    auto_capture = False
    last_auto_save = 0
    
    try:
        while saved_count < args.num:
            # 等待帧
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            
            # 对齐深度到彩色
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 转为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            frame_count += 1
            
            # 显示进度和深度信息
            h, w = color_image.shape[:2]
            cx, cy = w // 2, h // 2
            center_depth = depth_frame.get_distance(cx, cy)
            
            display = color_image.copy()
            # 绘制中心点距离
            cv2.circle(display, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(display, f"Depth: {center_depth:.2f}m", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Saved: {saved_count}/{args.num}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            cv2.putText(display, "SPACE:save  A:auto  Q:quit", (10, h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if auto_capture:
                cv2.putText(display, "[AUTO]", (w-120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("RealSense D435i - 采集中", display)
            
            # 自动采集模式
            if auto_capture:
                now = time.time()
                if now - last_auto_save >= 0.5:
                    save_frame(color_image, depth_image, color_dir, depth_dir,
                               saved_count, args.save_depth)
                    saved_count += 1
                    last_auto_save = now
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                save_frame(color_image, depth_image, color_dir, depth_dir,
                           saved_count, args.save_depth)
                saved_count += 1
                print(f"[保存] {saved_count}/{args.num}")
            elif key == ord('a'):
                auto_capture = not auto_capture
                print(f"[自动采集] {'开启' if auto_capture else '关闭'}")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n✅ 采集完成，共保存 {saved_count} 张图像")
        print(f"   保存路径: {color_dir.absolute()}")


def save_frame(color_img, depth_img, color_dir, depth_dir, idx, save_depth):
    """保存彩色图和深度图"""
    timestamp = int(time.time() * 1000)
    fname = f"frame_{idx:04d}_{timestamp}"
    
    cv2.imwrite(str(color_dir / f"{fname}.jpg"), color_img,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    if save_depth:
        cv2.imwrite(str(depth_dir / f"{fname}.png"), depth_img)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        # 可以添加清理代码
        # pipeline.stop()  # 如果有pipeline对象
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"程序运行出错: {e}")