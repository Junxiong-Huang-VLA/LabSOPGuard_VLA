"""
合并所有数据集
"""
import shutil
from pathlib import Path

# 源数据（旧数据）
old_train_img = Path('D:/cvdemo/data/annotated/images/train')
old_train_lbl = Path('D:/cvdemo/data/annotated/labels/train')
old_val_img = Path('D:/cvdemo/data/annotated/images/val')
old_val_lbl = Path('D:/cvdemo/data/annotated/labels/val')

# 目标（合并到 raw）
raw_img = Path('D:/cvdemo/lab_detection/data/raw/color')
raw_lbl = Path('D:/cvdemo/lab_detection/data/raw/labels')

print("开始合并数据集...")

# 复制旧训练集
copied_img = 0
copied_lbl = 0

for src_img in old_train_img.glob('*.jpg'):
    dst = raw_img / src_img.name
    if not dst.exists():
        shutil.copy2(src_img, dst)
        copied_img += 1

for src_lbl in old_train_lbl.glob('*.txt'):
    dst = raw_lbl / src_lbl.name
    if not dst.exists():
        shutil.copy2(src_lbl, dst)
        copied_lbl += 1

print(f"训练集: 复制了 {copied_img} 张图片, {copied_lbl} 个标签")

# 复制旧验证集
copied_img = 0
copied_lbl = 0

for src_img in old_val_img.glob('*.jpg'):
    dst = raw_img / src_img.name
    if not dst.exists():
        shutil.copy2(src_img, dst)
        copied_img += 1

for src_lbl in old_val_lbl.glob('*.txt'):
    dst = raw_lbl / src_lbl.name
    if not dst.exists():
        shutil.copy2(src_lbl, dst)
        copied_lbl += 1

print(f"验证集: 复制了 {copied_img} 张图片, {copied_lbl} 个标签")

# 统计
total_img = len(list(raw_img.glob('*.jpg')))
total_lbl = len(list(raw_lbl.glob('*.txt')))

print(f"\n合并完成！")
print(f"总图片数: {total_img}")
print(f"总标签数: {total_lbl}")
