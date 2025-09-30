import os
import cv2
import yaml
import glob

# dataset.yaml 路径
yaml_path = "datasets/dataset_end/datasets/dataset.yaml"

# 读取 dataset.yaml 里的类别数
with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
names = data.get("names", [])
num_classes = len(names)
print(f"检测到 {num_classes} 个类别: {names}")

# 训练和验证集路径
train_dir = data.get("train")
val_dir = data.get("val")

def check_dataset(img_dir):
    print(f"\n🔍 正在检查 {img_dir}")
    img_files = glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True) + \
                glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True) + \
                glob.glob(os.path.join(img_dir, "**", "*.jpeg"), recursive=True)

    for img_path in img_files:
        # 检查图片是否能打开
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 无法打开图片: {img_path}")

        # 标签路径
        label_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(label_path):
            print(f"⚠️ 缺少标签文件: {label_path}")
            continue

        # 检查标签内容
        with open(label_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"❌ 标签格式错误 {label_path} 第 {i} 行: {line.strip()}")
                    continue

                cls, x, y, w, h = parts
                try:
                    cls = int(cls)
                    x, y, w, h = map(float, (x, y, w, h))
                except ValueError:
                    print(f"❌ 数据类型错误 {label_path} 第 {i} 行: {line.strip()}")
                    continue

                if cls < 0 or cls >= num_classes:
                    print(f"❌ 类别 ID 越界 {label_path} 第 {i} 行: {cls}")

                for val in (x, y, w, h):
                    if not (0 <= val <= 1):
                        print(f"❌ 坐标越界 {label_path} 第 {i} 行: {line.strip()}")

# 分别检查 train 和 val
if train_dir:
    check_dataset(train_dir)
if val_dir:
    check_dataset(val_dir)

print("\n✅ 检查完成")
