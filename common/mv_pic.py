import os
import shutil

def merge_images(src_dirs, dst_dir):
    """
    将多个文件夹中的图片合并到一个文件夹，并从1开始顺序命名
    :param src_dirs: [list] 源文件夹路径列表
    :param dst_dir: [str] 目标文件夹路径
    """
    # 支持的图片扩展名
    img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    # 确保目标文件夹存在
    os.makedirs(dst_dir, exist_ok=True)
    
    counter = 1  # 从1开始命名
    
    for src_dir in src_dirs:
        for root, _, files in os.walk(src_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in img_exts:  # 只处理图片文件
                    new_name = f"{counter}{ext}"
                    src_path = os.path.join(root, f)
                    dst_path = os.path.join(dst_dir, new_name)
                    
                    # 避免重名覆盖
                    while os.path.exists(dst_path):
                        counter += 1
                        new_name = f"{counter}{ext}"
                        dst_path = os.path.join(dst_dir, new_name)
                    
                    shutil.copy2(src_path, dst_path)  # 保留原文件属性
                    print(f"复制: {src_path} -> {dst_path}")
                    counter += 1

if __name__ == "__main__":
    # 示例用法
    src_dirs = [
        "datasets/dataset_end/20250905无锁",
        "datasets/dataset_end/20250905有锁",
        "datasets/dataset_end/无锁",
        "datasets/dataset_end/样本有锁",
        "datasets/dataset_end/样本有锁2",
        "datasets/dataset_end/测试图片",
    ]
    dst_dir = "datasets/dataset_end/datasets"
    merge_images(src_dirs, dst_dir)
    print("图片合并完成 ✅")
