#!/usr/bin/env python3
"""
简化版命令行工具入口脚本
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="锁检测系统命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        help="要执行的命令"
    )
    
    parser.add_argument(
        "--list-commands",
        action="store_true",
        help="列出所有可用命令"
    )
    
    # 解析已知参数，剩余参数传递给具体命令
    args, remaining = parser.parse_known_args()
    
    if args.list_commands or not args.command:
        print("可用命令:")
        print("  train                启动锁检测模型训练")
        print("  dataset_stats        显示数据集统计信息")
        print("  export_model         导出训练好的模型")
        return
    
    # 执行命令
    if args.command == "train":
        train_command(remaining)
    elif args.command == "dataset_stats":
        dataset_stats_command(remaining)
    elif args.command == "export_model":
        export_model_command(remaining)
    else:
        print(f"未知命令: {args.command}")
        print("使用 --list-commands 查看可用命令")


def train_command(args):
    """训练命令"""
    parser = argparse.ArgumentParser(description="启动锁检测模型训练")
    parser.add_argument("--model-name", type=str, default="lock_detector", help="模型名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--img-size", type=int, default=640, help="图片尺寸")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--device", type=str, default="0", help="训练设备")
    parser.add_argument("--dataset-path", type=str, default="datasets/lock_detech", help="数据集路径")
    parser.add_argument("--validate", action="store_true", help="训练完成后验证模型")
    
    options = parser.parse_args(args)
    
    print(f"开始训练模型: {options.model_name}")
    print(f"数据集路径: {options.dataset_path}")
    print(f"训练配置: epochs={options.epochs}, batch_size={options.batch_size}")
    print("注意: 需要先安装所需依赖 (pip install -r requirements.txt)")
    print("实际训练功能将在安装依赖后可用")


def dataset_stats_command(args):
    """数据集统计命令"""
    import os
    from pathlib import Path
    
    dataset_path = Path("datasets/lock_detech")
    
    if not dataset_path.exists():
        print(f"数据集路径不存在: {dataset_path}")
        return
    
    print("数据集统计信息:")
    print("=" * 50)
    
    for split in ["train", "val", "test"]:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
            
        print(f"\n{split.upper()} 集合:")
        print(f"  路径: {split_path}")
        
        total_images = 0
        for class_name in ["locked", "unlocked"]:
            class_folder = split_path / class_name
            if class_folder.exists():
                image_count = len([f for f in class_folder.glob("*") if f.is_file()])
                print(f"  {class_name}: {image_count} 张图片")
                total_images += image_count
        
        print(f"  总计: {total_images} 张图片")
        
        # 检查标注文件
        labels_folder = split_path / "labels"
        if labels_folder.exists():
            label_count = len(list(labels_folder.glob("*.txt")))
            print(f"  标注文件: {label_count} 个")
        else:
            print(f"  标注文件: 未生成")
    
    print("\n" + "=" * 50)
    print("提示: 运行训练命令会自动生成YOLO格式标注文件")


def export_model_command(args):
    """模型导出命令"""
    parser = argparse.ArgumentParser(description="导出训练好的模型")
    parser.add_argument("model_path", type=str, help="模型文件路径")
    parser.add_argument("--format", type=str, default="onnx", 
                       choices=["onnx", "torchscript", "coreml", "tensorflow"], 
                       help="导出格式")
    
    options = parser.parse_args(args)
    
    print(f"导出模型: {options.model_path} -> {options.format}")
    print("注意: 需要先安装所需依赖 (pip install -r requirements.txt)")
    print("实际导出功能将在安装依赖后可用")


if __name__ == "__main__":
    main()