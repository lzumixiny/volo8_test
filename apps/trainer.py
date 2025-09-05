import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml
from loguru import logger
from ultralytics import YOLO


class LockModelTrainer:
    """锁模型训练器"""

    def __init__(self, dataset_path: str = "datasets/lock_detech", models_path: str = "models"):
        self.dataset_path = Path(dataset_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)

        # 使用现有的数据集结构，不需要重新创建
        # 检查数据集是否存在
        if not self.dataset_path.exists():
            logger.warning(f"数据集路径不存在: {self.dataset_path}")
            logger.info("将创建默认数据集结构")
            self._create_dataset_structure()

        # 锁类别定义 - 根据实际数据集调整
        self.lock_classes = {
            0: "locked",
            1: "unlocked",
        }

        # 创建数据集配置文件
        self._create_dataset_config()

    def _create_dataset_structure(self):
        """创建数据集目录结构"""
        for split in ["train", "val", "test"]:
            for folder in ["images", "labels"]:
                (self.dataset_path / split / folder).mkdir(parents=True, exist_ok=True)

    def _create_dataset_config(self):
        """创建数据集配置文件（分类任务不需要复杂的配置）"""
        # 分类任务直接使用文件夹结构，不需要创建配置文件
        logger.info("分类任务使用文件夹结构，不需要创建配置文件")

    def generate_yolo_labels(self):
        """分类任务不需要生成YOLO格式的标注文件"""
        logger.info("分类任务不需要生成YOLO格式的标注文件")
        # 分类任务直接使用文件夹结构，不需要标注文件

    def prepare_training_data(
        self, image_folder: str, class_mapping: Dict[str, int]
    ) -> bool:
        """准备训练数据

        Args:
            image_folder: 包含标注图片的文件夹路径
            class_mapping: 类别名称到ID的映射
        Returns:
            bool: 是否成功
        """
        try:
            source_folder = Path(image_folder)
            if not source_folder.exists():
                logger.error(f"源文件夹不存在: {source_folder}")
                return False

            # 统计文件
            image_files = (
                list(source_folder.glob("*.jpg"))
                + list(source_folder.glob("*.png"))
                + list(source_folder.glob("*.jpeg"))
                + list(source_folder.glob("*.bmp"))
            )

            if not image_files:
                logger.error(f"在 {source_folder} 中未找到图片文件")
                return False

            logger.info(f"找到 {len(image_files)} 张图片")

            # 分割数据集
            train_split = int(0.8 * len(image_files))
            val_split = int(0.9 * len(image_files))

            train_files = image_files[:train_split]
            val_files = image_files[train_split:val_split]
            test_files = image_files[val_split:]

            logger.info(
                f"训练集: {len(train_files)}, 验证集: {len(val_files)}, 测试集: {len(test_files)}"
            )

            # 复制文件到对应目录
            self._copy_files_to_split(train_files, "train", class_mapping)
            self._copy_files_to_split(val_files, "val", class_mapping)
            self._copy_files_to_split(test_files, "test", class_mapping)

            return True

        except Exception as e:
            logger.error(f"准备训练数据失败: {e}")
            return False

    def _copy_files_to_split(
        self, files: List[Path], split: str, class_mapping: Dict[str, int]
    ):
        """复制文件到指定分割集"""
        for img_file in files:
            try:
                # 复制图片
                dest_img = self.dataset_path / split / "images" / img_file.name
                shutil.copy2(img_file, dest_img)

                # 查找对应的标注文件
                label_file = img_file.with_suffix(".txt")
                if label_file.exists():
                    dest_label = self.dataset_path / split / "labels" / label_file.name
                    self._convert_label_format(label_file, dest_label, class_mapping)
                else:
                    logger.warning(f"未找到标注文件: {label_file}")

            except Exception as e:
                logger.error(f"复制文件失败 {img_file}: {e}")

    def _convert_label_format(
        self, src_label: Path, dest_label: Path, class_mapping: Dict[str, int]
    ):
        """转换标注格式"""
        try:
            with open(src_label, "r", encoding="utf-8") as f:
                lines = f.readlines()

            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 假设原始格式为: class_name x_center y_center width height
                    class_name = parts[0]
                    if class_name in class_mapping:
                        class_id = class_mapping[class_name]
                        converted_line = f"{class_id} {' '.join(parts[1:])}\n"
                        converted_lines.append(converted_line)

            with open(dest_label, "w", encoding="utf-8") as f:
                f.writelines(converted_lines)

        except Exception as e:
            logger.error(f"转换标注格式失败 {src_label}: {e}")

    def train_model(
        self,
        model_name: str = "lock_classifier",
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 224,
        learning_rate: float = 0.001,
        device: str = "0",
    ) -> Dict:
        """训练锁分类模型

        Args:
            model_name: 模型名称
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 图片尺寸 (分类模型默认224)
            learning_rate: 学习率
            device: 训练设备

        Returns:
            Dict: 训练结果
        """
        try:
            logger.info(f"开始训练分类模型: {model_name}")

            # 分类任务不需要生成YOLO标注文件
            # 直接使用文件夹结构作为分类标签

            # 记录训练开始
            from .database import db_manager

            training_start = datetime.now()

            training_record = {
                "model_name": model_name,
                "training_start": training_start,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dataset_size": len(
                    list((self.dataset_path / "train").glob("*/*.png")) +
                    list((self.dataset_path / "train").glob("*/*.jpg"))
                ),
                "status": "training",
            }

            record_id = db_manager.save_training_record(training_record)

            # 加载预训练分类模型
            model = YOLO("yolov8n-cls.pt")

            # 训练分类模型
            results = model.train(
                data=str(self.dataset_path),  # 分类模式直接使用数据集根目录
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                lr0=learning_rate,
                device=device,
                project=str(self.models_path),
                name=model_name,
                exist_ok=True,
                verbose=True,
            )

            # 记录训练结果
            training_end = datetime.now()
            training_result = {
                "training_end": training_end,
                "train_loss": getattr(results, 'train_loss', 0),
                "val_loss": getattr(results, 'val_loss', 0),
                "accuracy": getattr(results, 'top1', 0),
                "model_path": str(
                    self.models_path / model_name / "weights" / "best.pt"
                ),
                "status": "completed",
            }

            db_manager.update_training_record(record_id, training_result)

            logger.info(f"模型训练完成: {model_name}")

            return {
                "success": True,
                "model_path": training_result["model_path"],
                "accuracy": training_result["accuracy"],
                "train_loss": training_result["train_loss"],
                "val_loss": training_result["val_loss"],
                "training_time": (training_end - training_start).total_seconds(),
            }

        except Exception as e:
            logger.error(f"模型训练失败: {e}")

            # 更新训练记录为失败状态
            if "record_id" in locals():
                db_manager.update_training_record(record_id, {"status": "failed"})

            return {"success": False, "error": str(e)}

    def validate_model(self, model_path: str) -> Dict:
        """验证分类模型性能"""
        try:
            model = YOLO(model_path)

            # 在验证集上验证分类模型
            results = model.val(
                data=str(self.dataset_path), split="val"
            )

            return {
                "success": True,
                "accuracy": getattr(results, 'top1', 0),
                "accuracy_top5": getattr(results, 'top5', 0),
                "precision": getattr(results, 'precision', 0),
                "recall": getattr(results, 'recall', 0),
                "f1_score": getattr(results, 'f1', 0),
            }

        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            return {"success": False, "error": str(e)}

    def create_sample_dataset(self, num_samples: int = 100) -> bool:
        """创建示例数据集

        注意：这是一个示例方法，实际使用时需要替换为真实的数据收集方法
        """
        logger.info("创建示例数据集方法被调用，实际项目中应该替换为真实数据收集")

        # 这里应该实现数据爬取或数据收集逻辑
        # 由于涉及爬取，需要根据实际需求实现

        return True

    def get_dataset_stats(self) -> Dict:
        """获取数据集统计信息"""
        stats = {}

        for split in ["train", "val", "test"]:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            split_stats = {}
            total_images = 0
            total_labels = 0
            
            # 统计每个类别的图片数量
            for class_id, class_name in self.lock_classes.items():
                class_folder = split_path / class_name
                if class_folder.exists():
                    image_count = len(list(class_folder.glob("*.*")))
                    split_stats[class_name] = image_count
                    total_images += image_count
            
            # 统计标注文件数量
            labels_folder = split_path / "labels"
            if labels_folder.exists():
                total_labels = len(list(labels_folder.glob("*.txt")))
            
            stats[split] = {
                "total_images": total_images,
                "total_labels": total_labels,
                "class_distribution": split_stats,
                "path": str(split_path),
            }

        return stats

    def export_model(self, model_path: str, export_format: str = "onnx") -> str:
        """导出模型到指定格式"""
        try:
            model = YOLO(model_path)

            export_path = model.export(format=export_format)
            logger.info(f"模型导出成功: {export_path}")

            return str(export_path)

        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            return ""


# 全局训练器实例
trainer = LockModelTrainer()
