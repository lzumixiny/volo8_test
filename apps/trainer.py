import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml
from loguru import logger
from ultralytics import YOLO
from PIL import Image


class LockModelTrainer:
    """锁模型训练器 - 目标检测版本"""

    def __init__(self, dataset_path: str = "datasets/dataset_end/datasets", models_path: str = "models"):
        self.dataset_path = Path(dataset_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)

        # 使用现有的数据集结构，不需要重新创建
        # 检查数据集是否存在
        if not self.dataset_path.exists():
            logger.warning(f"数据集路径不存在: {self.dataset_path}")
            logger.info("将创建默认数据集结构")
            self._create_dataset_structure()

        # 锁类别定义 - 根据labelme标注调整
        self.lock_classes = {
            0: "upper_locked",
            1: "upper_unlocked",
            2: "upper_nolock",
            3: "lower_locked",
            4: "lower_unlocked",
            5: "lower_nolock",
            6: "knife_unlocked",
            7: "knife_locked",
            8: "lid_loose",
            9: "lid_tight"
        }

        # 类别名称到ID的映射
        self.class_name_to_id = {name: idx for idx, name in self.lock_classes.items()}

        # 
        self._create_dataset_config()

    def _create_dataset_structure(self):
        """创建数据集目录结构"""
        for split in ["train", "val", "test"]:
            for folder in ["images", "labels"]:
                (self.dataset_path / split / folder).mkdir(parents=True, exist_ok=True)

    def _create_dataset_config(self):
        """创建YOLO数据集配置文件"""
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.lock_classes),
            'names': self.lock_classes
        }

        config_path = self.dataset_path / 'dataset.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"YOLO数据集配置文件已创建: {config_path}")

    def generate_yolo_labels(self):
        """从labelme标注生成YOLO格式的标注文件"""
        logger.info("开始从labelme标注生成YOLO格式标注文件")

        # 统计处理的文件数量
        total_files = 0
        converted_files = 0

        for split in ["train", "val", "test"]:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue

            logger.info(f"处理 {split} 数据集...")

            # 查找所有labelme标注文件
            json_files = list(split_path.glob("*.json"))
            total_files += len(json_files)

            for json_file in json_files:
                try:
                    # 读取labelme标注文件
                    with open(json_file, 'r', encoding='utf-8') as f:
                        labelme_data = json.load(f)

                    # 获取对应的图像文件
                    image_file = json_file.parent / "images" / json_file.with_suffix('.png').name  # 假设是png格式
                    if not image_file.exists():
                        # 尝试其他格式
                        for ext in ['.jpg', '.jpeg', '.bmp']:
                            test_file = json_file.parent / "images" / json_file.with_suffix(ext).name
                            if test_file.exists():
                                image_file = test_file
                                break

                    if not image_file.exists():
                        logger.warning(f"未找到对应的图像文件: {json_file}")
                        continue

                    # 读取图像尺寸
                    with Image.open(image_file) as img:
                        img_width, img_height = img.size

                    # 转换标注
                    yolo_labels = []
                    if 'shapes' in labelme_data:
                        for shape in labelme_data['shapes']:
                            label = shape.get('label', '')
                            if label in self.class_name_to_id:
                                class_id = self.class_name_to_id[label]

                                # 处理多边形标注
                                if shape.get('shape_type') == 'polygon':
                                    points = shape.get('points', [])
                                    if len(points) >= 3:
                                        # 计算边界框
                                        x_coords = [p[0] for p in points]
                                        y_coords = [p[1] for p in points]
                                        x_min, x_max = min(x_coords), max(x_coords)
                                        y_min, y_max = min(y_coords), max(y_coords)

                                        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                                        x_center = (x_min + x_max) / 2 / img_width
                                        y_center = (y_min + y_max) / 2 / img_height
                                        width = (x_max - x_min) / img_width
                                        height = (y_max - y_min) / img_height

                                        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

                    # 保存YOLO格式标注文件
                    if yolo_labels:
                        label_dir = split_path / "labels"
                        label_dir.mkdir(exist_ok=True)

                        yolo_file = label_dir / (json_file.stem + ".txt")
                        with open(yolo_file, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(yolo_labels) + '\n')

                        # 移动图像文件到images目录
                        images_dir = split_path / "images"
                        images_dir.mkdir(exist_ok=True)

                        dest_image = images_dir / image_file.name
                        if dest_image != image_file:
                            shutil.move(str(image_file), str(dest_image))

                        converted_files += 1
                        logger.debug(f"转换完成: {json_file} -> {yolo_file}")

                except Exception as e:
                    logger.error(f"转换标注文件失败 {json_file}: {e}")

        logger.info(f"标注转换完成: {converted_files}/{total_files} 个文件")
        return converted_files > 0

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
        model_name: str = "lock_detector",
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        learning_rate: float = 0.001,
        device: str = "0",
    ) -> Dict:
        """训练锁目标检测模型

        Args:
            model_name: 模型名称
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 图片尺寸 (目标检测模型默认640)
            learning_rate: 学习率
            device: 训练设备

        Returns:
            Dict: 训练结果
        """
        try:
            logger.info(f"开始训练目标检测模型: {model_name}")

            # 首先生成YOLO格式的标注文件
            # if not self.generate_yolo_labels():
            #     logger.error("生成YOLO标注文件失败")
            #     return {"success": False, "error": "生成YOLO标注文件失败"}

            # 记录训练开始
            from .database import db_manager

            training_start = datetime.now()

            # 统计数据集大小
            train_images = len(list((self.dataset_path / "train" / "images").glob("*.*")))
            val_images = len(list((self.dataset_path / "val" / "images").glob("*.*")))

            training_record = {
                "model_name": model_name,
                "training_start": training_start,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dataset_size": train_images + val_images,
                "train_images": train_images,
                "val_images": val_images,
                "status": "training",
            }

            record_id = db_manager.save_training_record(training_record)

            # 加载预训练目标检测模型
            model = YOLO("yolo11n.pt")

            # 训练目标检测模型
            config_path = self.dataset_path / 'dataset.yaml'
            results = model.train(
                data=str(config_path),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                lr0=learning_rate,
                device=device,
                project=str(self.models_path),
                name=model_name,
                exist_ok=True,
                verbose=True,
                workers=2,
            )

            # 记录训练结果
            training_end = datetime.now()
            training_result = {
                "training_end": training_end,
                "train_loss": getattr(results, 'train_loss', 0),
                "val_loss": getattr(results, 'val_loss', 0),
                "map50": getattr(results, 'map50', 0),
                "map50_95": getattr(results, 'map50_95', 0),
                "precision": getattr(results, 'precision', 0),
                "recall": getattr(results, 'recall', 0),
                "model_path": str(
                    self.models_path / model_name / "weights" / "best.pt"
                ),
                "status": "completed",
            }

            # 如果数据库字段不存在，则跳过这些字段
            try:
                db_manager.update_training_record(record_id, training_result)
            except Exception as db_error:
                logger.warning(f"数据库更新失败，跳过数据库记录: {db_error}")

            logger.info(f"模型训练完成: {model_name}")

            return {
                "success": True,
                "model_path": training_result["model_path"],
                "map50": training_result["map50"],
                "map50_95": training_result["map50_95"],
                "precision": training_result["precision"],
                "recall": training_result["recall"],
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
        a = [1, 2, 3]
        a.reverse()
        """验证目标检测模型性能"""
        try:
            model = YOLO(model_path)

            # 在验证集上验证目标检测模型
            config_path = self.dataset_path / 'dataset.yaml'
            results = model.val(
                data=str(config_path), split="val"
            )

            return {
                "success": True,
                "map50": getattr(results, 'map50', 0),
                "map50_95": getattr(results, 'map50_95', 0),
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
        """获取目标检测数据集统计信息"""
        stats = {}

        for split in ["train", "val", "test"]:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue

            split_stats = {}
            total_images = 0
            total_labels = 0
            class_counts = {class_name: 0 for class_name in self.lock_classes.values()}

            # 统计图片数量
            images_folder = split_path / "images"
            if images_folder.exists():
                total_images = len(list(images_folder.glob("*.*")))

            # 统计标注文件数量和类别分布
            labels_folder = split_path / "labels"
            if labels_folder.exists():
                label_files = list(labels_folder.glob("*.txt"))
                total_labels = len(label_files)

                # 统计每个类别的标注数量
                for label_file in label_files:
                    try:
                        with open(label_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 1:
                                    class_id = int(parts[0])
                                    if class_id in self.lock_classes:
                                        class_name = self.lock_classes[class_id]
                                        class_counts[class_name] += 1
                    except Exception as e:
                        logger.warning(f"读取标注文件失败 {label_file}: {e}")

            stats[split] = {
                "total_images": total_images,
                "total_labels": total_labels,
                "class_distribution": class_counts,
                "total_annotations": sum(class_counts.values()),
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
