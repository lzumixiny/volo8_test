"""
训练命令 - 启动模型训练
"""

from pathlib import Path
from .commands import BaseCommand
from .trainer import trainer
from loguru import logger


class TrainCommand(BaseCommand):
    """训练命令"""

    help = "启动锁检测模型训练"
    name = "train"

    def add_arguments(self):
        """添加命令行参数"""
        self.parser.add_argument(
            "--model-name",
            type=str,
            default="lock_classifier",
            help="模型名称 (默认: lock_classifier)"
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="训练轮数 (默认: 100)"
        )
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            help="批次大小 (默认: 16)"
        )
        self.parser.add_argument(
            "--img-size",
            type=int,
            default=224,
            help="图片尺寸 (默认: 224)"
        )
        self.parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.001,
            help="学习率 (默认: 0.001)"
        )
        self.parser.add_argument(
            "--device",
            type=str,
            default="0",
            help="训练设备 (默认: 0)"
        )
        self.parser.add_argument(
            "--dataset-path",
            type=str,
            default="datasets/lock_detech",
            help="数据集路径 (默认: datasets/lock_detech)"
        )
        self.parser.add_argument(
            "--validate",
            action="store_true",
            help="训练完成后验证模型"
        )

    def handle(self, **options):
        """处理训练命令"""
        logger.info("开始执行训练命令")
        logger.info(f"训练配置: {options}")

        # 更新训练器的数据集路径
        dataset_path = options["dataset_path"]
        if str(Path(dataset_path)) != str(trainer.dataset_path):
            logger.info(f"使用数据集路径: {dataset_path}")
            trainer.dataset_path = Path(dataset_path)
            trainer._create_dataset_config()

        # 执行训练
        result = trainer.train_model(
            model_name=options["model_name"],
            epochs=options["epochs"],
            batch_size=options["batch_size"],
            img_size=options["img_size"],
            learning_rate=options["learning_rate"],
            device=options["device"]
        )

        if result["success"]:
            logger.info("训练完成!")
            logger.info(f"模型路径: {result['model_path']}")
            logger.info(f"准确率: {result['accuracy']:.4f}")
            logger.info(f"训练时间: {result['training_time']:.2f}秒")

            # 如果需要验证
            if options.get("validate"):
                self._validate_model(result["model_path"])

            return result
        else:
            logger.error(f"训练失败: {result['error']}")
            return result

    # 数据准备方法已移除，现在直接使用datasets文件夹中的数据

    def _validate_model(self, model_path: str):
        """验证分类模型"""
        logger.info("验证分类模型性能...")
        validation_result = trainer.validate_model(model_path)

        if validation_result["success"]:
            logger.info("验证结果:")
            logger.info(f"  准确率(Top-1): {validation_result['accuracy']:.4f}")
            logger.info(f"  准确率(Top-5): {validation_result['accuracy_top5']:.4f}")
            logger.info(f"  精确率: {validation_result['precision']:.4f}")
            logger.info(f"  召回率: {validation_result['recall']:.4f}")
            logger.info(f"  F1分数: {validation_result['f1_score']:.4f}")
        else:
            logger.error(f"验证失败: {validation_result['error']}")


class DatasetStatsCommand(BaseCommand):
    """数据集统计命令"""

    help = "显示数据集统计信息"
    name = "dataset_stats"

    def handle(self, **options):
        """处理数据集统计命令"""
        logger.info("获取数据集统计信息...")
        stats = trainer.get_dataset_stats()

        logger.info("数据集统计:")
        for split, data in stats.items():
            logger.info(f"  {split}:")
            logger.info(f"    总图片数: {data['total_images']}")
            logger.info(f"    总标注数: {data['total_labels']}")
            logger.info(f"    类别分布: {data['class_distribution']}")
            logger.info(f"    路径: {data['path']}")

        return stats


class ExportModelCommand(BaseCommand):
    """模型导出命令"""

    help = "导出训练好的模型"
    name = "export_model"

    def add_arguments(self):
        """添加命令行参数"""
        self.parser.add_argument(
            "model_path",
            type=str,
            help="模型文件路径"
        )
        self.parser.add_argument(
            "--format",
            type=str,
            default="onnx",
            choices=["onnx", "torchscript", "coreml", "tensorflow"],
            help="导出格式 (默认: onnx)"
        )

    def handle(self, **options):
        """处理模型导出命令"""
        logger.info(f"导出模型: {options['model_path']} -> {options['format']}")

        export_path = trainer.export_model(options["model_path"], options["format"])

        if export_path:
            logger.info(f"模型导出成功: {export_path}")
            return {"success": True, "export_path": export_path}
        else:
            logger.error("模型导出失败")
            return {"success": False, "error": "导出失败"}