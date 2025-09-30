#!/usr/bin/env python3
"""
测试模型推理功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, "/home/lzumi/allCode/python/lock_det")

from loguru import logger
from PIL import Image
from ultralytics import YOLO

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO")


def test_model_inference():
    """测试模型推理"""
    try:
        # 加载模型
        model_path = "models/lock_detechtor_final/weights/best.pt"
        if not Path(model_path).exists():
            logger.error(f"模型文件不存在: {model_path}")
            return

        model = YOLO(model_path)
        logger.info("模型加载成功")

        # 检查模型类别
        if hasattr(model, "names"):
            logger.info(f"模型类别: {model.names}")

        # 测试图片
        test_images = [
            "datasets/dataset_end/datasets/test/images/12.png",
            "datasets/dataset_end/datasets/test/images/2.png",
            "datasets/dataset_end/datasets/test/images/15.png",
        ]

        for img_path in test_images:
            if Path(img_path).exists():
                logger.info(f"\n测试图片: {img_path}")

                # 加载图片
                image = Image.open(img_path)

                # 进行推理
                results = model.predict(source=image, conf=0.5, verbose=True)

                # 分析结果
                if results and len(results) > 0:
                    result = results[0]
                    logger.info(f"检测到 {len(result.boxes)} 个对象")

                    if hasattr(result, "boxes") and result.boxes is not None:
                        boxes = result.boxes
                        for i, box in enumerate(boxes):
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())

                            logger.info(f"对象 {i + 1}:")
                            logger.info(
                                f"  类别: {class_id} -> {model.names[class_id]}"
                            )
                            logger.info(f"  置信度: {confidence:.3f}")
                            logger.info(
                                f"  边界框: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})"
                            )
                            logger.info(f"  宽度: {x2 - x1:.1f}, 高度: {y2 - y1:.1f}")

                            # 判断锁状态
                            class_name = model.names[class_id]
                            is_locked = (
                                "lock" in class_name.lower()
                                and "unlock" not in class_name.lower()
                            )
                            logger.info(
                                f"  锁状态: {'锁定' if is_locked else '未锁定'}"
                            )
                else:
                    logger.info("未检测到任何对象")
            else:
                logger.warning(f"图片不存在: {img_path}")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_model_inference()
