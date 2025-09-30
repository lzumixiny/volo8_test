import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from loguru import logger
from PIL import Image
from ultralytics import YOLO

from .database import DetectionResult, db_manager


class LockDetectionResult:
    """锁检测结果"""

    def __init__(self):
        self.is_safe = True  # 最终的安全性， 由上锁位+下锁位+盖子+锁扣同时决定
        self.total_locks = 0  # 识别到锁的数量
        self.unlocked_locks = 0  # 当前未锁定的数量  锁的定义是 上锁位+下锁位+锁扣
        self.locked_locks = 0  # 当前锁定的数量
        self.lock_details = []  # 详情
        self.confidence_score = 0.0  # 最大置信度
        self.detection_time = datetime.now()

    def add_lock(self, lock_type: str, is_locked: bool, confidence: float, bbox: Dict):
        """添加锁检测结果"""
        lock_detail = {
            "lock_type": lock_type,
            "is_locked": is_locked,
            "confidence": confidence,
            "bbox": bbox,
            "position_x": int(bbox["xmin"]),
            "position_y": int(bbox["ymin"]),
            "width": int(bbox["xmax"] - bbox["xmin"]),
            "height": int(bbox["ymax"] - bbox["ymin"]),
        }

        self.lock_details.append(lock_detail)
        self.total_locks += 1

        if is_locked:
            self.locked_locks += 1
        else:
            self.unlocked_locks += 1
            self.is_safe = False

        # 更新整体置信度
        self.confidence_score = max(self.confidence_score, confidence)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "is_safe": self.is_safe,
            "total_locks": self.total_locks,
            "unlocked_locks": self.unlocked_locks,
            "locked_locks": self.locked_locks,
            "lock_details": self.lock_details,
            "confidence_score": self.confidence_score,
            "detection_time": self.detection_time.isoformat(),
        }


class LockDetector:
    """锁检测器 - 支持多锁检测和状态标记"""

    def __init__(self, model_path: str = "models/lock_detechtor_final/weights/best.pt"):
        self.model_path = model_path
        self.model = YOLO("yolov8n.pt")  # 默认值
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
            9: "lid_tight",
        }
        self.lock_classes_cn = {
            0: "上锁位锁定",
            1: "上锁位未锁定",
            2: "上锁位无锁",
            3: "下锁位锁定",
            4: "下锁位未锁定",
            5: "下锁位无锁",
            6: "闸刀未锁定",
            7: "闸刀锁定",
            8: "盖子未盖紧",
            9: "盖子已盖紧",
        }
        self.load_model()

    def load_model(self):
        """加载目标检测模型"""
        try:
            if self.model_path and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"目标检测模型加载成功: {self.model_path}")
            else:
                logger.warning(
                    f"目标检测模型文件不存在: {self.model_path}，使用默认检测模型"
                )
                self.model = YOLO("yolov8n.pt")
        except Exception as e:
            logger.error(f"目标检测模型加载失败: {e}")
            self.model = YOLO("yolov8n.pt")

    def detect_locks(
        self, image: Image.Image, confidence_threshold: float = 0.5
    ) -> LockDetectionResult:
        """检测图片中的锁状态和位置"""
        try:
            result = LockDetectionResult()

            # 使用目标检测模型进行预测
            predictions = self.model.predict(
                source=image,
                conf=confidence_threshold,
                project="runs/detect/webout",
                name="output",
                verbose=False,
                save=True,
                exist_ok=True,
            )

            # 处理检测预测结果
            if predictions and len(predictions) > 0:
                pred = predictions[0]

                # 获取检测框
                if hasattr(pred, "boxes") and pred.boxes is not None:
                    boxes = pred.boxes
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        # 获取锁状态
                        lock_status = self.lock_classes_cn.get(class_id, "未知")

                        is_locked = class_id in (0, 3, 7, 9)

                        # 创建边界框
                        bbox = {
                            "xmin": float(x1),
                            "ymin": float(y1),
                            "xmax": float(x2),
                            "ymax": float(y2),
                        }

                        # 添加检测结果
                        result.add_lock(lock_status, is_locked, confidence, bbox)

            logger.info(
                f"锁检测结果: {result.total_locks} 个锁, {result.unlocked_locks} 个未锁定"
            )
            return result

        except Exception as e:
            logger.error(f"锁检测失败: {e}")
            return LockDetectionResult()

    # 分类任务不再需要复杂的状态判断方法

    def visualize_detection(
        self, image: Image.Image, detection_result: LockDetectionResult
    ) -> Image.Image:
        """可视化检测结果 - 为每个锁绘制边界框"""
        try:
            if not detection_result.lock_details:
                # 如果没有检测结果，返回原图
                return image

            # 创建图像副本
            result_image = image.copy()

            # # 转换为numpy数组进行绘制
            # import cv2
            # import numpy as np

            # img_array = np.array(result_image)

            # # 为每个锁绘制边界框
            # for i, lock_detail in enumerate(detection_result.lock_details):
            #     bbox = lock_detail["bbox"]
            #     lock_type = lock_detail["lock_type"]
            #     confidence = lock_detail["confidence"]
            #     is_locked = lock_detail["is_locked"]

            #     # 获取边界框坐标
            #     x1, y1, x2, y2 = (
            #         int(bbox["xmin"]),
            #         int(bbox["ymin"]),
            #         int(bbox["xmax"]),
            #         int(bbox["ymax"]),
            #     )

            #     # 根据锁状态选择颜色
            #     if is_locked:
            #         color = (0, 255, 0)  # 绿色 - 锁定
            #         status_text = "LOCKED"
            #     else:
            #         color = (255, 0, 0)  # 红色 - 未锁定
            #         status_text = "UNLOCKED"

            #     # 绘制边界框
            #     cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)

            #     # 准备标签文本
            #     label = f"{lock_type.upper()} {status_text} ({confidence:.2f})"

            #     # 计算文本大小
            #     (text_width, text_height), baseline = cv2.getTextSize(
            #         label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            #     )

            #     # 绘制标签背景
            #     cv2.rectangle(
            #         img_array,
            #         (x1, y1 - text_height - baseline - 10),
            #         (x1 + text_width, y1),
            #         color,
            #         -1,
            #     )

            #     # 绘制标签文本
            #     cv2.putText(
            #         img_array,
            #         label,
            #         (x1, y1 - 5),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.7,
            #         (255, 255, 255),
            #         2,
            #     )

            #     # 在锁的编号
            #     lock_id_text = f"Lock #{i + 1}"
            #     cv2.putText(
            #         img_array,
            #         lock_id_text,
            #         (x1, y2 + 25),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.6,
            #         color,
            #         2,
            #     )

            # # 转换回PIL图像
            # result_image = Image.fromarray(img_array)

            # # 在图像顶部添加总体状态信息
            # from PIL import ImageDraw, ImageFont

            # draw = ImageDraw.Draw(result_image)

            # # 尝试加载字体
            # try:
            #     font = ImageFont.truetype("arial.ttf", 24)
            #     small_font = ImageFont.truetype("arial.ttf", 18)
            # except:
            #     font = ImageFont.load_default()
            #     small_font = ImageFont.load_default()

            # # 准备总体状态文本
            # total_text = f"总计: {detection_result.total_locks} 个锁"
            # locked_text = f"锁定: {detection_result.locked_locks} 个"
            # unlocked_text = f"未锁定: {detection_result.unlocked_locks} 个"
            # safety_text = f"状态: {'安全' if detection_result.is_safe else '警告'}"

            # # 根据安全状态选择颜色
            # safety_color = (0, 255, 0) if detection_result.is_safe else (255, 0, 0)

            # # 在图像顶部绘制半透明背景
            # overlay = Image.new("RGBA", result_image.size, (0, 0, 0, 0))
            # overlay_draw = ImageDraw.Draw(overlay)

            # 绘制背景矩形
            # text_bbox = draw.textbbox((10, 10), total_text, font=font)
            # overlay_draw.rectangle(
            #     [
            #         text_bbox[0] - 5,
            #         text_bbox[1] - 5,
            #         text_bbox[2] + 5,
            #         text_bbox[3] + 5,
            #     ],
            #     fill=(0, 0, 0, 180),
            # )

            # 叠加到原图上
            # result_image = Image.alpha_composite(
            #     result_image.convert("RGBA"), overlay
            # ).convert("RGB")

            # 重新创建绘制对象
            # draw = ImageDraw.Draw(result_image)

            # # 绘制文本
            # draw.text((10, 10), total_text, fill=(255, 255, 255), font=font)
            # draw.text((10, 45), locked_text, fill=(0, 255, 0), font=small_font)
            # draw.text((10, 70), unlocked_text, fill=(255, 0, 0), font=small_font)
            # draw.text((10, 95), safety_text, fill=safety_color, font=small_font)

            return result_image

        except Exception as e:
            logger.error(f"可视化检测结果失败: {e}")
            return image

    def save_detection_result(
        self,
        image: Image.Image,
        detection_result: LockDetectionResult,
        message_id: str = "",
        user_id: str = "",
        group_id: str = "",
    ) -> int:
        """保存检测结果到数据库"""
        try:
            # 计算图片哈希
            image_hash = self._calculate_image_hash(image)

            # 创建检测结果对象
            result = DetectionResult(
                image_url="",  # 暂时为空
                image_hash=image_hash,
                detection_time=detection_result.detection_time,
                locks_detected=detection_result.total_locks,
                unlocked_locks=detection_result.unlocked_locks,
                lock_positions=json.dumps(detection_result.lock_details),
                confidence_score=detection_result.confidence_score,
                dingtalk_message_id=message_id,
                user_id=user_id,
                group_id=group_id,
                is_safe=detection_result.is_safe,
            )

            # 保存到数据库
            detection_id = db_manager.save_detection_result(result)

            # 保存锁的详细信息
            db_manager.save_lock_details(detection_id, detection_result.lock_details)

            logger.info(f"检测结果已保存到数据库: {detection_id}")
            return detection_id

        except Exception as e:
            logger.error(f"保存检测结果失败: {e}")
            return -1

    def _calculate_image_hash(self, image: Image.Image) -> str:
        """计算图片哈希值"""
        try:
            # 转换为灰度图
            gray = image.convert("L")

            # 计算MD5哈希
            import hashlib

            md5_hash = hashlib.md5(gray.tobytes()).hexdigest()

            return md5_hash

        except Exception as e:
            logger.error(f"计算图片哈希失败: {e}")
            return ""

    def get_detection_statistics(self) -> Dict:
        """获取检测统计信息"""
        try:
            return db_manager.get_statistics()
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    def get_detection_history(self, limit: int = 10) -> List[Dict]:
        """获取检测历史"""
        try:
            return db_manager.get_detection_history(limit=limit)
        except Exception as e:
            logger.error(f"获取检测历史失败: {e}")
            return []


# 全局检测器实例
detector = LockDetector()
