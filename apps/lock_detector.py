import json
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from .database import DetectionResult, db_manager


class LockDetectionResult:
    """锁检测结果"""
    
    def __init__(self):
        self.is_safe = True
        self.total_locks = 0
        self.unlocked_locks = 0
        self.locked_locks = 0
        self.lock_details = []
        self.confidence_score = 0.0
        self.detection_time = datetime.now()
    
    def add_lock(self, lock_type: str, is_locked: bool, confidence: float, bbox: Dict):
        """添加锁检测结果"""
        lock_detail = {
            'lock_type': lock_type,
            'is_locked': is_locked,
            'confidence': confidence,
            'bbox': bbox,
            'position_x': int(bbox['xmin']),
            'position_y': int(bbox['ymin']),
            'width': int(bbox['xmax'] - bbox['xmin']),
            'height': int(bbox['ymax'] - bbox['ymin'])
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
            'is_safe': self.is_safe,
            'total_locks': self.total_locks,
            'unlocked_locks': self.unlocked_locks,
            'locked_locks': self.locked_locks,
            'lock_details': self.lock_details,
            'confidence_score': self.confidence_score,
            'detection_time': self.detection_time.isoformat()
        }


class LockDetector:
    """锁分类器"""
    
    def __init__(self, model_path: str = "models/lock_classifier/weights/best.pt"):
        self.model_path = model_path
        self.model = None
        self.lock_classes = {
            0: "labels",
            1: "locked", 
            2: "unlocked"
        }
        self.load_model()
    
    def load_model(self):
        """加载分类模型"""
        try:
            if self.model_path and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"分类模型加载成功: {self.model_path}")
            else:
                logger.warning(f"分类模型文件不存在: {self.model_path}，使用默认分类模型")
                self.model = YOLO('yolov8n-cls.pt')
        except Exception as e:
            logger.error(f"分类模型加载失败: {e}")
            self.model = YOLO('yolov8n-cls.pt')
    
    def detect_locks(self, image: Image.Image, confidence_threshold: float = 0.5) -> LockDetectionResult:
        """分类图片中的锁状态"""
        try:
            result = LockDetectionResult()
            
            # 使用分类模型进行预测
            predictions = self.model.predict(
                source=image,
                conf=confidence_threshold,
                imgsz=224,
                verbose=False
            )
            
            # 处理分类预测结果
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                
                # 获取分类概率
                if hasattr(pred, 'probs') and pred.probs is not None:
                    probs = pred.probs
                    top1_class = int(probs.top1)
                    top1_conf = float(probs.top1conf)
                    
                    # 获取锁状态
                    lock_status = self.lock_classes.get(top1_class, "unknown")
                    is_locked = (lock_status == "locked")
                    
                    # 使用整个图像作为边界框
                    width, height = image.size
                    bbox = {
                        'xmin': 0,
                        'ymin': 0,
                        'xmax': width,
                        'ymax': height
                    }
                    
                    # 添加检测结果
                    result.add_lock(lock_status, is_locked, top1_conf, bbox)
            
            logger.info(f"锁分类结果: {result.total_locks} 个锁, {result.unlocked_locks} 个未锁定")
            return result
            
        except Exception as e:
            logger.error(f"锁分类失败: {e}")
            return LockDetectionResult()
    
    # 分类任务不再需要复杂的状态判断方法
    
    def visualize_detection(self, image: Image.Image, detection_result: LockDetectionResult) -> Image.Image:
        """可视化分类结果"""
        try:
            from .detech import create_classification_visualization
            
            if detection_result.lock_details:
                # 获取第一个（也是唯一一个）分类结果
                lock_detail = detection_result.lock_details[0]
                class_name = lock_detail['lock_type']
                confidence = lock_detail['confidence']
                
                # 使用分类可视化函数
                return create_classification_visualization(image, class_name, confidence)
            else:
                # 如果没有结果，返回原图
                return image
            
        except Exception as e:
            logger.error(f"可视化分类结果失败: {e}")
            return image
    
    def save_detection_result(self, image: Image.Image, detection_result: LockDetectionResult,
                            message_id: str = "", user_id: str = "", group_id: str = "") -> int:
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
                is_safe=detection_result.is_safe
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
            gray = image.convert('L')
            
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