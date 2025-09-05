"""
图像处理和检测工具函数
支持图像格式转换、模型预测、可视化等功能
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from PIL import Image
from loguru import logger
from ultralytics import YOLO
from io import BytesIO


def get_image_from_bytes(byte_array: bytes) -> Image.Image:
    """
    将字节数组转换为PIL图像对象
    
    Args:
        byte_array: 图像的字节数组
        
    Returns:
        PIL.Image.Image: 转换后的图像对象
    """
    try:
        # 使用BytesIO来处理字节数组
        image_stream = BytesIO(byte_array)
        image = Image.open(image_stream)
        # 转换为RGB模式（如果图像是RGBA或其他格式）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logger.error(f"图像转换失败: {e}")
        raise ValueError(f"无法解析图像文件: {e}")


def get_bytes_from_image(image: Image.Image) -> bytes:
    """
    将PIL图像对象转换为字节数组
    
    Args:
        image: PIL图像对象
        
    Returns:
        bytes: 图像的字节数组
    """
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return buffered.getvalue()
    except Exception as e:
        logger.error(f"图像转换失败: {e}")
        raise ValueError(f"无法转换图像为字节: {e}")


def detect_sample_model(image: Image.Image, model: YOLO = None, confidence_threshold: float = 0.5) -> pd.DataFrame:
    """
    使用模型进行图像检测（兼容检测和分类任务）
    
    Args:
        image: PIL图像对象
        model: YOLO模型对象
        confidence_threshold: 置信度阈值
        
    Returns:
        pd.DataFrame: 检测结果
    """
    try:
        if model is None:
            logger.warning("未提供模型，返回空结果")
            return pd.DataFrame()
        
        # 进行预测
        results = model.predict(
            source=image,
            conf=confidence_threshold,
            verbose=False
        )
        
        # 处理预测结果
        predict_data = []
        
        if results and len(results) > 0:
            result = results[0]
            
            # 判断是检测任务还是分类任务
            if hasattr(result, 'boxes') and result.boxes is not None:
                # 检测任务 - 有边界框
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names.get(class_id, f"class_{class_id}")
                    
                    predict_data.append({
                        'xmin': float(x1),
                        'ymin': float(y1),
                        'xmax': float(x2),
                        'ymax': float(y2),
                        'confidence': float(confidence),
                        'class': int(class_id),
                        'name': class_name
                    })
            
            elif hasattr(result, 'probs') and result.probs is not None:
                # 分类任务 - 只有类别概率
                probs = result.probs
                top1_class = int(probs.top1)
                top1_conf = float(probs.top1conf)
                class_name = model.names.get(top1_class, f"class_{top1_class}")
                
                # 对于分类任务，使用整个图像作为边界框
                width, height = image.size
                predict_data.append({
                    'xmin': 0,
                    'ymin': 0,
                    'xmax': float(width),
                    'ymax': float(height),
                    'confidence': top1_conf,
                    'class': top1_class,
                    'name': class_name
                })
        
        # 转换为DataFrame
        if predict_data:
            return pd.DataFrame(predict_data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        return pd.DataFrame()


def add_bboxs_on_img(image: Image.Image, predict: pd.DataFrame, model: YOLO = None) -> Image.Image:
    """
    在图像上绘制边界框和标签
    
    Args:
        image: 原始图像
        predict: 预测结果DataFrame
        model: YOLO模型（用于获取类别名称）
        
    Returns:
        PIL.Image.Image: 带有标注的图像
    """
    try:
        if predict.empty:
            logger.warning("预测结果为空，返回原图像")
            return image
        
        # 创建图像副本
        result_image = image.copy()
        
        # 如果是PIL图像，转换为numpy数组进行绘制
        import cv2
        img_array = np.array(result_image)
        
        # 获取类别名称
        class_names = {}
        if model and hasattr(model, 'names'):
            class_names = model.names
        
        # 绘制每个检测框
        for _, row in predict.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            confidence = row['confidence']
            class_id = row['class']
            
            # 获取类别名称
            class_name = class_names.get(class_id, f"class_{class_id}")
            
            # 选择颜色（基于类别ID）
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 红色
                (0, 0, 255),    # 蓝色
                (255, 255, 0),  # 黄色
                (255, 0, 255),  # 紫色
            ]
            color = colors[class_id % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            label = f"{class_name}: {confidence:.2f}"
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 绘制标签背景
            cv2.rectangle(img_array, (x1, y1 - text_height - baseline - 5), 
                        (x1 + text_width, y1), color, -1)
            
            # 绘制标签文本
            cv2.putText(img_array, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 转换回PIL图像
        return Image.fromarray(img_array)
        
    except Exception as e:
        logger.error(f"绘制边界框失败: {e}")
        return image


def classify_image(image: Image.Image, model: YOLO, confidence_threshold: float = 0.5) -> Tuple[str, float]:
    """
    使用分类模型对图像进行分类
    
    Args:
        image: PIL图像对象
        model: YOLO分类模型
        confidence_threshold: 置信度阈值
        
    Returns:
        Tuple[str, float]: (类别名称, 置信度)
    """
    try:
        # 进行预测
        results = model.predict(
            source=image,
            conf=confidence_threshold,
            verbose=False
        )
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'probs') and result.probs is not None:
                probs = result.probs
                top1_class = int(probs.top1)
                top1_conf = float(probs.top1conf)
                class_name = model.names.get(top1_class, f"class_{top1_class}")
                
                return class_name, top1_conf
        
        return "unknown", 0.0
        
    except Exception as e:
        logger.error(f"图像分类失败: {e}")
        return "unknown", 0.0


def create_classification_visualization(image: Image.Image, class_name: str, confidence: float) -> Image.Image:
    """
    创建分类结果的可视化图像
    
    Args:
        image: 原始图像
        class_name: 预测类别
        confidence: 置信度
        
    Returns:
        PIL.Image.Image: 带有分类结果的图像
    """
    try:
        from PIL import ImageDraw, ImageFont
        
        # 创建图像副本
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 准备标签文本
        label = f"分类结果: {class_name}"
        conf_text = f"置信度: {confidence:.2f}"
        
        # 根据类别选择颜色
        if class_name.lower() == "locked":
            color = (0, 255, 0)  # 绿色
        elif class_name.lower() == "unlocked":
            color = (255, 0, 0)  # 红色
        else:
            color = (0, 0, 255)  # 蓝色
        
        # 在图像顶部绘制半透明背景
        from PIL import Image, ImageDraw
        overlay = Image.new('RGBA', result_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # 绘制背景矩形
        text_bbox = draw.textbbox((10, 10), label, font=font)
        conf_bbox = draw.textbbox((10, 40), conf_text, font=font)
        
        # 合并边界框
        combined_bbox = [
            min(text_bbox[0], conf_bbox[0]) - 10,
            min(text_bbox[1], conf_bbox[1]) - 10,
            max(text_bbox[2], conf_bbox[2]) + 10,
            max(text_bbox[3], conf_bbox[3]) + 10
        ]
        
        # 绘制半透明背景
        overlay_draw.rectangle(combined_bbox, fill=(*color, 128))
        
        # 叠加到原图上
        result_image = Image.alpha_composite(
            result_image.convert('RGBA'), 
            overlay
        ).convert('RGB')
        
        # 重新创建绘制对象
        draw = ImageDraw.Draw(result_image)
        
        # 绘制文本
        draw.text((10, 10), label, fill=(255, 255, 255), font=font)
        draw.text((10, 40), conf_text, fill=(255, 255, 255), font=font)
        
        return result_image
        
    except Exception as e:
        logger.error(f"创建分类可视化失败: {e}")
        return image