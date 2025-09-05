import pytest
from PIL import Image
import numpy as np
from apps.lock_detector import LockDetector, LockDetectionResult


@pytest.fixture
def detector():
    """创建检测器实例"""
    return LockDetector()


@pytest.fixture
def test_image():
    """创建测试图片"""
    # 创建一个简单的测试图片
    img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_detector_initialization(detector):
    """测试检测器初始化"""
    assert detector.model is not None
    assert len(detector.lock_classes) > 0


def test_detection_result_class():
    """测试检测结果类"""
    result = LockDetectionResult()
    assert result.is_safe == True
    assert result.total_locks == 0
    assert result.unlocked_locks == 0
    
    # 添加一个锁
    result.add_lock("padlock", False, 0.95, {"xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200})
    assert result.total_locks == 1
    assert result.unlocked_locks == 1
    assert result.is_safe == False
    
    # 添加一个锁定的锁
    result.add_lock("locked_lock", True, 0.9, {"xmin": 300, "ymin": 300, "xmax": 400, "ymax": 400})
    assert result.total_locks == 2
    assert result.locked_locks == 1
    assert result.is_safe == False  # 仍然有未锁的锁


def test_detection_result_to_dict():
    """测试检测结果转换为字典"""
    result = LockDetectionResult()
    result.add_lock("padlock", False, 0.95, {"xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200})
    
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert result_dict['is_safe'] == False
    assert result_dict['total_locks'] == 1
    assert result_dict['unlocked_locks'] == 1
    assert len(result_dict['lock_details']) == 1


def test_detect_locks(detector, test_image):
    """测试锁检测功能"""
    result = detector.detect_locks(test_image)
    
    # 验证结果结构
    assert hasattr(result, 'is_safe')
    assert hasattr(result, 'total_locks')
    assert hasattr(result, 'unlocked_locks')
    assert hasattr(result, 'lock_details')
    assert isinstance(result.to_dict(), dict)


def test_visualize_detection(detector, test_image):
    """测试检测结果可视化"""
    result = LockDetectionResult()
    result.add_lock("padlock", False, 0.95, {"xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200})
    result.add_lock("locked_lock", True, 0.9, {"xmin": 300, "ymin": 300, "xmax": 400, "ymax": 400})
    
    result_image = detector.visualize_detection(test_image, result)
    assert isinstance(result_image, Image.Image)
    assert result_image.size == test_image.size


def test_save_detection_result(detector, test_image):
    """测试保存检测结果"""
    result = LockDetectionResult()
    result.add_lock("padlock", False, 0.95, {"xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200})
    
    detection_id = detector.save_detection_result(
        test_image, 
        result,
        message_id="test_msg",
        user_id="test_user",
        group_id="test_group"
    )
    
    assert detection_id > 0


def test_calculate_image_hash(detector, test_image):
    """测试图片哈希计算"""
    hash1 = detector._calculate_image_hash(test_image)
    hash2 = detector._calculate_image_hash(test_image)
    
    assert hash1 == hash2  # 同一张图片的哈希应该相同
    assert len(hash1) == 32  # MD5哈希长度


def test_get_detection_statistics(detector):
    """测试获取检测统计信息"""
    stats = detector.get_detection_statistics()
    assert isinstance(stats, dict)
    assert 'total_detections' in stats
    assert 'unsafe_detections' in stats
    assert 'total_locks' in stats
    assert 'total_unlocked' in stats


def test_get_detection_history(detector):
    """测试获取检测历史"""
    history = detector.get_detection_history(limit=5)
    assert isinstance(history, list)
    assert len(history) <= 5