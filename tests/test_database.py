import pytest
import tempfile
import os
from datetime import datetime
from apps.database import DatabaseManager, DetectionResult


@pytest.fixture
def temp_db():
    """创建临时数据库"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = DatabaseManager(db_path)
    yield db
    
    os.unlink(db_path)


def test_database_initialization(temp_db):
    """测试数据库初始化"""
    # 验证表是否创建
    stats = temp_db.get_statistics()
    assert 'total_detections' in stats
    assert 'unsafe_detections' in stats


def test_save_detection_result(temp_db):
    """测试保存检测结果"""
    result = DetectionResult(
        image_url="test.jpg",
        image_hash="abc123",
        detection_time=datetime.now(),
        locks_detected=3,
        unlocked_locks=1,
        confidence_score=0.95,
        is_safe=False
    )
    
    result_id = temp_db.save_detection_result(result)
    assert result_id > 0
    
    # 验证数据是否正确保存
    saved_result = temp_db.get_detection_by_id(result_id)
    assert saved_result['locks_detected'] == 3
    assert saved_result['is_safe'] == False


def test_get_detection_history(temp_db):
    """测试获取检测历史"""
    # 创建测试数据
    for i in range(5):
        result = DetectionResult(
            image_url=f"test{i}.jpg",
            image_hash=f"hash{i}",
            detection_time=datetime.now(),
            locks_detected=i+1,
            unlocked_locks=i%2,
            confidence_score=0.9,
            is_safe=True
        )
        temp_db.save_detection_result(result)
    
    history = temp_db.get_detection_history(limit=3)
    assert len(history) == 3
    assert history[0]['locks_detected'] == 5  # 最新的记录


def test_save_lock_details(temp_db):
    """测试保存锁详细信息"""
    # 先保存检测结果
    result = DetectionResult(
        image_url="test.jpg",
        image_hash="abc123",
        detection_time=datetime.now(),
        locks_detected=2,
        unlocked_locks=1,
        confidence_score=0.95,
        is_safe=False
    )
    
    result_id = temp_db.save_detection_result(result)
    
    # 保存锁详细信息
    lock_details = [
        {
            'lock_type': 'padlock',
            'is_locked': True,
            'confidence': 0.9,
            'position_x': 100,
            'position_y': 100,
            'width': 50,
            'height': 50
        },
        {
            'lock_type': 'door_lock',
            'is_locked': False,
            'confidence': 0.95,
            'position_x': 200,
            'position_y': 200,
            'width': 60,
            'height': 60
        }
    ]
    
    temp_db.save_lock_details(result_id, lock_details)
    
    # 验证统计数据
    stats = temp_db.get_statistics()
    assert stats['total_locks'] == 2
    assert stats['total_unlocked'] == 1


def test_training_records(temp_db):
    """测试训练记录"""
    record = {
        'model_name': 'test_model',
        'training_start': datetime.now(),
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'dataset_size': 1000,
        'status': 'completed'
    }
    
    record_id = temp_db.save_training_record(record)
    assert record_id > 0
    
    # 更新训练记录
    updates = {
        'training_end': datetime.now(),
        'train_loss': 0.5,
        'val_loss': 0.6,
        'map_score': 0.85,
        'model_path': '/models/test_model.pt'
    }
    
    temp_db.update_training_record(record_id, updates)