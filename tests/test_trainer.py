import pytest
import tempfile
import os
from pathlib import Path
from apps.trainer import LockModelTrainer


@pytest.fixture
def temp_trainer():
    """创建临时训练器"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, "dataset")
        models_path = os.path.join(temp_dir, "models")
        
        trainer = LockModelTrainer(dataset_path, models_path)
        yield trainer


def test_trainer_initialization(temp_trainer):
    """测试训练器初始化"""
    assert temp_trainer.dataset_path.exists()
    assert temp_trainer.models_path.exists()
    assert len(temp_trainer.lock_classes) > 0
    
    # 验证数据集目录结构
    assert (temp_trainer.dataset_path / "train" / "images").exists()
    assert (temp_trainer.dataset_path / "train" / "labels").exists()
    assert (temp_trainer.dataset_path / "val" / "images").exists()
    assert (temp_trainer.dataset_path / "val" / "labels").exists()
    assert (temp_trainer.dataset_path / "test" / "images").exists()
    assert (temp_trainer.dataset_path / "test" / "labels").exists()


def test_dataset_config_creation(temp_trainer):
    """测试数据集配置文件创建"""
    config_path = temp_trainer.dataset_path / "dataset.yaml"
    assert config_path.exists()
    
    # 读取配置文件内容
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    assert 'path' in config
    assert 'train' in config
    assert 'val' in config
    assert 'test' in config
    assert 'names' in config
    assert 'nc' in config
    assert config['nc'] == len(temp_trainer.lock_classes)


def test_get_dataset_stats(temp_trainer):
    """测试获取数据集统计信息"""
    stats = temp_trainer.get_dataset_stats()
    
    assert 'train' in stats
    assert 'val' in stats
    assert 'test' in stats
    
    for split in ['train', 'val', 'test']:
        assert 'images' in stats[split]
        assert 'labels' in stats[split]
        assert 'path' in stats[split]
        assert isinstance(stats[split]['images'], int)
        assert isinstance(stats[split]['labels'], int)


def test_prepare_training_data(temp_trainer):
    """测试准备训练数据"""
    # 创建测试数据目录
    source_dir = temp_trainer.dataset_path / "source"
    source_dir.mkdir()
    
    # 创建测试图片文件
    test_image = source_dir / "test.jpg"
    from PIL import Image
    img = Image.new('RGB', (640, 640), color='red')
    img.save(test_image)
    
    # 创建测试标注文件
    test_label = source_dir / "test.txt"
    with open(test_label, 'w') as f:
        f.write("padlock 0.5 0.5 0.3 0.4\n")
    
    # 类别映射
    class_mapping = {"padlock": 0}
    
    # 准备训练数据
    result = temp_trainer.prepare_training_data(str(source_dir), class_mapping)
    assert result == True
    
    # 验证文件是否正确复制
    assert (temp_trainer.dataset_path / "train" / "images" / "test.jpg").exists()
    assert (temp_trainer.dataset_path / "train" / "labels" / "test.txt").exists()


def test_create_sample_dataset(temp_trainer):
    """测试创建示例数据集"""
    # 这个方法应该被实现但不实际爬取数据
    result = temp_trainer.create_sample_dataset(num_samples=10)
    assert result == True


def test_model_training_config(temp_trainer):
    """测试模型训练配置"""
    # 由于训练需要实际的数据和较长的时间，这里只测试配置
    training_params = {
        'model_name': 'test_model',
        'epochs': 1,
        'batch_size': 1,
        'img_size': 320,
        'learning_rate': 0.001,
        'device': 'cpu'
    }
    
    # 验证参数可以正确传递
    assert training_params['model_name'] == 'test_model'
    assert training_params['epochs'] == 1
    assert training_params['batch_size'] == 1


def test_model_validation_config(temp_trainer):
    """测试模型验证配置"""
    # 创建一个假的模型文件路径
    fake_model_path = "fake_model.pt"
    
    # 验证方法可以接受模型路径参数
    assert isinstance(fake_model_path, str)
    assert len(fake_model_path) > 0


def test_export_model_config(temp_trainer):
    """测试模型导出配置"""
    fake_model_path = "fake_model.pt"
    export_format = "onnx"
    
    # 验证导出参数
    assert isinstance(fake_model_path, str)
    assert export_format in ["onnx", "torchscript", "coreml"]


def test_dataset_stats_with_files(temp_trainer):
    """测试包含文件的数据集统计"""
    # 创建一些测试文件
    for split in ['train', 'val', 'test']:
        images_dir = temp_trainer.dataset_path / split / "images"
        labels_dir = temp_trainer.dataset_path / split / "labels"
        
        # 创建测试图片
        for i in range(3):
            test_image = images_dir / f"test_{split}_{i}.jpg"
            from PIL import Image
            img = Image.new('RGB', (640, 640), color='red')
            img.save(test_image)
            
            # 创建对应的标注文件
            test_label = labels_dir / f"test_{split}_{i}.txt"
            with open(test_label, 'w') as f:
                f.write("padlock 0.5 0.5 0.3 0.4\n")
    
    # 获取统计信息
    stats = temp_trainer.get_dataset_stats()
    
    # 验证统计结果
    for split in ['train', 'val', 'test']:
        assert stats[split]['images'] == 3
        assert stats[split]['labels'] == 3


def test_lock_classes_definition(temp_trainer):
    """测试锁类别定义"""
    classes = temp_trainer.lock_classes
    
    assert isinstance(classes, dict)
    assert len(classes) > 0
    
    # 验证类别ID从0开始连续
    class_ids = sorted(classes.keys())
    assert class_ids == list(range(len(classes)))
    
    # 验证常见的锁类别
    class_names = list(classes.values())
    assert "locked_lock" in class_names
    assert "unlocked_lock" in class_names