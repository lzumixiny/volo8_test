# 智能锁检测系统 - 测试文档

## 1. 测试概述

### 1.1 测试目标
- 验证系统功能的正确性
- 确保API接口的稳定性
- 检测模型准确性
- 验证钉钉集成功能
- 性能和压力测试

### 1.2 测试环境
- **操作系统**: Linux/Windows/macOS
- **Python版本**: 3.13+
- **依赖包**: 见requirements.txt
- **数据库**: SQLite
- **测试工具**: pytest, requests

### 1.3 测试范围
- 单元测试
- 集成测试
- API测试
- 钉钉回调测试
- 性能测试

## 2. 测试环境搭建

### 2.1 环境准备
```bash
# 克隆项目
git clone <repository_url>
cd lock_det

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install pytest pytest-asyncio requests httpx

# 初始化数据库
python -c "from apps.database import db_manager; db_manager.init_database()"
```

### 2.2 配置文件
创建 `.env` 文件：
```bash
# 测试环境配置
DINGTALK_APP_KEY=test_app_key
DINGTALK_APP_SECRET=test_app_secret
DINGTALK_WEBHOOK_URL=test_webhook_url
```

### 2.3 测试数据准备
```bash
# 创建测试数据目录
mkdir -p test_data/images
mkdir -p test_data/labels

# 下载测试图片（示例）
wget -O test_data/images/test_lock.jpg https://example.com/test_lock.jpg
```

## 3. 单元测试

### 3.1 数据库测试

#### 3.1.1 测试文件: `tests/test_database.py`
```python
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
```

### 3.2 锁检测测试

#### 3.2.1 测试文件: `tests/test_lock_detector.py`
```python
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
    
    result_image = detector.visualize_detection(test_image, result)
    assert isinstance(result_image, Image.Image)
```

### 3.3 钉钉服务测试

#### 3.3.1 测试文件: `tests/test_dingtalk.py`
```python
import pytest
import json
from apps.dingtalk import DingTalkCallbackHandler, DingTalkMessage, DingTalkMessageSender

@pytest.fixture
def callback_handler():
    """创建回调处理器"""
    return DingTalkCallbackHandler("test_key", "test_secret")

@pytest.fixture
def message_sender():
    """创建消息发送器"""
    return DingTalkMessageSender("test_webhook")

def test_message_parsing():
    """测试消息解析"""
    message_data = {
        "chatbotUserId": "test_bot",
        "conversationType": "1",
        "msgId": "test_msg",
        "text": {
            "content": "@机器人 测试消息"
        },
        "images": {
            "imageUrl": ["http://example.com/test.jpg"],
            "downloadCode": ["test_code"]
        }
    }
    
    message = DingTalkMessage.from_dict(message_data)
    assert message.msg_id == "test_msg"
    assert message.text.content == "@机器人 测试消息"
    assert len(message.images.image_url) == 1

def test_signature_verification(callback_handler):
    """测试签名验证"""
    # 这里需要模拟签名验证逻辑
    # 实际测试中需要使用真实的签名算法
    pass

def test_mention_detection(callback_handler):
    """测试@检测"""
    message_data = {
        "text": {
            "content": "@机器人 请检测这个锁"
        },
        "atUsers": {
            "dingtalkId": ["test_bot"]
        }
    }
    
    message = DingTalkMessage.from_dict(message_data)
    assert callback_handler.is_mentioned_to_bot(message) == True
```

## 4. API测试

### 4.1 测试文件: `tests/test_api.py`
```python
import pytest
import asyncio
from fastapi.testclient import TestClient
from PIL import Image
import io
from main import app

@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)

@pytest.fixture
def test_image_file():
    """创建测试图片文件"""
    # 创建测试图片
    img = Image.new('RGB', (640, 640), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return ('test.jpg', img_bytes, 'image/jpeg')

def test_health_check(client):
    """测试健康检查接口"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True
    assert 'data' in data

def test_lock_detection(client, test_image_file):
    """测试锁检测接口"""
    response = client.post("/api/v1/lock/detect", files={"file": test_image_file})
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True
    assert 'detection_id' in data['data']
    assert 'result' in data['data']

def test_dingtalk_configuration(client):
    """测试钉钉配置接口"""
    config_data = {
        "app_key": "test_key",
        "app_secret": "test_secret",
        "webhook_url": "test_webhook"
    }
    
    response = client.post("/api/v1/dingtalk/configure", json=config_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True

def test_statistics_api(client):
    """测试统计信息接口"""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True
    assert 'detection_stats' in data['data']
    assert 'dataset_stats' in data['data']

def test_history_api(client):
    """测试检测历史接口"""
    response = client.get("/api/v1/history")
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True
    assert 'history' in data['data']

def test_training_api(client):
    """测试训练启动接口"""
    training_config = {
        "model_name": "test_model",
        "epochs": 10,
        "batch_size": 8
    }
    
    response = client.post("/api/v1/train/start", json=training_config)
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True
```

## 5. 集成测试

### 5.1 测试文件: `tests/test_integration.py`
```python
import pytest
import asyncio
from PIL import Image
import io
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)

def test_complete_workflow(client):
    """测试完整工作流程"""
    # 1. 配置钉钉
    config_data = {
        "app_key": "test_key",
        "app_secret": "test_secret",
        "webhook_url": "test_webhook"
    }
    client.post("/api/v1/dingtalk/configure", json=config_data)
    
    # 2. 创建测试图片
    img = Image.new('RGB', (640, 640), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # 3. 进行检测
    response = client.post("/api/v1/lock/detect", files={"file": ('test.jpg', img_bytes, 'image/jpeg')})
    assert response.status_code == 200
    
    detection_data = response.json()
    detection_id = detection_data['data']['detection_id']
    
    # 4. 检查历史记录
    response = client.get("/api/v1/history")
    history_data = response.json()
    assert len(history_data['data']['history']) > 0
    
    # 5. 检查统计信息
    response = client.get("/api/v1/stats")
    stats_data = response.json()
    assert stats_data['data']['detection_stats']['total_detections'] > 0

def test_error_handling(client):
    """测试错误处理"""
    # 测试无效文件
    response = client.post("/api/v1/lock/detect", files={"file": ('test.txt', b'invalid', 'text/plain')})
    assert response.status_code == 200  # 应该返回错误信息但状态码为200
    
    # 测试空文件
    response = client.post("/api/v1/lock/detect", files={"file": ('test.jpg', b'', 'image/jpeg')})
    assert response.status_code == 200
    
    # 测试无效配置
    response = client.post("/api/v1/dingtalk/configure", json={"invalid": "data"})
    assert response.status_code == 422  # 验证错误
```

## 6. 性能测试

### 6.1 测试文件: `tests/test_performance.py`
```python
import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)

@pytest.fixture
def test_image_file():
    """创建测试图片文件"""
    img = Image.new('RGB', (640, 640), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return ('test.jpg', img_bytes, 'image/jpeg')

def test_response_time(client, test_image_file):
    """测试响应时间"""
    start_time = time.time()
    
    for _ in range(10):
        response = client.post("/api/v1/lock/detect", files={"file": test_image_file})
        assert response.status_code == 200
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    
    print(f"平均响应时间: {avg_time:.2f}秒")
    assert avg_time < 5.0  # 平均响应时间应小于5秒

def test_concurrent_requests(client, test_image_file):
    """测试并发请求"""
    def make_request():
        response = client.post("/api/v1/lock/detect", files={"file": test_image_file})
        return response.status_code == 200
    
    # 并发执行10个请求
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [future.result() for future in futures]
    
    assert all(results)  # 所有请求都应该成功

def test_memory_usage():
    """测试内存使用"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 模拟多次检测
    for _ in range(50):
        img = Image.new('RGB', (640, 640), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"内存增长: {memory_increase:.2f}MB")
    assert memory_increase < 100  # 内存增长应小于100MB
```

## 7. 钉钉回调测试

### 7.1 测试文件: `tests/test_dingtalk_callback.py`
```python
import pytest
import json
import hashlib
import base64
import hmac
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)

def generate_signature(timestamp, secret):
    """生成钉钉签名"""
    string_to_sign = f"{timestamp}\n{secret}".encode('utf-8')
    hmac_code = hmac.new(
        secret.encode('utf-8'),
        string_to_sign,
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(hmac_code).decode('utf-8')

def test_dingtalk_webhook_callback(client):
    """测试钉钉回调"""
    # 准备测试数据
    timestamp = str(int(time.time()))
    secret = "test_secret"
    sign = generate_signature(timestamp, secret)
    
    callback_data = {
        "chatbotUserId": "test_bot",
        "conversationType": "1",
        "msgId": "test_msg",
        "createAt": timestamp,
        "conversationTitle": "测试群",
        "senderId": "test_user",
        "senderNick": "测试用户",
        "text": {
            "content": "@机器人 请检测这个锁"
        },
        "images": {
            "imageUrl": ["http://example.com/test.jpg"],
            "downloadCode": ["test_code"]
        }
    }
    
    # 配置钉钉
    config_data = {
        "app_key": "test_key",
        "app_secret": secret,
        "webhook_url": "test_webhook"
    }
    client.post("/api/v1/dingtalk/configure", json=config_data)
    
    # 发送回调
    headers = {
        "timestamp": timestamp,
        "sign": sign
    }
    
    response = client.post("/api/v1/dingtalk/webhook", 
                          json=callback_data, 
                          headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data['success'] == True

def test_dingtalk_webhook_invalid_signature(client):
    """测试无效签名"""
    timestamp = str(int(time.time()))
    
    callback_data = {
        "chatbotUserId": "test_bot",
        "conversationType": "1",
        "msgId": "test_msg",
        "text": {
            "content": "@机器人 测试"
        }
    }
    
    headers = {
        "timestamp": timestamp,
        "sign": "invalid_signature"
    }
    
    response = client.post("/api/v1/dingtalk/webhook", 
                          json=callback_data, 
                          headers=headers)
    
    assert response.status_code == 401
```

## 8. 测试运行

### 8.1 运行所有测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_database.py

# 运行特定测试函数
pytest tests/test_database.py::test_save_detection_result

# 生成覆盖率报告
pytest --cov=apps tests/

# 生成HTML覆盖率报告
pytest --cov=apps --cov-report=html tests/
```

### 8.2 测试配置
创建 `pytest.ini` 文件：
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

## 9. 测试报告

### 9.1 测试结果分析
运行测试后，分析以下指标：
- **测试覆盖率**: 目标 > 80%
- **通过率**: 目标 100%
- **性能指标**: 响应时间、内存使用
- **错误处理**: 各种异常情况的处理

### 9.2 持续集成
在 CI/CD 流程中集成测试：
```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.13
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    - name: Run tests
      run: |
        pytest --cov=apps --cov-report=xml tests/
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 10. 测试注意事项

### 10.1 测试数据管理
- 使用临时数据库进行测试
- 测试完成后清理测试数据
- 使用模拟对象替代外部依赖

### 10.2 异步测试
- 使用 `pytest-asyncio` 处理异步测试
- 确保异步操作的正确清理

### 10.3 性能测试
- 在隔离环境中进行性能测试
- 避免性能测试影响正常开发
- 监控系统资源使用情况

### 10.4 安全测试
- 不在测试代码中硬编码敏感信息
- 使用测试专用的配置和密钥
- 测试完成后清理测试数据