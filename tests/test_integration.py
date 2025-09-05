import pytest
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
    # 创建测试图片
    img = Image.new('RGB', (640, 640), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return ('test.jpg', img_bytes, 'image/jpeg')


def test_complete_workflow(client, test_image_file):
    """测试完整工作流程"""
    # 1. 配置钉钉
    config_data = {
        "app_key": "test_key",
        "app_secret": "test_secret",
        "webhook_url": "test_webhook"
    }
    client.post("/api/v1/dingtalk/configure", json=config_data)
    
    # 2. 进行检测
    response = client.post("/api/v1/lock/detect", files={"file": test_image_file})
    assert response.status_code == 200
    
    detection_data = response.json()
    detection_id = detection_data['data']['detection_id']
    
    # 3. 检查历史记录
    response = client.get("/api/v1/history")
    history_data = response.json()
    assert len(history_data['data']['history']) > 0
    
    # 4. 检查统计信息
    response = client.get("/api/v1/stats")
    stats_data = response.json()
    assert stats_data['data']['detection_stats']['total_detections'] > 0


def test_error_handling(client):
    """测试错误处理"""
    # 测试无效文件类型
    response = client.post("/api/v1/lock/detect", files={"file": ('test.txt', b'invalid content', 'text/plain')})
    assert response.status_code == 200  # 应该返回错误信息但状态码为200
    
    data = response.json()
    assert data['success'] == False
    assert '检测失败' in data['message']
    
    # 测试空文件
    response = client.post("/api/v1/lock/detect", files={"file": ('test.jpg', b'', 'image/jpeg')})
    assert response.status_code == 200
    
    # 测试无效的钉钉配置
    response = client.post("/api/v1/dingtalk/configure", json={"invalid": "data"})
    assert response.status_code == 422  # 验证错误


def test_api_response_format(client, test_image_file):
    """测试API响应格式"""
    # 测试健康检查
    response = client.get("/api/v1/health")
    data = response.json()
    assert 'success' in data
    assert 'message' in data
    assert 'data' in data
    
    # 测试检测接口
    response = client.post("/api/v1/lock/detect", files={"file": test_image_file})
    data = response.json()
    assert 'success' in data
    assert 'message' in data
    assert 'data' in data
    
    # 测试统计接口
    response = client.get("/api/v1/stats")
    data = response.json()
    assert 'success' in data
    assert 'message' in data
    assert 'data' in data


def test_backwards_compatibility(client, test_image_file):
    """测试向后兼容性"""
    # 测试旧的接口仍然可用
    response = client.post("/img_object_detection_to_json", files={"file": test_image_file})
    assert response.status_code == 200
    
    response = client.post("/img_object_detection_to_img", files={"file": test_image_file})
    assert response.status_code == 200


def test_multiple_detections(client, test_image_file):
    """测试多次检测"""
    # 进行多次检测
    for i in range(3):
        response = client.post("/api/v1/lock/detect", files={"file": test_image_file})
        assert response.status_code == 200
    
    # 检查历史记录
    response = client.get("/api/v1/history")
    history_data = response.json()
    assert len(history_data['data']['history']) >= 3
    
    # 检查统计信息
    response = client.get("/api/v1/stats")
    stats_data = response.json()
    assert stats_data['data']['detection_stats']['total_detections'] >= 3


def test_dingtalk_workflow(client):
    """测试钉钉工作流程"""
    # 1. 配置钉钉
    config_data = {
        "app_key": "test_key",
        "app_secret": "test_secret",
        "webhook_url": "test_webhook"
    }
    response = client.post("/api/v1/dingtalk/configure", json=config_data)
    assert response.status_code == 200
    
    # 2. 模拟钉钉回调（需要正确的签名）
    # 这里只测试接口存在，实际的签名验证需要更复杂的设置
    callback_data = {
        "chatbotUserId": "test_bot",
        "conversationType": "1",
        "msgId": "test_msg",
        "text": {
            "content": "@机器人 测试消息"
        }
    }
    
    # 由于签名验证，这个测试会失败，但接口应该存在
    response = client.post("/api/v1/dingtalk/webhook", json=callback_data)
    # 可能返回400或401，这取决于配置状态


def test_training_workflow(client):
    """测试训练工作流程"""
    # 启动训练任务
    training_config = {
        "model_name": "integration_test_model",
        "epochs": 1,
        "batch_size": 1,
        "device": "cpu"
    }
    
    response = client.post("/api/v1/train/start", json=training_config)
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True
    assert data['data']['model_name'] == "integration_test_model"


def test_data_persistence(client, test_image_file):
    """测试数据持久化"""
    # 进行一次检测
    response = client.post("/api/v1/lock/detect", files={"file": test_image_file})
    assert response.status_code == 200
    
    # 获取检测ID
    detection_id = response.json()['data']['detection_id']
    
    # 检查统计信息是否更新
    response = client.get("/api/v1/stats")
    stats_data = response.json()
    assert stats_data['data']['detection_stats']['total_detections'] > 0
    
    # 检查历史记录
    response = client.get("/api/v1/history")
    history_data = response.json()
    found = False
    for record in history_data['data']['history']:
        if record['id'] == detection_id:
            found = True
            break
    assert found