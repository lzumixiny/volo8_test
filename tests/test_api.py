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


def test_health_check(client):
    """测试健康检查接口"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True
    assert 'data' in data
    assert 'database' in data['data']
    assert 'model_loaded' in data['data']


def test_lock_detection(client, test_image_file):
    """测试锁检测接口"""
    response = client.post("/api/v1/lock/detect", files={"file": test_image_file})
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True
    assert 'detection_id' in data['data']
    assert 'result' in data['data']
    assert 'image_base64' in data['data']


def test_lock_detection_with_user_id(client, test_image_file):
    """测试带用户ID的锁检测"""
    response = client.post("/api/v1/lock/detect", 
                          files={"file": test_image_file},
                          data={"user_id": "test_user"})
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True


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
    assert data['message'] == "钉钉配置成功"


def test_dingtalk_configuration_invalid(client):
    """测试无效的钉钉配置"""
    config_data = {
        "app_key": "test_key"
        # 缺少必需的 app_secret
    }
    
    response = client.post("/api/v1/dingtalk/configure", json=config_data)
    assert response.status_code == 422  # 验证错误


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
    assert isinstance(data['data']['history'], list)


def test_history_api_with_params(client):
    """测试带参数的检测历史接口"""
    response = client.get("/api/v1/history?limit=5&offset=10")
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True


def test_dingtalk_webhook_without_config(client):
    """测试未配置钉钉的webhook调用"""
    callback_data = {
        "chatbotUserId": "test_bot",
        "conversationType": "1",
        "msgId": "test_msg",
        "text": {
            "content": "@机器人 测试"
        }
    }
    
    response = client.post("/api/v1/dingtalk/webhook", json=callback_data)
    assert response.status_code == 400
    
    data = response.json()
    assert data['success'] == False
    assert "钉钉服务未配置" in data['message']


def test_training_api(client):
    """测试训练启动接口"""
    training_config = {
        "model_name": "test_model",
        "epochs": 10,
        "batch_size": 8,
        "img_size": 640,
        "learning_rate": 0.001,
        "device": "auto"
    }
    
    response = client.post("/api/v1/train/start", json=training_config)
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True
    assert data['data']['model_name'] == "test_model"


def test_training_api_default_config(client):
    """测试默认训练配置"""
    training_config = {}  # 使用默认配置
    
    response = client.post("/api/v1/train/start", json=training_config)
    assert response.status_code == 200
    
    data = response.json()
    assert data['success'] == True
    assert data['data']['model_name'] == "lock_detector"  # 默认值


def test_legacy_interfaces_still_work(client, test_image_file):
    """测试旧接口仍然可用"""
    # 测试旧的JSON接口
    response = client.post("/img_object_detection_to_json", files={"file": test_image_file})
    assert response.status_code == 200
    
    # 测试旧的图片接口
    response = client.post("/img_object_detection_to_img", files={"file": test_image_file})
    assert response.status_code == 200


def test_redirect_to_docs(client):
    """测试重定向到文档"""
    response = client.get("/")
    assert response.status_code == 307  # 临时重定向


def test_healthcheck_endpoint(client):
    """测试健康检查端点"""
    response = client.get("/healthcheck")
    assert response.status_code == 200
    
    data = response.json()
    assert "healthcheck" in data
    assert data["healthcheck"] == "一切正常！"