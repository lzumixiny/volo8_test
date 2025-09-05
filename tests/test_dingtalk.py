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
    assert message.images.download_code[0] == "test_code"


def test_message_parsing_empty():
    """测试空消息解析"""
    message_data = {}
    message = DingTalkMessage.from_dict(message_data)
    assert message.msg_id == ""
    assert message.text.content == ""


def test_mention_detection(callback_handler):
    """测试@检测"""
    # 测试包含@的消息
    message_data = {
        "text": {
            "content": "@机器人 请检测这个锁"
        },
        "atUsers": {
            "dingtalkId": ["test_bot"]
        }
    }
    
    message = DingTalkMessage.from_dict(message_data)
    callback_handler.app_key = "test_bot"
    assert callback_handler.is_mentioned_to_bot(message) == True
    
    # 测试不包含@的消息
    message_data2 = {
        "text": {
            "content": "请检测这个锁"
        },
        "atUsers": {
            "dingtalkId": ["other_bot"]
        }
    }
    
    message2 = DingTalkMessage.from_dict(message_data2)
    assert callback_handler.is_mentioned_to_bot(message2) == False


def test_extract_image_urls():
    """测试图片URL提取"""
    message = DingTalkMessage()
    message.images.image_url = ["http://example.com/img1.jpg", "http://example.com/img2.jpg"]
    
    handler = DingTalkCallbackHandler("test", "test")
    urls = handler.extract_image_urls(message)
    
    assert len(urls) == 2
    assert "http://example.com/img1.jpg" in urls
    assert "http://example.com/img2.jpg" in urls


def test_extract_text_content():
    """测试文本内容提取"""
    message = DingTalkMessage()
    message.text.content = "这是一条测试消息"
    
    handler = DingTalkCallbackHandler("test", "test")
    content = handler.extract_text_content(message)
    
    assert content == "这是一条测试消息"


def test_message_sender_initialization():
    """测试消息发送器初始化"""
    sender = DingTalkMessageSender("test_webhook")
    assert sender.webhook_url == "test_webhook"
    
    sender2 = DingTalkMessageSender()
    assert sender2.webhook_url == ""


def test_at_users_parsing():
    """测试@用户解析"""
    message_data = {
        "atUsers": {
            "dingtalkId": ["user1", "user2"],
            "staffId": ["staff1", "staff2"]
        }
    }
    
    message = DingTalkMessage.from_dict(message_data)
    assert len(message.at_users.dingtalk_id) == 2
    assert len(message.at_users.staff_id) == 2
    assert "user1" in message.at_users.dingtalk_id
    assert "staff1" in message.at_users.staff_id


def test_images_parsing():
    """测试图片信息解析"""
    message_data = {
        "images": {
            "imageUrl": ["img1.jpg", "img2.jpg"],
            "downloadCode": ["code1", "code2"],
            "imageSize": [
                {"width": 800, "height": 600},
                {"width": 1024, "height": 768}
            ]
        }
    }
    
    message = DingTalkMessage.from_dict(message_data)
    assert len(message.images.image_url) == 2
    assert len(message.images.download_code) == 2
    assert len(message.images.image_size) == 2
    assert message.images.image_size[0]["width"] == 800