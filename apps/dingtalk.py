import json
import hashlib
import hmac
import base64
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
from fastapi import HTTPException, Request
from pydantic import BaseModel
from PIL import Image
import io


class DingTalkMessage:
    """钉钉消息模型"""
    
    def __init__(self):
        self.chatbot_user_id = ""
        self.conversation_type = ""
        self.msg_id = ""
        self.create_at = ""
        self.conversation_title = ""
        self.sender_id = ""
        self.sender_nick = ""
        self.session_webhook = ""
        self.session_webhook_expired_time = 0
        self.content = Content()
        self.images = Images()
        self.msg_type = ""
        self.at_users = AtUsers()
        self.text = Text()
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DingTalkMessage':
        """从字典创建消息对象"""
        msg = cls()
        msg.chatbot_user_id = data.get('chatbotUserId', '')
        msg.conversation_type = data.get('conversationType', '')
        msg.msg_id = data.get('msgId', '')
        msg.create_at = data.get('createAt', '')
        msg.conversation_title = data.get('conversationTitle', '')
        msg.sender_id = data.get('senderId', '')
        msg.sender_nick = data.get('senderNick', '')
        msg.session_webhook = data.get('sessionWebhook', '')
        msg.session_webhook_expired_time = data.get('sessionWebhookExpiredTime', 0)
        msg.msg_type = data.get('msgType', '')
        
        if 'content' in data:
            msg.content = Content.from_dict(data['content'])
        if 'images' in data:
            msg.images = Images.from_dict(data['images'])
        if 'atUsers' in data:
            msg.at_users = AtUsers.from_dict(data['atUsers'])
        if 'text' in data:
            msg.text = Text.from_dict(data['text'])
        
        return msg


class Content(BaseModel):
    """消息内容"""
    content: str = ""


class Images(BaseModel):
    """图片信息"""
    download_code: List[str] = []
    image_url: List[str] = []
    image_size: List[Dict] = []
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Images':
        images = cls()
        images.download_code = data.get('downloadCode', [])
        images.image_url = data.get('imageUrl', [])
        images.image_size = data.get('imageSize', [])
        return images


class AtUsers(BaseModel):
    """@用户信息"""
    dingtalk_id: List[str] = []
    staff_id: List[str] = []
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AtUsers':
        at_users = cls()
        at_users.dingtalk_id = data.get('dingtalkId', [])
        at_users.staff_id = data.get('staffId', [])
        return at_users


class Text(BaseModel):
    """文本内容"""
    content: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Text':
        text = cls()
        text.content = data.get('content', '')
        return text


class DingTalkCallbackHandler:
    """钉钉回调处理器"""
    
    def __init__(self, app_key: str, app_secret: str):
        self.app_key = app_key
        self.app_secret = app_secret
        self.access_token = ""
        self.expires_in = 0
        
    def verify_signature(self, request: Request, timestamp: str, sign: str) -> bool:
        """验证钉钉回调签名"""
        try:
            # 获取请求体
            body = request.body()
            if isinstance(body, str):
                body = body.encode('utf-8')
            
            # 计算签名
            string_to_sign = f"{timestamp}\n{sign}".encode('utf-8')
            hmac_code = hmac.new(
                self.app_secret.encode('utf-8'),
                string_to_sign,
                digestmod=hashlib.sha256
            ).digest()
            
            # Base64编码
            signature = base64.b64encode(hmac_code).decode('utf-8')
            
            return signature == sign
            
        except Exception as e:
            logger.error(f"签名验证失败: {e}")
            return False
    
    def parse_callback_message(self, data: Dict) -> DingTalkMessage:
        """解析回调消息"""
        try:
            return DingTalkMessage.from_dict(data)
        except Exception as e:
            logger.error(f"解析回调消息失败: {e}")
            raise HTTPException(status_code=400, detail="消息格式错误")
    
    def is_mentioned_to_bot(self, message: DingTalkMessage) -> bool:
        """检查消息是否@了机器人"""
        # 检查文本中是否包含@机器人
        if message.text and message.text.content:
            text_content = message.text.content.lower()
            # 检查是否包含@或者机器人关键词
            if '@' in text_content or '机器人' in text_content:
                return True
        
        # 检查atUsers中是否包含机器人
        if message.at_users:
            if self.app_key in message.at_users.dingtalk_id:
                return True
        
        return False
    
    def extract_image_urls(self, message: DingTalkMessage) -> List[str]:
        """提取图片URL"""
        image_urls = []
        
        if message.images and message.images.image_url:
            image_urls.extend(message.images.image_url)
        
        return image_urls
    
    def extract_text_content(self, message: DingTalkMessage) -> str:
        """提取文本内容"""
        if message.text:
            return message.text.content
        return ""
    
    async def download_image(self, image_url: str, download_code: str = "") -> Optional[Image.Image]:
        """下载图片"""
        try:
            async with aiohttp.ClientSession() as session:
                # 构建下载URL
                if download_code:
                    download_url = f"{image_url}?downloadCode={download_code}"
                else:
                    download_url = image_url
                
                async with session.get(download_url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        image = Image.open(io.BytesIO(image_data)).convert('RGB')
                        return image
                    else:
                        logger.error(f"下载图片失败: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"下载图片失败: {e}")
            return None


class DingTalkMessageSender:
    """钉钉消息发送器"""
    
    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url
    
    async def send_text_message(self, content: str, at_mobiles: List[str] = None) -> bool:
        """发送文本消息"""
        try:
            message = {
                "msgtype": "text",
                "text": {
                    "content": content
                }
            }
            
            if at_mobiles:
                message["at"] = {
                    "atMobiles": at_mobiles,
                    "isAtAll": False
                }
            
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"发送文本消息失败: {e}")
            return False
    
    async def send_image_message(self, image_url: str, content: str = "") -> bool:
        """发送图片消息"""
        try:
            message = {
                "msgtype": "image",
                "image": {
                    "url": image_url
                }
            }
            
            if content:
                message["text"] = {
                    "content": content
                }
            
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"发送图片消息失败: {e}")
            return False
    
    async def send_markdown_message(self, title: str, content: str) -> bool:
        """发送Markdown消息"""
        try:
            message = {
                "msgtype": "markdown",
                "markdown": {
                    "title": title,
                    "text": content
                }
            }
            
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"发送Markdown消息失败: {e}")
            return False
    
    async def send_detection_result(self, 
                                  result_text: str, 
                                  image_url: str = "", 
                                  at_user: str = "") -> bool:
        """发送检测结果"""
        try:
            # 构建Markdown格式的消息
            markdown_content = f"""
## 🔒 锁检测结果

{result_text}

---
检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            if image_url:
                markdown_content += f"\n![检测结果]({image_url})"
            
            message = {
                "msgtype": "markdown",
                "markdown": {
                    "title": "锁检测结果",
                    "text": markdown_content
                }
            }
            
            # 如果需要@用户
            if at_user:
                message["at"] = {
                    "atMobiles": [at_user],
                    "isAtAll": False
                }
            
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"发送检测结果失败: {e}")
            return False
    
    async def _send_message(self, message: Dict) -> bool:
        """发送消息到钉钉"""
        try:
            if not self.webhook_url:
                logger.error("Webhook URL 未设置")
                return False
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=message,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('errcode') == 0:
                            logger.info("消息发送成功")
                            return True
                        else:
                            logger.error(f"消息发送失败: {result}")
                            return False
                    else:
                        logger.error(f"HTTP请求失败: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False


# 全局实例（需要在实际使用时配置）
dingtalk_handler = DingTalkCallbackHandler("", "")
dingtalk_sender = DingTalkMessageSender()