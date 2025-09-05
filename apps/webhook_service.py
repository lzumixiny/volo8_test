import asyncio
import json
import base64
import io
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from fastapi import HTTPException, Request
from PIL import Image

from .dingtalk import DingTalkCallbackHandler, DingTalkMessageSender, DingTalkMessage
from .lock_detector import LockDetector, LockDetectionResult
from .database import db_manager


class DingTalkWebhookService:
    """钉钉Webhook服务"""
    
    def __init__(self, app_key: str, app_secret: str, webhook_url: str = ""):
        self.callback_handler = DingTalkCallbackHandler(app_key, app_secret)
        self.message_sender = DingTalkMessageSender(webhook_url)
        self.lock_detector = LockDetector()
        
        # 如果有session webhook，使用session webhook
        self.session_webhook = ""
        self.session_webhook_expired_time = 0
    
    async def handle_callback(self, request: Request) -> Dict:
        """处理钉钉回调"""
        try:
            # 获取请求参数
            timestamp = request.headers.get('timestamp', '')
            sign = request.headers.get('sign', '')
            
            # 验证签名
            if not self.callback_handler.verify_signature(request, timestamp, sign):
                raise HTTPException(status_code=401, detail="签名验证失败")
            
            # 解析请求体
            body = await request.json()
            logger.info(f"收到钉钉回调: {json.dumps(body, ensure_ascii=False)}")
            
            # 解析消息
            message = self.callback_handler.parse_callback_message(body)
            
            # 检查是否@了机器人
            if not self.callback_handler.is_mentioned_to_bot(message):
                logger.info("消息未@机器人，忽略处理")
                return {"success": True, "message": "消息已忽略"}
            
            # 处理消息
            await self._process_message(message)
            
            return {"success": True, "message": "处理成功"}
            
        except Exception as e:
            logger.error(f"处理钉钉回调失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_message(self, message: DingTalkMessage):
        """处理消息"""
        try:
            # 更新session webhook
            if message.session_webhook and message.session_webhook_expired_time > 0:
                self.session_webhook = message.session_webhook
                self.session_webhook_expired_time = message.session_webhook_expired_time
                self.message_sender.webhook_url = message.session_webhook
                logger.info("更新session webhook")
            
            # 提取图片URL
            image_urls = self.callback_handler.extract_image_urls(message)
            
            if not image_urls:
                # 没有图片，发送提示消息
                await self._send_no_image_message(message)
                return
            
            # 处理第一张图片
            image_url = image_urls[0]
            download_code = ""
            if message.images and message.images.download_code:
                download_code = message.images.download_code[0]
            
            # 下载图片
            image = await self.callback_handler.download_image(image_url, download_code)
            if not image:
                await self._send_download_error_message(message)
                return
            
            # 检测锁
            detection_result = self.lock_detector.detect_locks(image)
            
            # 保存检测结果
            detection_id = self.lock_detector.save_detection_result(
                image, 
                detection_result,
                message_id=message.msg_id,
                user_id=message.sender_id,
                group_id=message.conversation_title
            )
            
            # 生成可视化结果
            result_image = self.lock_detector.visualize_detection(image, detection_result)
            
            # 发送检测结果
            await self._send_detection_result(message, detection_result, result_image)
            
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await self._send_error_message(message, str(e))
    
    async def _send_no_image_message(self, message: DingTalkMessage):
        """发送没有图片的提示消息"""
        try:
            content = f"""
@{message.sender_nick} 您好！

我注意到您@了我，但是没有在消息中包含图片。

请发送包含锁的图片，我将帮您检测锁的状态。

使用方法：
1. 在群聊中@我
2. 附上需要检测的图片
3. 我会自动检测并返回结果
            """
            
            await self.message_sender.send_markdown_message(
                "图片检测提示",
                content
            )
            
        except Exception as e:
            logger.error(f"发送提示消息失败: {e}")
    
    async def _send_download_error_message(self, message: DingTalkMessage):
        """发送下载失败的错误消息"""
        try:
            content = f"""
@{message.sender_nick} 抱歉！

图片下载失败，请检查图片格式或稍后重试。

如果问题持续存在，请联系管理员。
            """
            
            await self.message_sender.send_markdown_message(
                "图片下载失败",
                content
            )
            
        except Exception as e:
            logger.error(f"发送错误消息失败: {e}")
    
    async def _send_error_message(self, message: DingTalkMessage, error: str):
        """发送错误消息"""
        try:
            content = f"""
@{message.sender_nick} 抱歉！

处理过程中出现错误：{error}

请稍后重试或联系管理员。
            """
            
            await self.message_sender.send_markdown_message(
                "处理错误",
                content
            )
            
        except Exception as e:
            logger.error(f"发送错误消息失败: {e}")
    
    async def _send_detection_result(self, message: DingTalkMessage, 
                                   detection_result: LockDetectionResult, 
                                   result_image: Image.Image):
        """发送检测结果"""
        try:
            # 构建结果文本
            result_text = self._build_result_text(detection_result)
            
            # 转换图片为base64
            image_base64 = self._image_to_base64(result_image)
            
            # 构建Markdown消息
            markdown_content = f"""
@{message.sender_nick} 

## 🔒 锁检测结果

{result_text}

### 检测详情
- 检测时间: {detection_result.detection_time.strftime('%Y-%m-%d %H:%M:%S')}
- 置信度: {detection_result.confidence_score:.2f}
- 检测图片: 
![检测结果](data:image/jpeg;base64,{image_base64})

---
*Powered by Lock Detection AI*
            """
            
            success = await self.message_sender.send_markdown_message(
                "锁检测结果",
                markdown_content
            )
            
            if success:
                logger.info("检测结果发送成功")
            else:
                logger.error("检测结果发送失败")
                
        except Exception as e:
            logger.error(f"发送检测结果失败: {e}")
    
    def _build_result_text(self, detection_result: LockDetectionResult) -> str:
        """构建结果文本"""
        if detection_result.is_safe:
            return """
✅ **检测完成 - 一切正常！**

📊 **检测结果:**
- 总计检测到 {total_locks} 个锁
- 所有锁都已正常锁定
- 未发现安全隐患

🔒 **安全状态: 正常**
            """.format(total_locks=detection_result.total_locks)
        else:
            text = f"""
⚠️ **检测完成 - 发现安全隐患！**

📊 **检测结果:**
- 总计检测到 {detection_result.total_locks} 个锁
- {detection_result.locked_locks} 个锁已正常锁定
- {detection_result.unlocked_locks} 个锁未锁定 ❌

🔒 **安全状态: 警告**

**未锁定的锁:**
"""
            
            # 添加未锁定锁的详细信息
            for i, lock_detail in enumerate(detection_result.lock_details):
                if not lock_detail['is_locked']:
                    text += f"\n{i+1}. {lock_detail['lock_type']} (置信度: {lock_detail['confidence']:.2f})"
            
            text += "\n\n**建议:** 请立即检查并锁定所有未锁的锁！"
            
            return text
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将图片转换为base64编码"""
        try:
            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 调整图片大小以适应钉钉
            max_size = 800
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 转换为base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            logger.error(f"图片转换失败: {e}")
            return ""
    
    async def send_manual_detection_result(self, image: Image.Image, user_id: str = "") -> bool:
        """手动发送检测结果（用于API调用）"""
        try:
            # 检测锁
            detection_result = self.lock_detector.detect_locks(image)
            
            # 保存检测结果
            detection_id = self.lock_detector.save_detection_result(
                image, 
                detection_result,
                user_id=user_id
            )
            
            # 生成可视化结果
            result_image = self.lock_detector.visualize_detection(image, detection_result)
            
            # 发送检测结果
            result_text = self._build_result_text(detection_result)
            image_base64 = self._image_to_base64(result_image)
            
            markdown_content = f"""
## 🔒 锁检测结果

{result_text}

### 检测详情
- 检测时间: {detection_result.detection_time.strftime('%Y-%m-%d %H:%M:%S')}
- 置信度: {detection_result.confidence_score:.2f}
- 检测图片: 
![检测结果](data:image/jpeg;base64,{image_base64})

---
*Powered by Lock Detection AI*
            """
            
            return await self.message_sender.send_markdown_message(
                "锁检测结果",
                markdown_content
            )
            
        except Exception as e:
            logger.error(f"手动发送检测结果失败: {e}")
            return False


# 全局服务实例
webhook_service = None


def init_webhook_service(app_key: str, app_secret: str, webhook_url: str = ""):
    """初始化Webhook服务"""
    global webhook_service
    webhook_service = DingTalkWebhookService(app_key, app_secret, webhook_url)
    logger.info("钉钉Webhook服务已初始化")