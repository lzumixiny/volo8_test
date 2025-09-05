# 智能锁检测系统 - API文档

## 概述

智能锁检测系统是基于YOLOv8的锁状态识别API，支持钉钉机器人回调、图片检测、模型训练等功能。该系统可以自动检测图片中的锁，判断锁是否正常锁定，并通过钉钉机器人发送检测结果。

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   钉钉群聊      │    │   FastAPI      │    │   数据库        │
│                 │◄──►│   服务         │◄──►│   (SQLite)     │
│                 │    │                 │    │                 │
│  @机器人发送     │    │  - 锁检测API    │    │  - 检测结果     │
│  图片           │    │  - 钉钉回调     │    │  - 训练记录     │
│                 │    │  - 模型训练     │    │  - 统计信息     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │   YOLOv8       │
                       │   模型         │
                       │                 │
                       │  - 锁检测       │
                       │  - 状态判断     │
                       │  - 可视化       │
                       └─────────────────┘
```

## 技术栈

- **后端框架**: FastAPI
- **机器学习**: YOLOv8 (Ultralytics)
- **数据库**: SQLite
- **图像处理**: PIL, scikit-image
- **异步处理**: aiohttp
- **日志系统**: loguru
- **数据验证**: Pydantic

## API接口

### 1. 锁检测接口

#### POST /api/v1/lock/detect

检测图片中的锁状态

**请求参数:**
- `file`: 图片文件 (必需)
- `user_id`: 用户ID (可选)

**响应示例:**
```json
{
  "success": true,
  "message": "检测完成",
  "data": {
    "detection_id": 1,
    "result": {
      "is_safe": false,
      "total_locks": 3,
      "unlocked_locks": 1,
      "locked_locks": 2,
      "lock_details": [
        {
          "lock_type": "unlocked_lock",
          "is_locked": false,
          "confidence": 0.95,
          "bbox": {
            "xmin": 100,
            "ymin": 200,
            "xmax": 300,
            "ymax": 400
          }
        }
      ],
      "confidence_score": 0.95,
      "detection_time": "2025-09-04T10:30:00"
    },
    "image_base64": "base64编码的检测结果图片"
  }
}
```

### 2. 钉钉配置接口

#### POST /api/v1/dingtalk/configure

配置钉钉机器人

**请求参数:**
```json
{
  "app_key": "your_app_key",
  "app_secret": "your_app_secret",
  "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=xxx"
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "钉钉配置成功"
}
```

### 3. 钉钉回调接口

#### POST /api/v1/dingtalk/webhook

钉钉机器人回调接口，处理群聊中的@消息

**请求头:**
- `timestamp`: 时间戳
- `sign`: 签名

**请求体:** 钉钉回调消息格式

**响应示例:**
```json
{
  "success": true,
  "message": "处理成功"
}
```

### 4. 模型训练接口

#### POST /api/v1/train/start

启动模型训练任务

**请求参数:**
```json
{
  "model_name": "lock_detector",
  "epochs": 100,
  "batch_size": 16,
  "img_size": 640,
  "learning_rate": 0.001,
  "device": "auto"
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "训练任务已启动",
  "data": {
    "model_name": "lock_detector"
  }
}
```

### 5. 统计信息接口

#### GET /api/v1/stats

获取系统统计信息

**响应示例:**
```json
{
  "success": true,
  "message": "获取统计信息成功",
  "data": {
    "detection_stats": {
      "total_detections": 100,
      "unsafe_detections": 5,
      "total_locks": 250,
      "total_unlocked": 8,
      "today_detections": 10,
      "safety_rate": 95.0
    },
    "dataset_stats": {
      "train": {
        "images": 800,
        "labels": 800,
        "path": "dataset/train/images"
      },
      "val": {
        "images": 100,
        "labels": 100,
        "path": "dataset/val/images"
      },
      "test": {
        "images": 100,
        "labels": 100,
        "path": "dataset/test/images"
      }
    }
  }
}
```

### 6. 检测历史接口

#### GET /api/v1/history

获取检测历史记录

**请求参数:**
- `limit`: 限制数量 (默认: 10)
- `offset`: 偏移量 (默认: 0)

**响应示例:**
```json
{
  "success": true,
  "message": "获取检测历史成功",
  "data": {
    "history": [
      {
        "id": 1,
        "image_url": "",
        "image_hash": "abc123",
        "detection_time": "2025-09-04T10:30:00",
        "locks_detected": 3,
        "unlocked_locks": 1,
        "lock_positions": [...],
        "confidence_score": 0.95,
        "is_safe": false,
        "created_at": "2025-09-04T10:30:00"
      }
    ]
  }
}
```

### 7. 健康检查接口

#### GET /api/v1/health

系统健康检查

**响应示例:**
```json
{
  "success": true,
  "message": "系统正常运行",
  "data": {
    "database": "connected",
    "model_loaded": true,
    "total_detections": 100
  }
}
```

## 兼容的旧接口

系统保留了原有的对象检测接口，确保向后兼容：

### POST /img_object_detection_to_json

从图像中进行对象检测并返回JSON结果

### POST /img_object_detection_to_img

从图像中进行对象检测并在图像上绘制边界框

## 错误处理

所有接口都遵循统一的错误响应格式：

```json
{
  "success": false,
  "message": "错误描述"
}
```

## 钉钉机器人使用说明

### 1. 配置钉钉机器人

1. 在钉钉群中添加自定义机器人
2. 获取 `app_key` 和 `app_secret`
3. 调用 `/api/v1/dingtalk/configure` 接口配置

### 2. 使用方法

在群聊中@机器人并发送包含锁的图片：

```
@锁检测机器人 [图片]
```

系统会自动：
1. 识别@消息
2. 下载图片
3. 检测锁状态
4. 发送检测结果

### 3. 检测结果示例

**安全状态:**
```
✅ 检测完成 - 一切正常！

📊 检测结果:
- 总计检测到 3 个锁
- 所有锁都已正常锁定
- 未发现安全隐患

🔒 安全状态: 正常
```

**警告状态:**
```
⚠️ 检测完成 - 发现安全隐患！

📊 检测结果:
- 总计检测到 3 个锁
- 2 个锁已正常锁定
- 1 个锁未锁定 ❌

🔒 安全状态: 警告

未锁定的锁:
1. padlock (置信度: 0.95)

建议: 请立即检查并锁定所有未锁的锁！
```

## 环境变量配置

```bash
# 钉钉机器人配置
DINGTALK_APP_KEY=your_app_key
DINGTALK_APP_SECRET=your_app_secret
DINGTALK_WEBHOOK_URL=your_webhook_url

# 服务配置
HOST=0.0.0.0
PORT=8000
```

## 启动服务

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API文档

启动服务后，访问以下地址查看完整的API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## 数据库结构

系统使用SQLite数据库，包含以下表：

### detection_results
检测结果主表
- `id`: 主键
- `image_url`: 图片URL
- `image_hash`: 图片哈希
- `detection_time`: 检测时间
- `locks_detected`: 检测到的锁数量
- `unlocked_locks`: 未锁的锁数量
- `lock_positions`: 锁位置信息(JSON)
- `confidence_score`: 置信度
- `dingtalk_message_id`: 钉钉消息ID
- `user_id`: 用户ID
- `group_id`: 群组ID
- `is_safe`: 是否安全
- `created_at`: 创建时间

### lock_details
锁详细信息表
- `id`: 主键
- `detection_id`: 关联的检测结果ID
- `lock_type`: 锁类型
- `is_locked`: 是否锁定
- `confidence`: 置信度
- `position_x`: X坐标
- `position_y`: Y坐标
- `width`: 宽度
- `height`: 高度

### training_records
训练记录表
- `id`: 主键
- `model_name`: 模型名称
- `training_start`: 训练开始时间
- `training_end`: 训练结束时间
- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `train_loss`: 训练损失
- `val_loss`: 验证损失
- `map_score`: mAP分数
- `model_path`: 模型路径
- `dataset_size`: 数据集大小
- `status`: 状态
- `created_at`: 创建时间