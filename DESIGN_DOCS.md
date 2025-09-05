# 智能锁检测系统 - 设计文档

## 1. 项目概述

### 1.1 项目背景
智能锁检测系统是一个基于深度学习的锁状态识别系统，通过YOLOv8模型自动检测图片中的锁，判断锁是否正常锁定，并支持通过钉钉机器人进行交互。

### 1.2 项目目标
- 实现高精度的锁检测和状态识别
- 提供便捷的钉钉机器人交互接口
- 支持模型的训练和优化
- 完整的数据管理和审计功能
- 高可用性和可扩展性

### 1.3 技术选型
- **深度学习框架**: YOLOv8 (Ultralytics)
- **Web框架**: FastAPI
- **数据库**: SQLite
- **图像处理**: PIL, scikit-image
- **异步处理**: aiohttp
- **配置管理**: 环境变量

## 2. 系统架构设计

### 2.1 整体架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   用户界面      │    │   应用服务层    │    │   数据存储层    │
│                 │    │                 │    │                 │
│  - 钉钉群聊     │    │  - FastAPI      │    │  - SQLite       │
│  - API客户端    │    │  - 锁检测服务   │    │  - 模型文件     │
│  - 管理界面     │    │  - 钉钉服务     │    │  - 日志文件     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │   AI模型层      │
                       │                 │
                       │  - YOLOv8模型   │
                       │  - 特征提取     │
                       │  - 状态判断     │
                       └─────────────────┘
```

### 2.2 模块划分

#### 2.2.1 应用层 (main.py)
- API路由管理
- 请求处理和响应
- 中间件配置
- 系统启动配置

#### 2.2.2 服务层 (apps/)
- **lock_detector.py**: 锁检测核心逻辑
- **dingtalk.py**: 钉钉消息处理
- **webhook_service.py**: Webhook服务
- **trainer.py**: 模型训练服务
- **database.py**: 数据库管理

#### 2.2.3 数据层
- SQLite数据库
- 模型文件存储
- 日志文件

## 3. 核心功能设计

### 3.1 锁检测模块

#### 3.1.1 检测流程
1. **图像预处理**: 格式转换、尺寸调整
2. **模型推理**: YOLOv8模型预测
3. **结果解析**: 边界框、置信度、类别
4. **状态判断**: 基于特征判断锁状态
5. **结果生成**: 结构化检测结果

#### 3.1.2 锁状态判断算法
```python
def _determine_lock_status(self, lock_type, confidence, image, bbox):
    # 基于类别名称判断
    if lock_type == "locked_lock":
        return True
    elif lock_type == "unlocked_lock":
        return False
    
    # 基于图像特征判断
    lock_region = image.crop(bbox)
    return self._analyze_lock_region(lock_region)
```

#### 3.1.3 特征分析算法
- **亮度分析**: 计算区域亮度标准差
- **边缘检测**: 使用Canny边缘检测
- **纹理分析**: 分析锁孔区域的纹理特征

### 3.2 钉钉集成模块

#### 3.2.1 消息处理流程
1. **签名验证**: 验证回调请求的合法性
2. **消息解析**: 解析钉钉消息格式
3. **@识别**: 检测是否@了机器人
4. **图片提取**: 从消息中提取图片URL
5. **图片下载**: 异步下载图片
6. **检测处理**: 调用锁检测服务
7. **结果发送**: 发送Markdown格式的检测结果

#### 3.2.2 消息格式处理
- 支持文本、图片、@用户等多种消息类型
- 自动过滤无关消息
- 支持session webhook动态更新

### 3.3 模型训练模块

#### 3.3.1 数据集管理
- 自动创建数据集目录结构
- 支持数据集分割（训练/验证/测试）
- 数据集配置文件生成

#### 3.3.2 训练流程
1. **数据准备**: 数据集加载和预处理
2. **模型配置**: 训练参数设置
3. **模型训练**: YOLOv8训练流程
4. **结果验证**: 模型性能评估
5. **模型保存**: 保存训练好的模型

#### 3.3.3 训练记录管理
- 训练过程记录
- 性能指标统计
- 模型版本管理

### 3.4 数据管理模块

#### 3.4.1 数据库设计
- **detection_results**: 检测结果主表
- **lock_details**: 锁详细信息表
- **training_records**: 训练记录表

#### 3.4.2 数据操作
- 检测结果存储
- 历史记录查询
- 统计信息计算
- 数据备份和恢复

## 4. 数据库设计

### 4.1 表结构设计

#### 4.1.1 detection_results 表
| 字段名 | 类型 | 约束 | 描述 |
|--------|------|------|------|
| id | INTEGER | PRIMARY KEY | 主键 |
| image_url | TEXT | NOT NULL | 图片URL |
| image_hash | TEXT | UNIQUE NOT NULL | 图片哈希 |
| detection_time | DATETIME | NOT NULL | 检测时间 |
| locks_detected | INTEGER | DEFAULT 0 | 检测到的锁数量 |
| unlocked_locks | INTEGER | DEFAULT 0 | 未锁的锁数量 |
| lock_positions | TEXT | | 锁位置信息(JSON) |
| confidence_score | REAL | DEFAULT 0.0 | 置信度 |
| dingtalk_message_id | TEXT | | 钉钉消息ID |
| user_id | TEXT | | 用户ID |
| group_id | TEXT | | 群组ID |
| is_safe | BOOLEAN | DEFAULT TRUE | 是否安全 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 创建时间 |

#### 4.1.2 lock_details 表
| 字段名 | 类型 | 约束 | 描述 |
|--------|------|------|------|
| id | INTEGER | PRIMARY KEY | 主键 |
| detection_id | INTEGER | FOREIGN KEY | 关联检测结果ID |
| lock_type | TEXT | | 锁类型 |
| is_locked | BOOLEAN | | 是否锁定 |
| confidence | REAL | | 置信度 |
| position_x | INTEGER | | X坐标 |
| position_y | INTEGER | | Y坐标 |
| width | INTEGER | | 宽度 |
| height | INTEGER | | 高度 |

#### 4.1.3 training_records 表
| 字段名 | 类型 | 约束 | 描述 |
|--------|------|------|------|
| id | INTEGER | PRIMARY KEY | 主键 |
| model_name | TEXT | NOT NULL | 模型名称 |
| training_start | DATETIME | NOT NULL | 训练开始时间 |
| training_end | DATETIME | | 训练结束时间 |
| epochs | INTEGER | | 训练轮数 |
| batch_size | INTEGER | | 批次大小 |
| learning_rate | REAL | | 学习率 |
| train_loss | REAL | | 训练损失 |
| val_loss | REAL | | 验证损失 |
| map_score | REAL | | mAP分数 |
| model_path | TEXT | | 模型路径 |
| dataset_size | INTEGER | | 数据集大小 |
| status | TEXT | DEFAULT 'pending' | 状态 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 创建时间 |

### 4.2 索引设计
- `detection_results(image_hash)`: 唯一索引，防止重复检测
- `detection_results(detection_time)`: 普通索引，加速时间范围查询
- `lock_details(detection_id)`: 外键索引，加速关联查询
- `training_records(model_name)`: 普通索引，加速模型查询

## 5. API设计

### 5.1 RESTful API设计原则
- 使用标准的HTTP方法（GET、POST）
- 统一的响应格式
- 清晰的错误处理
- 完整的参数验证

### 5.2 API版本控制
- 所有API使用 `/api/v1/` 前缀
- 保持向后兼容性
- 支持多版本共存

### 5.3 安全设计
- 钉钉回调签名验证
- 文件上传大小限制
- 输入参数验证
- 错误信息脱敏

## 6. 部署设计

### 6.1 容器化部署
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 环境配置
```bash
# 钉钉配置
DINGTALK_APP_KEY=your_app_key
DINGTALK_APP_SECRET=your_app_secret
DINGTALK_WEBHOOK_URL=your_webhook_url

# 服务配置
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

### 6.3 启动脚本
```bash
#!/bin/bash
# 启动服务
uvicorn main:app --host $HOST --port $PORT --reload
```

## 7. 性能优化

### 7.1 模型优化
- 使用量化模型减少内存占用
- 实现模型缓存机制
- 支持GPU加速

### 7.2 数据库优化
- 合理的索引设计
- 批量操作优化
- 连接池管理

### 7.3 并发处理
- 异步IO处理
- 请求队列管理
- 资源限制控制

## 8. 监控和日志

### 8.1 日志设计
- 分级日志记录
- 结构化日志格式
- 日志轮转和压缩

### 8.2 监控指标
- API响应时间
- 检测准确率
- 系统资源使用率
- 错误率统计

### 8.3 健康检查
- 数据库连接检查
- 模型加载状态检查
- 磁盘空间检查

## 9. 扩展性设计

### 9.1 模型扩展
- 支持多种检测模型
- 模型版本管理
- A/B测试支持

### 9.2 功能扩展
- 支持更多消息平台
- 批量检测功能
- 用户权限管理

### 9.3 部署扩展
- 支持集群部署
- 负载均衡
- 自动扩缩容

## 10. 安全考虑

### 10.1 数据安全
- 图片数据加密存储
- 敏感信息脱敏
- 访问权限控制

### 10.2 系统安全
- API访问限制
- 防止SQL注入
- 文件上传安全

### 10.3 隐私保护
- 用户数据保护
- 检测结果保密
- 日志信息脱敏

## 11. 测试策略

### 11.1 单元测试
- 核心算法测试
- 数据库操作测试
- API接口测试

### 11.2 集成测试
- 端到端测试
- 钉钉回调测试
- 模型训练测试

### 11.3 性能测试
- 并发性能测试
- 内存使用测试
- 响应时间测试

## 12. 维护和运维

### 12.1 备份策略
- 数据库定期备份
- 模型文件备份
- 配置文件备份

### 12.2 更新策略
- 滚动更新
- 版本回滚
- 数据迁移

### 12.3 故障处理
- 故障检测
- 自动恢复
- 告警机制