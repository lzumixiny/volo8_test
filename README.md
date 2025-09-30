# 智能锁检测系统

基于YOLOv8的锁状态识别系统，支持图片检测、模型训练、钉钉通知等功能。

## 项目特性

- 🔍 **智能检测**: 基于YOLOv8的锁状态识别
- 🚀 **快速训练**: 支持命令行训练，自动化数据预处理
- 📱 **钉钉集成**: 支持钉钉机器人通知
- 🌐 **API服务**: 提供RESTful API接口
- 📊 **数据统计**: 完整的检测历史和统计信息

## 快速开始

### 环境要求

- Python 3.13+
- uv (包管理器)

### 安装依赖

```bash
# 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

### 数据集准备

将数据集按以下结构组织：

```

数据集中包含的类别： 
labels = [
    "upper_locked",
    "upper_unlocked",
    "upper_nolock",
    "lower_locked",
    "lower_unlocked",
    "lower_nolock",
    "knife_unlocked",
    "knife_locked",
    "lid_loose",
    "lid_tight"
]


datasets/lock_detech/
├── train/
│   ├── locked/        # 锁定状态图片
│   └── unlocked/      # 未锁定状态图片
├── val/
│   ├── locked/        # 验证集
│   └── unlocked/
└── test/
    ├── locked/        # 测试集
    └── unlocked/
```

### 模型训练

```bash
# 查看可用命令
python manage_simple.py --list-commands

# 查看数据集统计
python manage_simple.py dataset_stats

# 开始训练
python manage_simple.py train --epochs 50 --batch-size 8

# 验证模型
python manage_simple.py train --validate
```

### 启动API服务

```bash
# 启动FastAPI服务
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看API文档。

```bash
# 启动前端
cd frontend
python -m http.server 8080
```

## 项目结构

```
├── apps/                    # 应用模块
│   ├── commands.py         # 命令管理器
│   ├── train_commands.py   # 训练命令
│   ├── trainer.py          # 模型训练器
│   ├── lock_detector.py    # 锁检测器
│   ├── database.py         # 数据库管理
│   └── ...
├── datasets/               # 数据集
├── models/                 # 训练好的模型
├── tests/                  # 测试文件
├── main.py                 # FastAPI应用
├── manage_simple.py        # 命令行工具
└── pyproject.toml          # 项目配置
```

## 详细文档

- [命令行工具说明](COMMANDS.md)
- [API文档](API_DOCS.md)
- [系统架构](ARCHITECTURE.md)
- [设计文档](DESIGN_DOCS.md)
- [测试文档](TEST_DOCS.md)

## 开发指南

### 依赖管理

本项目使用 `uv` 进行依赖管理：

```bash
# 添加依赖
uv add package_name

# 移除依赖
uv remove package_name

# 更新依赖
uv update
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_api.py
```

## 许可证

MIT License