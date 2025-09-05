# 锁检测系统命令行工具

## 概述

本项目已将 `start_training` 功能从API接口改造为独立的命令行工具，类似Django的Command系统。

## 安装依赖

首先安装项目依赖：

```bash
pip install loguru ultralytics pandas pillow pyyaml
```

或者安装完整依赖：

```bash
pip install -r pyproject.toml
```

## 命令行工具使用

### 查看可用命令

```bash
python manage_simple.py --list-commands
```

### 训练模型

```bash
# 基本训练（使用默认数据集路径）
python manage_simple.py train

# 自定义参数训练
python manage_simple.py train \
  --model-name my_lock_detector \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640 \
  --learning-rate 0.001 \
  --device 0 \
  --dataset-path datasets/lock_detech \
  --validate
```

### 查看数据集统计

```bash
python manage_simple.py dataset_stats
```

### 导出模型

```bash
# 导出为ONNX格式
python manage_simple.py export_model models/lock_detector/weights/best.pt

# 导出为其他格式
python manage_simple.py export_model models/lock_detector/weights/best.pt --format torchscript
```

## 命令参数说明

### train 命令

- `--model-name`: 模型名称 (默认: lock_detector)
- `--epochs`: 训练轮数 (默认: 100)
- `--batch-size`: 批次大小 (默认: 16)
- `--img-size`: 图片尺寸 (默认: 640)
- `--learning-rate`: 学习率 (默认: 0.001)
- `--device`: 训练设备 (默认: 0)
- `--dataset-path`: 数据集路径 (默认: datasets/lock_detech)
- `--validate`: 训练完成后验证模型

### export_model 命令

- `model_path`: 模型文件路径 (必需)
- `--format`: 导出格式 (默认: onnx，可选: onnx, torchscript, coreml, tensorflow)

## 与原API的区别

1. **移除了API接口**: 不再需要通过HTTP请求启动训练
2. **直接命令行调用**: 更适合批处理和自动化任务
3. **更好的参数控制**: 支持丰富的命令行参数
4. **独立运行**: 不依赖FastAPI服务器

## 数据集说明

### 数据集结构

训练系统支持以下数据集结构：

```
datasets/lock_detech/
├── train/
│   ├── locked/        # 锁定状态的图片
│   └── unlocked/      # 未锁定状态的图片
├── val/
│   ├── locked/        # 验证集锁定状态图片
│   └── unlocked/      # 验证集未锁定状态图片
└── test/
    ├── locked/        # 测试集锁定状态图片
    └── unlocked/      # 测试集未锁定状态图片
```

### 数据集统计

当前数据集包含：
- 训练集：153张图片 (129张锁定，24张未锁定)
- 验证集：58张图片 (45张锁定，13张未锁定)
- 测试集：31张图片 (24张锁定，7张未锁定)

### 自动标注生成

系统会自动为图片生成YOLO格式的标注文件。对于分类任务，每张图片会被标注为对应类别（整个图片作为目标区域）。

## 依赖管理

本项目使用 `uv` 进行依赖管理：

```bash
# 安装依赖
uv sync

# 添加新依赖
uv add package_name

# 移除依赖
uv remove package_name

# 更新依赖
uv update
```

## 注意事项

1. 确保已安装所有依赖包：`uv sync`
2. 训练数据按照上述目录结构组织
3. 训练过程可能需要较长时间，建议在GPU环境下运行
4. 确保有足够的磁盘空间存储训练结果
5. 首次运行时会自动生成标注文件
6. 不要使用 `pip` 安装依赖，统一使用 `uv` 管理