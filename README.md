# Learning AI from Scratch

从零开始学习人工智能和深度学习的项目。

## 项目简介

本项目旨在通过实践的方式，从零开始学习 AI 和深度学习的基础知识。项目包含完整的训练流程、数据加载、配置管理等模块。

## 项目结构

```
Learning-AI-from-Scratch/
├── my-dl-plan/              # 深度学习计划目录
│   ├── configs/             # 配置文件目录
│   │   └── baseline.yaml    # 基线配置文件
│   ├── src/                 # 源代码目录
│   │   └── data/            # 数据处理模块
│   │       └── mnist.py     # MNIST 数据加载器
│   ├── notes/               # 笔记目录
│   │   └── env.md           # 环境配置笔记
│   ├── requirements.txt     # Python 依赖包
│   └── train.py             # 主训练脚本
├── Plan.md                  # 学习计划
└── README.md                # 项目说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 2.9.1
- CUDA（可选，用于 GPU 加速）

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd Learning-AI-from-Scratch
```

### 2. 创建虚拟环境（推荐）

使用 conda：

```bash
conda create -n ai-learning python=3.10
conda activate ai-learning
```

或使用 venv：

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
cd my-dl-plan
pip install -r requirements.txt
```

## 如何运行

### 基本运行

使用默认配置文件运行训练：

```bash
cd my-dl-plan
python train.py --config configs/baseline.yaml
```

### 配置文件说明

配置文件采用 YAML 格式，包含三个主要部分：

- **run**: 运行配置
  - `name`: 运行名称
  - `seed`: 随机种子
  - `results_dir`: 结果保存目录

- **train**: 训练配置
  - `epochs`: 训练轮数
  - `batch_size`: 批次大小
  - `lr`: 学习率

- **system**: 系统配置
  - `device`: 设备选择（`auto`/`cpu`/`cuda`）
  - `num_workers`: 数据加载的 worker 数量

### 示例配置文件

```yaml
run:
  name: "baseline"
  seed: 42
  results_dir: "results"

train:
  epochs: 1
  batch_size: 64
  lr: 0.001

system:
  device: "auto"   # auto | cpu | cuda
  num_workers: 2
```

### 自定义配置

你可以创建自己的配置文件，然后运行：

```bash
python train.py --config configs/your_config.yaml
```

## 功能特性

- ✅ 完整的训练流程框架
- ✅ YAML 配置文件支持
- ✅ 自动设备选择（CPU/CUDA）
- ✅ 可复现的随机种子设置
- ✅ 系统信息打印
- ✅ MNIST 数据加载支持
- ✅ 结果目录自动创建（带时间戳）

## 输出说明

训练运行后，会在 `results_dir` 目录下创建一个带时间戳的结果目录，格式为：`{run_name}_{YYYYMMDD-HHMMSS}`

例如：`results/baseline_20251220-143022/`

## 注意事项

1. 首次运行时会自动下载 MNIST 数据集
2. 如果使用 GPU，确保已正确安装 CUDA 和 cuDNN
3. `device: "auto"` 会自动检测并使用可用的 GPU，否则使用 CPU
4. 确保有足够的磁盘空间存储数据集和结果

## 开发计划

详见 [Plan.md](Plan.md)

## 许可证

[根据项目实际情况添加许可证信息]

