# Learning AI from Scratch

从零开始学习人工智能和深度学习的项目。本项目采用**12周（84天）每日计划**，每天2小时，系统性地从基础训练脚手架到多卡分布式训练，最终完成大模型训练与优化的完整流程。

## 📋 学习计划概览

本项目遵循**12周（84天）每日计划**，每天固定2小时，分为：
- **20分钟**：输入学习
- **80分钟**：工程/实验
- **20分钟**：记录/复盘

### 12周路线图

| 周次 | 主题 | 核心目标 |
|------|------|----------|
| **Week 1** | 训练脚手架 v0 | 单卡跑通，可复现训练闭环（MNIST） |
| **Week 2** | 多卡 DDP + AMP | 4卡分布式训练，吞吐/显存 profiling |
| **Week 3** | 训练学核心 | CIFAR10 + MLP/CNN，规范 ablation |
| **Week 4** | Transformer 组件 | 实现 decoder-only block，字符级 LM |
| **Week 5** | LM Pipeline | Tokenizer + pack + ppl eval + benchmark |
| **Week 6** | 多卡实验规范 | 同预算对照，研究式决策能力 |
| **Week 7** | SFT 闭环 | 指令微调 + 评测 harness |
| **Week 8** | LoRA / QLoRA | PEFT 对照与决策 |
| **Week 9** | DPO | 偏好优化训练与评测 |
| **Week 10** | 分布式显存方案 | FSDP/ZeRO 使用与性能分析 |
| **Week 11** | 推理优化 | KV cache + FlashAttention |
| **Week 12** | 作品集打包 | 面试资产整理 |

详细每日计划请参考 [Plan.md](Plan.md)

## 🏗️ 项目结构

```
Learning-AI-from-Scratch/
├── my-dl-plan/              # 深度学习计划目录
│   ├── configs/             # 配置文件目录
│   │   └── baseline.yaml    # 基线配置文件
│   ├── src/                 # 源代码目录
<<<<<<< Updated upstream
│   │   ├── data/            # 数据处理模块
│   │   │   └── mnist.py     # MNIST 数据加载器
│   │   ├── models/          # 模型定义（待完善）
│   │   ├── train/           # 训练模块（待完善）
│   │   ├── eval/            # 评估模块（待完善）
│   │   └── utils/           # 工具函数（待完善）
│   ├── scripts/             # 脚本目录
│   │   └── slurm/           # Slurm 提交脚本（待完善）
│   ├── results/             # 结果目录
│   │   └── week01/          # 每周结果
│   │       ├── metrics.csv  # 指标记录
│   │       ├── plots/       # 图表
│   │       └── notes.md     # 实验笔记
=======
│   │   ├── models/          # 模型定义
│   │   └── data/            # 数据处理模块
│   │       └── mnist.py     # MNIST 数据加载器
>>>>>>> Stashed changes
│   ├── notes/               # 笔记目录
│   │   ├── env.md           # 环境配置笔记
│   │   └── day3.md          # Day 3 训练笔记
│   ├── requirements.txt     # Python 依赖包
│   ├── train.py             # 主训练脚本
│   └── REPRODUCE.md         # 复现指南（待完善）
├── Plan.md                  # 学习计划
└── README.md                # 项目说明文档
```

## ⚙️ 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.9.1
- **CUDA**: 可选，用于 GPU 加速
- **集群环境**: 支持 Slurm（可选，也可使用 tmux/nohup）

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd Learning-AI-from-Scratch
```

### 2. 创建虚拟环境

使用 conda（推荐）：

```bash
conda create -n dl python=3.10
conda activate dl
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
# Apple (macOS / Apple Silicon)
pip install -r requirements-apple.txt

# CUDA (Linux + NVIDIA GPU)
pip install -r requirements-cuda.txt
```

### 4. 运行训练

#### 单卡训练（当前阶段）

```bash
cd my-dl-plan
python train.py --config configs/baseline.yaml
```

<<<<<<< Updated upstream
#### 多卡训练（Week 2+）

使用 Slurm 提交（示例）：

```bash
sbatch scripts/slurm/mnist_ddp.sh
```

或直接使用 torchrun：

```bash
torchrun --nproc_per_node=4 train.py --config configs/mnist_ddp.yaml
```

## 📝 配置文件说明
=======
### Day 3 线性模型（Softmax Regression）

```bash
cd my-dl-plan
python train.py --config configs/linear_mnist.yaml
```

### 配置文件说明
>>>>>>> Stashed changes

配置文件采用 YAML 格式，包含三个主要部分：

### 示例配置

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

### 配置项说明

- **run**: 运行配置
  - `name`: 运行名称
  - `seed`: 随机种子（用于可复现性）
  - `results_dir`: 结果保存目录

- **train**: 训练配置
  - `epochs`: 训练轮数
  - `batch_size`: 批次大小
  - `lr`: 学习率

- **system**: 系统配置
  - `device`: 设备选择（`auto`/`cpu`/`cuda`）
  - `num_workers`: 数据加载的 worker 数量

## 📊 当前进度

### Week 1: 训练脚手架 v0

**已完成**：
- ✅ Day 1: 环境与骨架（repo 初始化、基础环境）
- ✅ Day 2: 数据管线（MNIST 数据加载器）
- 🔄 Day 3-7: 进行中...

**待完成**：
- [ ] Day 3: 最小模型（Softmax Regression）
- [ ] Day 4: 训练循环（ckpt + eval）
- [ ] Day 5: 复现性（seed 与配置）
- [ ] Day 6: 最小对照（学习率）
- [ ] Day 7: 周清理与版本标记

## 🎯 每周交付物

每周必须产出：

1. **REPRODUCE.md** 更新（能从零复现）
2. **results/weekXX/metrics.csv + plots/**（指标与图表）
3. **results/weekXX/notes.md**（假设-方法-结果-解释-下一步）

## 📈 输出说明

训练运行后，会在 `results_dir` 目录下创建一个带时间戳的结果目录：

```
results/
└── baseline_20251220-143022/
    ├── checkpoints/         # 模型检查点
    ├── metrics.csv          # 训练指标
    └── logs/                # 训练日志
```

## 🔧 集群工作流（Week 2+）

### Slurm 提交脚本模板

```bash
#!/bin/bash
#SBATCH -J mnist_ddp
#SBATCH -p a100
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
source ~/.bashrc
conda activate dl

export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=8

<<<<<<< Updated upstream
python -m torch.distributed.run --nproc_per_node=4 \
  train.py --config configs/mnist_ddp.yaml
```

## 📚 学习原则

1. **每天2小时固定切分**：20分钟学习 + 80分钟工程 + 20分钟复盘
2. **每天必须产出可验证结果**：训练脚本可运行 / 提交成功 / 指标曲线生成
3. **每周固定三件套交付**：REPRODUCE.md + metrics.csv + notes.md
4. **短作业 + 次日验收**：每天2小时用于准备、提交、诊断、写结论

## ⚠️ 注意事项
=======
Day 3 会额外写入 `results/week01/metrics.csv`，包含每个 epoch 的 `train_loss` 与 `val_acc`。

## 注意事项
>>>>>>> Stashed changes

1. 首次运行时会自动下载 MNIST 数据集
2. 如果使用 GPU，确保已正确安装 CUDA 和 cuDNN
3. `device: "auto"` 会自动检测并使用可用的 GPU，否则使用 CPU
4. 确保有足够的磁盘空间存储数据集和结果
5. 多卡训练需要确保 NCCL 环境正确配置

## 🔗 相关文档

- [Plan.md](Plan.md) - 详细的学习计划
- [REPRODUCE.md](my-dl-plan/REPRODUCE.md) - 复现指南（待完善）
- [notes/env.md](my-dl-plan/notes/env.md) - 环境配置笔记

## 📅 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-20 | Day 1: 创建 repo，环境配置，基础训练脚本 |
| 2025-12-21 | Day 2: MNIST 数据加载器实现 |

## 📄 许可证

[根据项目实际情况添加许可证信息]
