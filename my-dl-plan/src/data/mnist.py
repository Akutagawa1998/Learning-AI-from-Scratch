# src/data/mnist.py
"""
MNIST 数据管线（train/val）

Day 2 目标：
- 数据加载可控（可复现拆分）
- shape 完全清楚（(B,1,28,28) 或 flatten 为 (B,784)）
- 归一化策略明确，并在运行时打印关键统计

依赖：
- torch
- torchvision
"""

# __future__ 是 Python 的一个模块，用于导入未来的 Python 特性（在当前版本中可能默认关闭的特性）
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# @dataclass 是 Python 的一个装饰器，用于创建数据类，自动生成__init__方法、__repr__方法、__eq__方法等
# frozen=True 表示数据类是不可变的，即不能修改其属性
@dataclass(frozen=True)
class MNISTDataConfig:
    """
    data_dir:
        MNIST 下载/缓存目录
    batch_size:
        DataLoader batch size
    num_workers:
        DataLoader worker 数（本地可设 2~8；服务器也可更大）
    pin_memory:
        GPU 训练常用 True（能加速 H2D 拷贝）
    persistent_workers:
        num_workers>0 时可 True，减少每个 epoch 重建 worker 的开销
    val_split:
        从原始 train 集中切出多少比例作为 val（如 0.1）
    seed:
        拆分可复现用
    flatten:
        True: 输出 X 为 (B,784)，适合 MLP
        False: 输出 X 为 (B,1,28,28)，适合 CNN
    normalize:
        "none": 仅 ToTensor() -> [0,1]
        "mnist": 使用经典 MNIST mean/std 标准化
    print_batch_stats:
        True 时在构建 dataloader 后抓取一个 batch 打印 shape/统计
    """

    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    val_split: float = 0.1
    seed: int = 42

    flatten: bool = False
    normalize: str = "mnist"  # "none" or "mnist"
    print_batch_stats: bool = True


# 在 Python 中，以下划线开头的函数名表示“内部使用”或“私有”的约定，通常用于表示函数仅供内部使用，不对外暴露
# 这个私有和内部使用指的是模块级别的
def _build_transforms(config: MNISTDataConfig) -> transforms.Compose:
    """
    构建 torchvision transform。

    MNIST 原始像素是 [0,255] 的 uint8。
    transforms.ToTensor() 会把它转成 float32 并缩放到 [0,1]。

    如果 normalize="mnist"：
    使用常见 MNIST mean/std 进行标准化：
        x_norm = (x - mean) / std
    注意：标准化后的数值范围不再是 [0,1]，而是大致以 0 为中心。
    """