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
    # 创建一个列表，用于存储变换
    # 这个列表是一个记录变换的列表，用于后续的compose操作
    tfms = []

    # 1) 把(H,W)的图像转换为(1,H,W)的图像并缩放到[0,1]
    # ToTensor() 会把它转成 float32 并缩放到 [0,1]。
    tfms.append(transforms.ToTensor())

    # 2) 如果normalize="mnist"，则使用常见MNIST mean/std进行标准化
    if config.normalize == "mnist":
        # 经典 MNIST 统计量（在 ToTensor 后的 [0,1] 空间上）
        # mean=0.1307, std=0.3081 经常用于 baseline
        tfms.append(transforms.Normalize((0.1307,), (0.3081,)))

    # # 3) 如果print_batch_stats为True，则在构建dataloader后抓取一个batch打印shape/统计
    # if config.print_batch_stats:
    #     tfms.append(transforms.Lambda(lambda x: print(x.shape, x.min(), x.max())))

    # 4) 返回一个Compose对象，用于将多个变换组合在一起
    return transforms.Compose(tfms)



def build_mnist_dataloaders(
    cfg: Union[MNISTDataConfig, Any],
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    构建 MNIST 的 train/val DataLoader，并返回 meta 信息（用于 notes/日志）。

    支持两种配置类型：
    - MNISTDataConfig: 完整的数据配置对象
    - Config (from train.py): 从 train.py 传入的配置对象

    返回：
      train_loader, val_loader, meta

    meta 会包含：
      - input_shape: 期望模型输入（不含 batch 维）
      - normalize_strategy: 归一化/标准化描述
      - split_sizes: train/val 样本数
    """
    if isinstance(cfg, MNISTDataConfig):
        data_dir = cfg.data_dir
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        pin_memory = cfg.pin_memory
        persistent_workers = cfg.persistent_workers
        val_split = cfg.val_split
        seed = cfg.seed
        flatten = cfg.flatten
        normalize = cfg.normalize
        print_batch_stats = cfg.print_batch_stats
    else:
        # 如果是 Config 对象（从 train.py 传入），提取相应字段
        data_dir = "data"  # 默认值
        batch_size = cfg.train.batch_size
        num_workers = cfg.system.num_workers
        pin_memory = True  # 默认值
        persistent_workers = True  # 默认值
        val_split = 0.1  # 默认值
        seed = cfg.run.seed
        flatten = False  # 默认值
        normalize = "mnist"  # 默认值
        print_batch_stats = True  # 默认值


    # 1) 构建变换
    transform = _build_transforms(MNISTDataConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        val_split=val_split,
        seed=seed,
        flatten=flatten,
        normalize=normalize,
        print_batch_stats=print_batch_stats,
    ))

    # 2) 构建数据集 - 使用提取的 data_dir 变量
    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    # 3) 从训练集中拆分出训练集和验证集 - 使用提取的变量
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"Invalid val_split={val_split}. Must be in (0,1)")
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    # 使用generator+seed 确保可复现拆分
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)


    # 4) 构建Dataloader：
    # 训练集一般需要shuffle，验证集不需要
    persistent = num_workers > 0 and persistent_workers
    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=persistent
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=persistent
    )

    # ---------- 5) 生成 meta 信息 ----------
    # input_shape：不含 batch 维度
    # flatten=False => (1,28,28)
    # flatten=True  => (784,)
    input_shape = (784,) if flatten else (1, 28, 28)

    if normalize == "none":
        normalize_strategy = "ToTensor() only: uint8[0,255] -> float32[0,1]"
    else:
        normalize_strategy = "ToTensor() + Normalize(mean=0.1307, std=0.3081)"

    meta = {
        "input_shape": input_shape,
        "normalize_strategy": normalize_strategy,
        "split_sizes": (train_size, val_size),
        "seed": seed,
        "flatten": flatten,
        "normalize": normalize,
        "batch_size": batch_size,
    }

    if print_batch_stats:
        print_mnist_batch_stats(train_dl, tag="train")
        print_mnist_batch_stats(val_dl, tag="val")


    return train_dl, val_dl, meta

@torch.no_grad()
def print_mnist_batch_stats(loader: DataLoader, tag: str = "train") -> None:
    """
    打印一个 batch 的 shape / dtype / 数值范围 / 均值方差，确保你对数据完全清楚。

    你需要在 notes 里记录这些输出（至少记录 X.shape, y.shape 和归一化策略）。
    """
    x, y = next(iter(loader))

    # x shape:
    # - flatten=False: (B,1,28,28)
    # - flatten=True : (B,784)
    # y shape: (B,)
    print(f"[{tag}] X.shape = {tuple(x.shape)}, y.shape = {tuple(y.shape)}")
    print(f"[{tag}] X.dtype = {x.dtype}, y.dtype = {y.dtype}")

    # 注意：如果使用 Normalize，min/max/mean/std 不是 [0,1] 语义
    # 这里打印的是“当前 transform 后”的张量统计，便于核对
    x_min = float(x.min().cpu())
    x_max = float(x.max().cpu())
    x_mean = float(x.mean().cpu())
    x_std = float(x.std(unbiased=False).cpu())

    print(f"[{tag}] X.min={x_min:.4f}, X.max={x_max:.4f}, X.mean={x_mean:.4f}, X.std={x_std:.4f}")
    print(f"[{tag}] y.min={int(y.min())}, y.max={int(y.max())} (labels should be 0..9)")
        