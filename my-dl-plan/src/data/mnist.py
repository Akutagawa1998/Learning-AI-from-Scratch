# src/data/mnist.py
"""
MNIST 数据管线（train/val）。

Day 2 目标：
- 数据加载可控（可复现拆分）
- shape 完全清楚（(B,1,28,28) 或 flatten 为 (B,784)）
- 归一化策略明确，并在运行时打印关键统计
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass(frozen=True)
class MNISTDataConfig:
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


def _build_transforms(config: MNISTDataConfig) -> transforms.Compose:
    tfms = [transforms.ToTensor()]
    if config.normalize == "mnist":
        tfms.append(transforms.Normalize(mean=(MNIST_MEAN,), std=(MNIST_STD,)))
    elif config.normalize != "none":
        raise ValueError(f"Unknown normalize='{config.normalize}'. Use 'none' or 'mnist'.")
    if config.flatten:
        tfms.append(transforms.Lambda(lambda x: x.view(-1)))
    return transforms.Compose(tfms)


@torch.no_grad()
def _print_batch_stats(tag: str, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = batch
    x_stats = (x.min().item(), x.max().item(), x.mean().item(), x.std().item())
    print(f"[{tag}] X.shape = {tuple(x.shape)}")
    print(f"[{tag}] y.shape = {tuple(y.shape)}")
    print(
        f"[{tag}] X.min/max/mean/std = "
        f"{x_stats[0]:.4f} / {x_stats[1]:.4f} / {x_stats[2]:.4f} / {x_stats[3]:.4f}"
    )


def build_mnist_dataloaders(
    cfg: MNISTDataConfig,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    tfms = _build_transforms(cfg)

    train_full = datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        transform=tfms,
        download=True,
    )
    val_size = int(len(train_full) * cfg.val_split)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=generator)

    persistent = cfg.num_workers > 0 and cfg.persistent_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent,
    )

    info = {
        "normalize": cfg.normalize,
        "flatten": cfg.flatten,
        "mean": MNIST_MEAN,
        "std": MNIST_STD,
        "train_size": train_size,
        "val_size": val_size,
    }

    if cfg.print_batch_stats:
        _print_batch_stats("train", next(iter(train_loader)))

    return train_loader, val_loader, info
