# train.py
from __future__ import annotations


# os is a module for interacting with the operating system
import os
# argparse is a module for parsing command-line arguments
import argparse 
# platform is a module for getting the platform of the operating system
import platform
# sys is a module for getting the system information
import sys
# time is a module for getting the time
import time 
# dataclasses is a module for creating data classes
from dataclasses import dataclass
# pathlib is a module for working with paths
from pathlib import Path
# typing is a module for type hints
from typing import Any, Dict

# yaml is a module for parsing YAML files
import yaml

try:
    import torch
except ImportError as e:
    torch = None
    _torch_import_error = e


# @dataclass是装饰器，用于创建数据类，自动生成__init__方法、__repr__方法、__eq__方法等
@dataclass
class RunConfig:
    name: str
    seed: int
    results_dir: str


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float


@dataclass
class SystemConfig:
    device: str  # auto|cpu|cuda
    num_workers: int


@dataclass
class Config:
    run: RunConfig
    train: TrainConfig
    system: SystemConfig


# 解析命令行参数
def parse_args() -> argparse.Namespace:
    # 创建一个ArgumentParser对象
    p = argparse.ArgumentParser()
    # 添加一个参数，--config，类型为str，必填，帮助信息为"Path to YAML config"
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    # 解析命令行参数
    return p.parse_args()


# 加载YAML文件  path: YAML文件路径
def load_yaml(path: str) -> Dict[str, Any]:
    # 打开YAML文件
    with open(path, "r", encoding="utf-8") as f:
        # 加载YAML文件
        return yaml.safe_load(f)
    # 返回YAML文件内容


# 构建配置 Config: 包含run、train、system三个配置的类
def build_config(d: Dict[str, Any]) -> Config:
    # 最小化验证，清晰错误
    # 遍历d中的键，如果键不在["run", "train", "system"]中，则抛出KeyError异常
    for k in ["run", "train", "system"]:
        if k not in d:
            raise KeyError(f"Missing top-level key '{k}' in config")

    # 获取run、train、system配置
    run = d["run"]
    train = d["train"]
    system = d["system"]

    # 构建配置
    cfg = Config(
        run=RunConfig(
            name=str(run.get("name", "baseline")),
            seed=int(run.get("seed", 42)),
            results_dir=str(run.get("results_dir", "results")),
        ),
        train=TrainConfig(
            epochs=int(train.get("epochs", 1)),
            batch_size=int(train.get("batch_size", 64)),
            lr=float(train.get("lr", 1e-3)),
        ),
        system=SystemConfig(
            device=str(system.get("device", "auto")),
            num_workers=int(system.get("num_workers", 2)),
        ),
    )
    # 返回配置
    return cfg


# 选择设备 device_cfg: 设备配置
def pick_device(device_cfg: str) -> str:
    # 如果torch模块导入失败，则抛出RuntimeError异常
    if torch is None:
        raise RuntimeError(f"PyTorch import failed: {_torch_import_error}")

    # 将device_cfg转换为小写
    device_cfg = device_cfg.lower()
    # 如果device_cfg为"cpu"，则返回"cpu"
    if device_cfg == "cpu":
        return "cpu"
    # 如果device_cfg为"mps"，则返回"mps"
    if device_cfg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Config requested MPS but torch.backends.mps.is_available() is False")
        return "mps"    
    # 如果device_cfg为"cuda"，则返回"cuda"
    if device_cfg == "cuda":
        # 如果torch.cuda.is_available()为False，则抛出RuntimeError异常
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is False")
        return "cuda"
    # 如果device_cfg为"auto"，则返回"cuda" if torch.cuda.is_available() else "cpu"
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # 如果device_cfg为其他值，则抛出ValueError异常
    raise ValueError(f"Unknown system.device='{device_cfg}'. Use auto|cpu|cuda")


def set_seed(seed: int) -> None:
    if torch is None:
        return
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_run_dir(results_dir: str, run_name: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(results_dir) / f"{run_name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def print_system_info() -> None:
    print("=== System Info ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    if torch is None:
        print(f"PyTorch: import failed: {_torch_import_error}")
        return
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS device count: {torch.backends.mps.device_count()}")
        print(f"MPS device 0: {torch.backends.mps.get_device_name(0)}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
    print("===================")


# 主函数    
def main() -> None:
    # 解析命令行参数
    args = parse_args()
    # 加载YAML文件
    raw = load_yaml(args.config)
    # 构建配置
    cfg = build_config(raw)
    # 打印系统信息
    print_system_info()
    # 选择设备
    device = pick_device(cfg.system.device)
    print(f"Selected device: {device}")
    # 设置随机种子
    set_seed(cfg.run.seed)
    # 确保运行目录
    run_dir = ensure_run_dir(cfg.run.results_dir, cfg.run.name)

if __name__ == "__main__":
    main()
