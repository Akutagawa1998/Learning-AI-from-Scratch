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