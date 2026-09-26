"""全局随机种子：尽量保证全链路可复现。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def same_seeds(seed: int) -> None:
    """固定 Python / NumPy / PyTorch / CUDA 随机源，并开启确定性后端。"""
    seed = int(seed)

    # 影响子进程；当前进程的 hash 随机化仍需在启动前设置 PYTHONHASHSEED。
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CUDA >= 10.2 下 CuBLAS 确定性所需。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    # warn_only：无确定性实现的算子只告警，避免直接崩溃。
    torch.use_deterministic_algorithms(True, warn_only=True)
