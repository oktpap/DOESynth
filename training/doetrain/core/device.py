"""Выбор вычислительного устройства."""

import os

import torch

REQUIRE_GPU_ENV = "DOETRAIN_REQUIRE_GPU"


def gpu_required() -> bool:
    """Обязателен ли GPU: так запускаем обучение и тесты на ПК."""
    return os.environ.get(REQUIRE_GPU_ENV) == "1"


def get_device() -> torch.device:
    """CUDA, если она есть, иначе CPU. Если GPU обязателен, без CUDA падает."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if gpu_required():
        raise RuntimeError(f"CUDA недоступна, а {REQUIRE_GPU_ENV}=1: на CPU считать не будем")
    return torch.device("cpu")
