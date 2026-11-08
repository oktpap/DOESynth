"""Архитектуры сетей и загрузка весов.

Файл весов — словарь {"model": архитектура, "kwargs": параметры конструктора,
"state_dict": веса, "image_power": конвенция изображения (2 — |K|²), "meta": …}; формат
общий для всех экспериментов команды, файлы перенесены из прогонов без изменений.
"""

from functools import cache
from pathlib import Path

import torch
from torch import nn

from doesynth.models.phasenet3d import PhaseNet3D
from doesynth.models.project import CGHNet, SuperposNet, UNet, UnrolledNet
from doesynth.models.yresnet import YResNet
from doesynth.optics import IMAGE_POWER

ARCHITECTURES: dict[str, type[nn.Module]] = {
    "unet": UNet,
    "superpos": SuperposNet,
    "unrolled": UnrolledNet,
    "cghnet": CGHNet,
    "phasenet3d": PhaseNet3D,
    "yresnet": YResNet,
}


@cache
def load(path: Path, device: str = "cpu") -> nn.Module:
    """Сеть в режиме инференса; кэшируется по (путь, устройство)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    name = ckpt["model"]
    if name not in ARCHITECTURES:
        raise ValueError(f"{path.name}: неизвестная архитектура {name!r}")
    model = ARCHITECTURES[name](**ckpt["kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    power = ckpt.get("image_power", 2)
    if power != IMAGE_POWER:
        raise ValueError(f"{path.name}: веса обучены под |K|^{power}, решение считает |K|^2")
    return model.to(device).eval()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
