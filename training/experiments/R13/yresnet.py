"""R13: Y-shaped ResNet других участников команды (proto/YShapedResNet_patch.zip) в постановке task.pdf.

Архитектура — их YShapedResNet без изменений (models_yresnet.py, копия из архива): общий
ResNet-кодировщик → две независимые ветви декодера «Re» и «Im» → комплексная голова → проекция
|U| = A_in. Отличаются только вход и выход, как у PhaseNet3D в R12. В task.pdf освещение —
плоская волна, поэтому канал освещения — единицы (у других участников команды — гауссов профиль 200 мкм).
Выход (Re U, Im U) переводится в φ = atan2, как у core.net.UNet; при |U| = 1 это и есть
фаза их поля. Цели подаются как в core.net — T ∈ [0, 1].

Импорт модуля регистрирует «yresnet» в core.net.MODELS: после этого модель обучает
doetrain.train.train и загружает core.net.load_model (формат весов проекта).
"""

import sys
from pathlib import Path

import torch
from torch import nn

from doetrain.core import net

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_yresnet import YShapedResNet


class YResNet(nn.Module):
    """[B, 4, N, N] (цели T) → фаза [B, N, N]."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.net = YShapedResNet(in_channels=5, base=base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"вход — [B, 4, N, N], получено {tuple(x.shape)}")
        plane_wave = torch.ones_like(x[:, :1])
        ri = self.net(torch.cat([plane_wave, x], dim=1))
        return torch.atan2(ri[:, 1], ri[:, 0])


net.MODELS.setdefault("yresnet", YResNet)
