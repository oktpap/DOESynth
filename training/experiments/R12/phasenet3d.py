"""R12: PhaseNet3D_Physics других участников команды (R11) в постановке task.pdf.

Архитектура — их PhaseNet3DPhysics без изменений (experiments/R11-phasenet3d/PhaseNet3D_Physics/
models.py): четыре цели как объём глубины → Conv3d-блоки → спектральные блоки → осевое внимание →
одна фаза ДОЭ. Отличаются только вход и выход. В task.pdf освещение — плоская волна, поэтому
канал освещения — единицы (у других участников команды — гауссов профиль 200 мкм). Выход (cos φ, sin φ)
переводится в φ = atan2, как у core.net.UNet. Цели подаются как в core.net — T ∈ [0, 1].

Импорт модуля регистрирует «phasenet3d» в core.net.MODELS: после этого модель обучает
doetrain.train.train и загружает core.net.load_model (формат весов проекта).
"""

import sys
from pathlib import Path

import torch
from torch import nn

from doetrain.core import net

_R11 = Path(__file__).resolve().parent.parent / "R11-phasenet3d"
if str(_R11) not in sys.path:
    sys.path.insert(0, str(_R11))
from PhaseNet3D_Physics.models import PhaseNet3DPhysics


class PhaseNet3D(nn.Module):
    """[B, 4, N, N] (цели T) → фаза [B, N, N]."""

    def __init__(self, base: int = 16, modes: int = 12):
        super().__init__()
        self.net = PhaseNet3DPhysics(in_channels=5, base=base, modes=modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"вход — [B, 4, N, N], получено {tuple(x.shape)}")
        plane_wave = torch.ones_like(x[:, :1])
        ab = self.net(torch.cat([plane_wave, x], dim=1))
        return torch.atan2(ab[:, 1], ab[:, 0])


net.MODELS.setdefault("phasenet3d", PhaseNet3D)
