"""PhaseNet3D (архитектура других участников команды, эксперимент R12) в постановке условия.

Сеть PhaseNet3DPhysics перенесена без изменений: четыре цели как объём глубины → Conv3d-блоки →
спектральные блоки → осевое внимание → одна фаза ДОЭ. Освещение по условию — плоская волна,
поэтому канал освещения — единицы. Выход (cos φ, sin φ) переводится в φ = atan2.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels):
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvBlock2D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SpectralVolumeBlock(nn.Module):
    """Обучаемое смешивание низших пространственных частот в каждом срезе глубины."""

    def __init__(self, channels, modes=12):
        super().__init__()
        self.modes = modes
        self.wr = nn.Parameter(0.02 * torch.randn(channels, modes, modes))
        self.wi = nn.Parameter(0.02 * torch.randn(channels, modes, modes))
        self.mix = nn.Conv3d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        h, w = x.shape[-2:]
        xf = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        my = min(self.modes, h // 2)
        mx = min(self.modes, w // 2 + 1)
        weight = torch.complex(self.wr[:, :my, :mx], self.wi[:, :my, :mx])
        yf = torch.zeros_like(xf)
        yf[:, :, :, :my, :mx] = xf[:, :, :, :my, :mx] * weight[None, :, None]
        yf[:, :, :, -my:, :mx] = xf[:, :, :, -my:, :mx] * weight[None, :, None]
        y = torch.fft.irfft2(yf, s=(h, w), dim=(-2, -1), norm="ortho")
        return x + self.scale * self.mix(y)


class ResidualVolumeBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class PhaseNet3DPhysics(nn.Module):
    """Объёмный кодировщик четырёх плоскостей + фурье-блоки + одна фазовая голова ДОЭ."""

    def __init__(self, in_channels=5, base=8, modes=12):
        super().__init__()
        if in_channels != 5:
            raise ValueError("PhaseNet3DPhysics requires 5 input channels")
        self.stem = nn.Conv3d(2, base, 3, padding=1)
        self.vol1 = ResidualVolumeBlock(base)
        self.fourier1 = SpectralVolumeBlock(base, modes)
        self.down = nn.Conv3d(
            base, base * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.vol2 = ResidualVolumeBlock(base * 2)
        self.fourier2 = SpectralVolumeBlock(base * 2, max(4, modes // 2))
        self.depth_score = nn.Conv3d(base * 2, 1, 1)
        self.skip = ConvBlock2D(5, base)
        self.fuse = ConvBlock2D(base * 3, base * 2)
        self.refine = ConvBlock2D(base * 2, base)
        self.out = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        object_volume = x[:, 1:5].unsqueeze(1)
        illumination = x[:, 0:1].unsqueeze(2).expand(-1, -1, 4, -1, -1)
        volume = torch.cat((object_volume, illumination), dim=1)
        v = self.stem(volume)
        v = self.fourier1(self.vol1(v))
        v = self.down(v)
        v = self.fourier2(self.vol2(v))
        scores = self.depth_score(v).squeeze(1)
        weights = torch.softmax(scores, dim=1).unsqueeze(1)
        axial = (v * weights).sum(dim=2)
        axial = F.interpolate(axial, size=x.shape[-2:], mode="bilinear", align_corners=False)
        spatial = self.skip(x)
        fused = self.fuse(torch.cat((spatial, axial), dim=1))
        features = self.refine(fused)
        return F.normalize(self.out(features), p=2, dim=1, eps=1e-8)


class PhaseNet3D(nn.Module):
    """[B, 4, N, N] (цели T) → фаза [B, N, N]."""

    def __init__(self, base: int = 16, modes: int = 12):
        super().__init__()
        self.net = PhaseNet3DPhysics(in_channels=5, base=base, modes=modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plane_wave = torch.ones_like(x[:, :1])
        ab = self.net(torch.cat([plane_wave, x], dim=1))
        return torch.atan2(ab[:, 1], ab[:, 0])
