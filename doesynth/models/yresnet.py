"""Y-shaped ResNet (архитектура других участников команды, эксперимент R13) в постановке условия.

Общий ResNet-кодировщик → две ветви декодера (Re и Im) → комплексная голова → проекция
|U| = A_in. Сеть перенесена без изменений; освещение — плоская волна (канал из единиц),
выход (Re U, Im U) переводится в фазу φ = atan2.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_groups(out_channels), out_channels),
            )
        )
        self.act = nn.GELU()

    def forward(self, x):
        identity = self.skip(x)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + identity)


class SharedResidualEncoder(nn.Module):
    def __init__(self, in_channels: int, base: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 5, padding=2, bias=False),
            nn.GroupNorm(_groups(base), base),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(ResidualBlock2D(base, base), ResidualBlock2D(base, base))
        self.down2 = ResidualBlock2D(base, 2 * base, stride=2)
        self.stage2 = nn.Sequential(
            ResidualBlock2D(2 * base, 2 * base), ResidualBlock2D(2 * base, 2 * base)
        )
        self.down3 = ResidualBlock2D(2 * base, 4 * base, stride=2)
        self.stage3 = nn.Sequential(
            ResidualBlock2D(4 * base, 4 * base), ResidualBlock2D(4 * base, 4 * base)
        )

    def forward(self, x):
        e1 = self.stage1(self.stem(x))
        e2 = self.stage2(self.down2(e1))
        e3 = self.stage3(self.down3(e2))
        return e1, e2, e3


class ResidualDecoderBranch(nn.Module):
    def __init__(self, base: int):
        super().__init__()
        self.bottleneck = nn.Sequential(
            ResidualBlock2D(4 * base, 4 * base), ResidualBlock2D(4 * base, 4 * base)
        )
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.decode2 = nn.Sequential(
            ResidualBlock2D(4 * base, 2 * base), ResidualBlock2D(2 * base, 2 * base)
        )
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.decode1 = nn.Sequential(ResidualBlock2D(2 * base, base), ResidualBlock2D(base, base))

    def forward(self, e1, e2, e3):
        y = self.bottleneck(e3)
        y = self.up2(y)
        if y.shape[-2:] != e2.shape[-2:]:
            y = F.interpolate(y, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.decode2(torch.cat((y, e2), dim=1))
        y = self.up1(y)
        if y.shape[-2:] != e1.shape[-2:]:
            y = F.interpolate(y, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.decode1(torch.cat((y, e1), dim=1))


class ComplexMergeHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight_real = nn.Conv2d(channels, 1, 1, bias=False)
        self.weight_imag = nn.Conv2d(channels, 1, 1, bias=False)
        self.bias_real = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.bias_imag = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, real_features, imag_features):
        wr_r = self.weight_real(real_features)
        wi_i = self.weight_imag(imag_features)
        wi_r = self.weight_imag(real_features)
        wr_i = self.weight_real(imag_features)
        return wr_r - wi_i + self.bias_real, wi_r + wr_i + self.bias_imag


class YShapedResNet(nn.Module):
    def __init__(self, in_channels: int = 5, base: int = 16, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.encoder = SharedResidualEncoder(in_channels, base)
        self.real_branch = ResidualDecoderBranch(base)
        self.imag_branch = ResidualDecoderBranch(base)
        self.complex_head = ComplexMergeHead(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illumination = x[:, 0:1].clamp_min(0.0)
        e1, e2, e3 = self.encoder(x)
        raw_real, raw_imag = self.complex_head(
            self.real_branch(e1, e2, e3), self.imag_branch(e1, e2, e3)
        )
        magnitude = torch.sqrt(raw_real.square() + raw_imag.square())
        safe = magnitude.clamp_min(self.eps)
        unit_real, unit_imag = raw_real / safe, raw_imag / safe
        degenerate = magnitude <= self.eps
        unit_real = torch.where(degenerate, torch.ones_like(unit_real), unit_real)
        unit_imag = torch.where(degenerate, torch.zeros_like(unit_imag), unit_imag)
        return torch.cat((illumination * unit_real, illumination * unit_imag), dim=1)


class YResNet(nn.Module):
    """[B, 4, N, N] (цели T) → фаза [B, N, N]."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.net = YShapedResNet(in_channels=5, base=base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plane_wave = torch.ones_like(x[:, :1])
        ri = self.net(torch.cat([plane_wave, x], dim=1))
        return torch.atan2(ri[:, 1], ri[:, 0])
