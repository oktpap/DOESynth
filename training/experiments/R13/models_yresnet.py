"""Independent Y-shaped residual network for complex DOE-field prediction.

The network receives five real-valued channels:
    [Gaussian illumination amplitude, target intensity planes 1..4]

A shared residual encoder feeds two independent residual decoders (real and
imaginary branches). Their feature maps are combined by a complex-linear head:
    Re(W * (a + i b)) = Wr(a) - Wi(b)
    Im(W * (a + i b)) = Wi(a) + Wr(b)

The raw complex output is projected onto the phase-only DOE constraint
|U_D​OE(x,y)| = A_in(x,y). The returned tensor is [B,2,H,W] containing the
physical real and imaginary components, not a unit phasor.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResidualBlock2D(nn.Module):
    """Two-convolution residual block with a projection shortcut if needed."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                               bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1
                     else nn.Sequential(
                         nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                         nn.GroupNorm(_groups(out_channels), out_channels)))
        self.act = nn.GELU()

    def forward(self, x):
        identity = self.skip(x)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + identity)


class SharedResidualEncoder(nn.Module):
    """Shared spatial feature extractor before the Y-shaped split."""
    def __init__(self, in_channels: int, base: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 5, padding=2, bias=False),
            nn.GroupNorm(_groups(base), base),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock2D(base, base),
            ResidualBlock2D(base, base),
        )
        self.down2 = ResidualBlock2D(base, 2 * base, stride=2)
        self.stage2 = nn.Sequential(
            ResidualBlock2D(2 * base, 2 * base),
            ResidualBlock2D(2 * base, 2 * base),
        )
        self.down3 = ResidualBlock2D(2 * base, 4 * base, stride=2)
        self.stage3 = nn.Sequential(
            ResidualBlock2D(4 * base, 4 * base),
            ResidualBlock2D(4 * base, 4 * base),
        )

    def forward(self, x):
        e1 = self.stage1(self.stem(x))
        e2 = self.stage2(self.down2(e1))
        e3 = self.stage3(self.down3(e2))
        return e1, e2, e3


class ResidualDecoderBranch(nn.Module):
    """One independent arm of the Y: real or imaginary field component."""
    def __init__(self, base: int):
        super().__init__()
        self.bottleneck = nn.Sequential(
            ResidualBlock2D(4 * base, 4 * base),
            ResidualBlock2D(4 * base, 4 * base),
        )
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.decode2 = nn.Sequential(
            ResidualBlock2D(4 * base, 2 * base),
            ResidualBlock2D(2 * base, 2 * base),
        )
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.decode1 = nn.Sequential(
            ResidualBlock2D(2 * base, base),
            ResidualBlock2D(base, base),
        )

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
    """Complex-linear mixing of the two branch feature maps.

    It implements the real/imaginary cross-add/subtract relations of a complex
    convolution, followed by two scalar projections to Re(U) and Im(U).
    """
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
        real = wr_r - wi_i + self.bias_real
        imag = wi_r + wr_i + self.bias_imag
        return real, imag


class YShapedResNet(nn.Module):
    """Y-shaped ResNet that predicts one complex field for a phase-only DOE.

    Input:
        x: [B,5,H,W] = illumination amplitude + four target intensity images.

    Output:
        field_ri: [B,2,H,W] = [Re(U_D​OE), Im(U_D​OE)].
        The field magnitude is projected to x[:,0], so |U_D​OE| equals the
        supplied illumination amplitude pixel-by-pixel.

    `forward_complex` returns a native complex tensor [B,H,W].
    """
    def __init__(self, in_channels: int = 5, base: int = 16, eps: float = 1e-8):
        super().__init__()
        if in_channels != 5:
            raise ValueError("YShapedResNet expects 5 channels: illumination + four target planes")
        if base < 4:
            raise ValueError("base must be >= 4")
        self.in_channels = in_channels
        self.base = base
        self.eps = eps
        self.encoder = SharedResidualEncoder(in_channels, base)
        self.real_branch = ResidualDecoderBranch(base)
        self.imag_branch = ResidualDecoderBranch(base)
        self.complex_head = ComplexMergeHead(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 5:
            raise ValueError(f"expected 5 channels in input [B,5,H,W], got {tuple(x.shape)}")
        if x.shape[-2] < 8 or x.shape[-1] < 8:
            raise ValueError("spatial dimensions must be at least 8x8")
        illumination = x[:, 0:1].clamp_min(0.0)
        e1, e2, e3 = self.encoder(x)
        real_features = self.real_branch(e1, e2, e3)
        imag_features = self.imag_branch(e1, e2, e3)
        raw_real, raw_imag = self.complex_head(real_features, imag_features)

        # Projection onto the phase-only DOE constraint |U| = A_in.
        magnitude = torch.sqrt(raw_real.square() + raw_imag.square())
        safe_magnitude = magnitude.clamp_min(self.eps)
        unit_real = raw_real / safe_magnitude
        unit_imag = raw_imag / safe_magnitude
        # Keep the modulus constraint exact even for a degenerate zero vector.
        degenerate = magnitude <= self.eps
        unit_real = torch.where(degenerate, torch.ones_like(unit_real), unit_real)
        unit_imag = torch.where(degenerate, torch.zeros_like(unit_imag), unit_imag)
        real = illumination * unit_real
        imag = illumination * unit_imag
        return torch.cat((real, imag), dim=1)

    def forward_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Return complex DOE field U = Re(U) + i Im(U), shape [B,H,W]."""
        field_ri = self.forward(x)
        return torch.complex(field_ri[:, 0], field_ri[:, 1])

    @staticmethod
    def phase_from_field(field_ri: torch.Tensor) -> torch.Tensor:
        if field_ri.ndim != 4 or field_ri.shape[1] != 2:
            raise ValueError("field_ri must have shape [B,2,H,W]")
        return torch.atan2(field_ri[:, 1], field_ri[:, 0])
