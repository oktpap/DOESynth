"""Архитектуры команды: U-Net, обученная суперпозиция, развёрнутый GS и 3D-CGH-Net.

Все сети получают 4 целевых изображения T ∈ [0, 1] как 4 канала и выдают фазу ДОЭ [B, N, N].
Фаза кодируется двумя каналами (a, b) → atan2(b, a), чтобы выход не упирался в разрыв на ±π.
Обучение без учителя: цели → сеть → фаза → дифференцируемый Френель → 1 − Пирсон по
восстановлениям (эталонные фазовые маски не нужны).

- UNet — прямой U-Net «цели → фаза» (эксперименты R02, R03, R07).
- SuperposNet — обученная суперпозиция: сеть ставит каждой цели фазу и усиление в её плоскости,
  перенос на ДОЭ — точный обратный оператор Френеля (R08-superpos).
- UnrolledNet — суперпозиция и 6 развёрнутых итераций многоплоскостного GS с обучаемой
  проекцией (R06).
- CGHNet — 3D-CGH-Net (Rymov et al., Opt. Lasers Eng. 184 (2025) 108645): Y-net (общий
  кодировщик, ветви амплитуды и фазы полей плоскостей) + блок переноса «обратный Френель каждой
  плоскости → сумма → фаза» и её доработки (R08): пропуск амплитуды цели, итерации GS, обучаемые
  шаги спуска, второй проход, кодировщик в плоскости ДОЭ, чередование пикселей.
"""

import torch
from torch import nn

from doesynth.methods import plane_losses, unit_energy
from doesynth.optics import (
    IMAGE_POWER,
    TASK,
    backpropagate,
    check_power,
    chirps,
    propagate,
    target_amplitude,
)


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
    )


class UNet(nn.Module):
    """U-Net с depth понижениями разрешения; на 128×128 и depth = 4 дно 8×8 видит всё поле."""

    def __init__(self, in_ch: int = 4, base: int = 32, depth: int = 4):
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(in_ch, chans[0])])
        self.enc.extend(_block(chans[i], chans[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth))
        )
        self.dec = nn.ModuleList(_block(2 * chans[i], chans[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chans[0], 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else nn.functional.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec, strict=True):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        ab = self.head(x)
        return torch.atan2(ab[:, 1], ab[:, 0])


class _Trunk(nn.Module):
    """U-Net с заданным числом выходных каналов; билинейный апсемплинг + свёртка 1×1."""

    def __init__(self, in_ch: int, out_ch: int, base: int, depth: int):
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(in_ch, chans[0])])
        self.enc.extend(_block(chans[i], chans[i + 1]) for i in range(depth))
        self.reduce = nn.ModuleList(
            nn.Conv2d(chans[i + 1], chans[i], 1) for i in reversed(range(depth))
        )
        self.dec = nn.ModuleList(_block(2 * chans[i], chans[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chans[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else nn.functional.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for reduce, dec in zip(self.reduce, self.dec, strict=True):
            up = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
            x = dec(torch.cat([reduce(up), skips.pop()], dim=1))
        return self.head(x)


class SuperposNet(nn.Module):
    """Обученная суперпозиция: ствол ставит каждой плоскости фазу ψ_p и усиление m_p,
    поля A_p·m_p·exp(jψ_p) переносятся на ДОЭ точным обратным оператором и складываются;
    малый кодировщик с нулевой головой правит сумму до чисто фазового ДОЭ."""

    def __init__(
        self,
        base: int = 32,
        depth: int = 4,
        refine_base: int = 16,
        refine_depth: int = 2,
        image_power: int = IMAGE_POWER,
    ):
        super().__init__()
        self.image_power = check_power(image_power)
        planes = len(TASK.distances)
        self.trunk = _Trunk(planes, 3 * planes, base, depth)
        self.refiner = _Trunk(3, 2, refine_base, refine_depth) if refine_base > 0 else None
        self.register_buffer("chirp", chirps(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = self.trunk(x).split(x.shape[1], dim=1)
        amp = target_amplitude(x, self.image_power) * 2 * torch.sigmoid(c)
        total = backpropagate(torch.polar(amp, torch.atan2(b, a)), self.chirp).sum(dim=-3)
        if self.refiner is None:
            return torch.atan2(total.imag, total.real)
        unit = total / total.abs().clamp_min(1e-12)
        m = total.abs() / total.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        delta = self.refiner(torch.stack([unit.real, unit.imag, m - 1], dim=1))
        return torch.atan2(unit.imag + delta[:, 1], unit.real + delta[:, 0])


class _Update(nn.Module):
    """Обучаемая проекция одной итерации развёрнутого GS (два масштаба, голова нулевая)."""

    def __init__(self, planes: int, base: int):
        super().__init__()
        self.fine = _block(2 * planes, base)
        self.coarse = _block(base, 2 * base)
        self.reduce = nn.Conv2d(2 * base, base, 1)
        self.merge = _block(2 * base, base)
        self.head = nn.Conv2d(base, 3 * planes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.fine(x)
        h = self.coarse(nn.functional.max_pool2d(f, 2))
        up = nn.functional.interpolate(h, scale_factor=2, mode="bilinear")
        return self.head(self.merge(torch.cat([f, self.reduce(up)], dim=1)))


class UnrolledNet(nn.Module):
    """Развёрнутый многоплоскостной GS с обучаемой проекцией: старт — обученная суперпозиция,
    затем iters итераций со своими весами; при нулевых головах итерация — ровно проекция GS."""

    def __init__(
        self,
        iters: int = 6,
        base: int = 16,
        init_base: int = 16,
        init_depth: int = 4,
        image_power: int = IMAGE_POWER,
    ):
        super().__init__()
        self.image_power = check_power(image_power)
        planes = len(TASK.distances)
        self.start = SuperposNet(
            base=init_base, depth=init_depth, refine_base=0, image_power=image_power
        )
        self.updates = nn.ModuleList(_Update(planes, base) for _ in range(iters))
        self.register_buffer("chirp", chirps(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        planes = x.shape[1]
        amp = unit_energy(target_amplitude(x, self.image_power))
        phase = self.start(x)
        for update in self.updates:
            k = propagate(torch.polar(torch.ones_like(phase), phase), self.chirp)
            intensity = k.real.square() + k.imag.square()
            image = intensity if self.image_power == 2 else (intensity + 1e-12).sqrt()
            image = image / image.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            da, u, v = update(torch.cat([x, image], dim=1)).split(planes, dim=1)
            unit = k * torch.rsqrt(intensity + 1e-12)
            g = unit * torch.complex(1 + u, v) * (amp + da)
            total = backpropagate(g, self.chirp).sum(dim=-3)
            phase = torch.atan2(total.imag, total.real)
        return phase


class _Decoder(nn.Module):
    """Ветвь декодера над общим кодировщиком (билинейный апсемплинг + свёртка 1×1)."""

    def __init__(self, chans: list[int], out_ch: int):
        super().__init__()
        depth = len(chans) - 1
        self.reduce = nn.ModuleList(
            nn.Conv2d(chans[i + 1], chans[i], 1) for i in reversed(range(depth))
        )
        self.dec = nn.ModuleList(_block(2 * chans[i], chans[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chans[0], out_ch, 1)

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        skips = list(skips)
        x = skips.pop()
        for reduce, dec in zip(self.reduce, self.dec, strict=True):
            up = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
            x = dec(torch.cat([reduce(up), skips.pop()], dim=1))
        return self.head(x)


def _polar_signed(amp: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """amp·exp(j·atan2(b, a)) при amp любого знака (отрицательная амплитуда — сдвиг на π)."""
    unit = torch.complex(a, b) * torch.rsqrt(a.square() + b.square() + 1e-12)
    return amp * unit


class CGHNet(nn.Module):
    """3D-CGH-Net: Y-net (общий кодировщик, ветви амплитуды и фазы) + блок переноса.

    Доработки R08 (у варианта статьи все выключены): amp_skip — амплитуда ветви через пропуск
    амплитуды цели; refine_base — кодировщик в плоскости ДОЭ; gs_iters — итерации GS после
    блока переноса; gd_iters — обучаемые шаги спуска по сдаваемому loss внутри сети;
    pass2_base — второй проход по восстановлениям; interleave — чередование пикселей.
    """

    def __init__(
        self,
        base: int = 32,
        depth: int = 4,
        interleave: int = 1,
        amp_skip: bool = False,
        refine_base: int = 0,
        refine_depth: int = 2,
        gs_iters: int = 0,
        gd_iters: int = 0,
        gd_step: float = 0.1,
        gd_kind: str = "norm",
        gd_first_order: bool = False,
        pass2_base: int = 0,
        image_power: int = IMAGE_POWER,
    ):
        super().__init__()
        self.image_power = check_power(image_power)
        self.planes = planes = len(TASK.distances)
        self.r = r = interleave
        self.amp_skip = amp_skip
        self.gs_iters = gs_iters
        self.gd_iters = gd_iters
        if gd_kind not in ("norm", "adam"):
            raise ValueError(f"gd_kind — norm или adam, получено {gd_kind}")
        self.gd_kind = gd_kind
        if gd_iters:
            self.gd_log_step = nn.Parameter(torch.full((gd_iters,), float(gd_step)).log())
        chans = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(planes * r * r, chans[0])])
        self.enc.extend(_block(chans[i], chans[i + 1]) for i in range(depth))
        self.amp_dec = _Decoder(chans, (2 if amp_skip else 1) * planes * r * r)
        self.phase_dec = _Decoder(chans, 2 * planes * r * r)
        self.refiner = _Trunk(3, 2, refine_base, refine_depth) if refine_base > 0 else None
        self.pass2 = _Trunk(2 * planes, 2 * planes, pass2_base, depth) if pass2_base > 0 else None
        self.register_buffer("chirp", chirps(), persistent=False)

    def plane_fields(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.pixel_unshuffle(x, self.r) if self.r > 1 else x
        skips = []
        for i, block in enumerate(self.enc):
            h = block(h if i == 0 else nn.functional.max_pool2d(h, 2))
            skips.append(h)
        amp_out, phase_out = self.amp_dec(skips), self.phase_dec(skips)
        if self.r > 1:
            amp_out = nn.functional.pixel_shuffle(amp_out, self.r)
            phase_out = nn.functional.pixel_shuffle(phase_out, self.r)
        a, b = phase_out.split(self.planes, dim=1)
        if self.amp_skip:
            g, res = amp_out.split(self.planes, dim=1)
            target = unit_energy(target_amplitude(x, self.image_power))
            amp = target * 2 * torch.sigmoid(g) + res
        else:
            amp = nn.functional.softplus(amp_out)
        return _polar_signed(amp, a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fields = self.plane_fields(x)
        total = backpropagate(fields, self.chirp).sum(dim=-3)
        if self.pass2 is not None:
            first = torch.polar(torch.ones_like(total.real), torch.atan2(total.imag, total.real))
            k = propagate(first, self.chirp)
            image = k.real.square() + k.imag.square()
            image = image / image.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            re, im = self.pass2(torch.cat([x, image], dim=1)).split(self.planes, dim=1)
            fields = fields + torch.complex(re, im)
            total = backpropagate(fields, self.chirp).sum(dim=-3)
        if self.refiner is not None:
            unit = total / total.abs().clamp_min(1e-12)
            m = total.abs() / total.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            delta = self.refiner(torch.stack([unit.real, unit.imag, m - 1], dim=1))
            phase = torch.atan2(unit.imag + delta[:, 1], unit.real + delta[:, 0])
        else:
            phase = torch.atan2(total.imag, total.real)
        if self.gs_iters:
            amp = fields.abs()
            for _ in range(self.gs_iters):
                k = propagate(torch.polar(torch.ones_like(phase), phase), self.chirp)
                unit = k * torch.rsqrt(k.real.square() + k.imag.square() + 1e-12)
                total = backpropagate(amp * unit, self.chirp).sum(dim=-3)
                phase = torch.atan2(total.imag, total.real)
        if self.gd_kind == "adam":
            return self._adam_steps(phase, x)
        for k in range(self.gd_iters):
            p, g = self._grad(phase, x)
            g = g * torch.rsqrt(g.square().mean(dim=(-2, -1), keepdim=True) + 1e-20)
            phase = (p - self.gd_log_step[k].exp() * g).detach()
        return phase

    def _grad(self, phase: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(фаза, ∂loss/∂фаза) для шагов спуска внутри сети (инференс, без графа второго
        порядка); loss — Σ по плоскостям 1 − Пирсон, как у дооптимизации."""
        with torch.enable_grad():
            p = phase.detach().requires_grad_(True)
            amp = propagate(torch.polar(torch.ones_like(p), p), self.chirp).abs()
            loss = plane_losses(amp, x, "pearson", self.image_power).sum()
            (g,) = torch.autograd.grad(loss, p)
        return p.detach(), g

    def _adam_steps(self, phase: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = v = torch.zeros_like(phase)
        for k in range(self.gd_iters):
            p, g = self._grad(phase, x)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g.square()
            m_hat, v_hat = m / (1 - b1 ** (k + 1)), v / (1 - b2 ** (k + 1))
            step = self.gd_log_step[k].exp() * m_hat / ((v_hat + 1e-30).sqrt() + eps)
            phase = (p - step).detach()
        return phase
