"""R08: 3D-CGH-Net (Rymov et al., Opt. Lasers Eng. 184 (2025) 108645, proto/3D-CGH-Net.pdf)
в постановке task.pdf, и её доработки. Код эксперимента; в core.net не входит.

Модель статьи (разд. 2.1–2.2, рис. 1–2): U-Net с общим кодировщиком и двумя ветвями декодера
(Y-net): одна ветвь выдаёт амплитуду, другая — фазу поля в каждой плоскости восстановления
(по P каналов). Блок переноса собирает поле плоскости A_p·exp(jψ_p), переносит каждую плоскость
на ДОЭ своим обратным оператором (у нас — точный обратный к формуле с одним БПФ, optics),
поля складываются, ДОЭ — фаза суммы. Loss (2) — 1 − Пирсон по всему объёму сразу (сумма по
i, j и n в числителе и знаменателе), а не сумма Пирсонов плоскостей. Пиксельное чередование
(interleave, рис. 3) — space-to-depth на входе и обратно на выходе ветвей.

Опции (у статьи все выключены):
- amp_skip — предложение авторов из разд. 5: амплитуда ветви комбинируется с амплитудой цели
  через пропуск со входа: A_p = A_цели·2σ(g_p) + r_p, головы нулевые, так что при инициализации
  A_p = A_цели (√T для |K|², нормировка на энергию ДОЭ — как у GS и superposition_init) и модель =
  суперпозиция с фазами плоскостей из фазовой ветви. Это не классическая superposition_init: там
  фазы — случайный диффузор, здесь на старте — гладкий выход сети, и на старте КК хуже
  (0,197 против 0,258 на 20 объектах, ревью); выученные фазы — результат обучения;
- refine_base > 0 — малый кодировщик в плоскости ДОЭ (как у core.net.SuperposNet) с нулевой
  головой: правит комплексную сумму до чисто фазового ДОЭ, при нулевой правке — ровно arg суммы;
- gs_iters > 0 — столько итераций проекции global GS после блока переноса, без обучаемых весов,
  цель проекции — амплитуды |A_p| самой сети: сеть предлагает поля плоскостей, итерации
  согласуют их с ограничением «только фаза»;
- gd_iters > 0 — столько шагов градиентного спуска по сдаваемому loss (Σ по плоскостям
  1 − Пирсон(|K|^power, T)) внутри сети, после блока переноса: φ ← φ − α_k·g/rms(g), g = ∂L/∂φ
  через прямую модель, α_k — обучаемые (по одному на шаг, старт 0,1 рад). При обучении градиент
  идёт и через шаги (create_graph): сеть учится выдавать старт, хороший именно для этих шагов, —
  в отличие от гибрида «сеть + Adam», где сеть о шагах не знает;
- gd_kind = "adam" — те же шаги, но по правилу Adam (β = 0,9 / 0,999, ε = 1e-8, поправка
  смещения), lr каждого шага обучаемый (старт gd_step): нормировка по каждому пикселю, а не по
  полю. До обучения gd_iters шагов при gd_step = lr в точности равны гибриду «сеть + Adam»
  (methods.adam), так что обучение через шаги проверяется против гибрида при равных затратах;
- gd_first_order — при обучении направление шага считается без графа (как first-order MAML):
  dφ_k+1/dφ_k = I, сеть получает ∂L/∂φ после шагов напрямую, обучаемые lr шагов — свой градиент.
  Повод: с градиентом второго порядка через поэлементную нормировку Adam (m/√v, при малых
  градиентах фазы множитель 1/√v огромен) обучение разрушало модель (adam8, adam8_ft). На
  инференсе фаза та же, что без этой опции. Не помогло (adam8_fo): отображение «старт → 8 шагов
  Adam» почти кусочно-постоянное (шаг ≈ lr·знак градиента в каждом пикселе);
- pass2_base > 0 — второй проход: ДОЭ первого прохода переносится в плоскости, вторая сеть (U-Net
  core.net._Trunk) по целям и восстановлениям |K_p|² (нормированным на среднее) выдаёт поправку
  полей плоскостей ΔF_p (Re, Im), поля F_p + ΔF_p снова идут в блок переноса. Голова нулевая:
  на старте модель = первый проход. Цель — главное слабое место блока переноса статьи: сеть не
  видит, что дала суперпозиция полей (разбор потерь первой волны), а здесь видит один раз.

Модели регистрируются в core.net.MODELS под именем «cghnet» при импорте модуля, поэтому
core.net.load_model читает их веса (импортируйте этот модуль до load_model).
"""

import torch
from torch import nn

from doetrain.core import net
from doetrain.core.methods import plane_losses, unit_energy
from doetrain.core.net import _block
from doetrain.core.optics import (
    IMAGE_POWER,
    TASK,
    backpropagate,
    check_power,
    chirps,
    propagate,
    target_amplitude,
)


class _Decoder(nn.Module):
    """Ветвь декодера U-Net над общим кодировщиком: билинейный апсемплинг + свёртка 1×1
    (как core.net._Trunk — без шахматных артефактов ConvTranspose)."""

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


class CGHNet(nn.Module):
    """3D-CGH-Net: Y-net (общий кодировщик, ветви амплитуды и фазы) + блок переноса."""

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
        self.gd_first_order = gd_first_order
        if gd_iters:  # параметр только при gd_iters > 0: веса прежних вариантов читаются как есть
            self.gd_log_step = nn.Parameter(torch.full((gd_iters,), float(gd_step)).log())
        chans = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(planes * r * r, chans[0])])
        self.enc.extend(_block(chans[i], chans[i + 1]) for i in range(depth))
        self.amp_dec = _Decoder(chans, (2 if amp_skip else 1) * planes * r * r)
        self.phase_dec = _Decoder(chans, 2 * planes * r * r)
        if amp_skip:  # A_p = A_цели при инициализации
            nn.init.zeros_(self.amp_dec.head.weight)
            nn.init.zeros_(self.amp_dec.head.bias)
        self.refiner = None
        if refine_base > 0:
            self.refiner = net._Trunk(3, 2, refine_base, refine_depth)
            nn.init.zeros_(self.refiner.head.weight)
            nn.init.zeros_(self.refiner.head.bias)
        self.pass2 = None
        if pass2_base > 0:
            self.pass2 = net._Trunk(2 * planes, 2 * planes, pass2_base, depth)
            nn.init.zeros_(self.pass2.head.weight)
            nn.init.zeros_(self.pass2.head.bias)
        self.register_buffer("chirp", chirps(), persistent=False)

    def plane_fields(self, x: torch.Tensor) -> torch.Tensor:
        """[B, P, N, N] цели T → комплексные поля плоскостей [B, P, N, N] (выход двух ветвей)."""
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
        """[B, P, N, N] → фаза ДОЭ [B, N, N]."""
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
            phase = p - self.gd_log_step[k].exp() * g
            phase = phase if self.training else phase.detach()
        return phase

    def _grad(self, phase: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(фаза как узел графа, ∂loss/∂фаза): loss — как у methods.adam (Σ по плоскостям и
        батчу; задачи батча независимы); при обучении градиент дифференцируем (create_graph)."""
        second = self.training and not self.gd_first_order
        with torch.enable_grad():
            p = phase if second else phase.detach().requires_grad_(True)
            amp = propagate(torch.polar(torch.ones_like(p), p), self.chirp).abs()
            loss = plane_losses(amp, x, "pearson", self.image_power).sum()
            (g,) = torch.autograd.grad(loss, p, create_graph=second)
        # первый порядок: шаг идёт от исходной фазы (граф сети сохраняется), направление — константа
        return (phase if self.training else p), g

    def _adam_steps(self, phase: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Развёрнутый torch.optim.Adam (betas 0,9/0,999, eps 1e-8) с обучаемым lr шагов."""
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = v = torch.zeros_like(phase)
        for k in range(self.gd_iters):
            p, g = self._grad(phase, x)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g.square()
            m_hat, v_hat = m / (1 - b1 ** (k + 1)), v / (1 - b2 ** (k + 1))
            # +1e-30 под корнем: у √ в нуле бесконечная производная (второй порядок при обучении)
            phase = p - self.gd_log_step[k].exp() * m_hat / ((v_hat + 1e-30).sqrt() + eps)
            phase = phase if self.training else phase.detach()
        return phase


def _polar_signed(amp: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """amp·exp(j·atan2(b, a)) при amp любого знака (отрицательная амплитуда — сдвиг фазы на π)."""
    unit = torch.complex(a, b) * torch.rsqrt(a.square() + b.square() + 1e-12)
    return amp * unit


net.MODELS["cghnet"] = CGHNet


def volume_pearson_loss(amp: torch.Tensor, targets: torch.Tensor, power: int) -> torch.Tensor:
    """Loss (2) статьи: 1 − Пирсон между объёмами [B, P, N, N] (средние и суммы — по всем
    плоскостям сразу) → [B]. amp = |K|, изображение — |K|^power, targets — T."""
    img = amp if check_power(power) == 1 else amp.square()
    x, y = img.flatten(1), targets.flatten(1)
    x = x - x.mean(1, keepdim=True)
    y = y - y.mean(1, keepdim=True)
    r = (x * y).sum(1) / torch.sqrt(x.square().sum(1) * y.square().sum(1) + 1e-12)
    return 1 - r
