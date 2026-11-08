"""Итерационные методы синтеза одной фазы под P плоскостей и дооптимизация фазы сети.

targets в GS — целевые АМПЛИТУДЫ [P, N, N] (√T при изображении-интенсивности);
images в Adam — сами изображения T ∈ [0, 1]. Все методы возвращают непрерывную фазу [N, N];
квантование в 8 бит — отдельный шаг (pipeline).
"""

import torch

from doesynth.optics import IMAGE_POWER, backpropagate, propagate

C1 = (0.01 * 255) ** 2
C2 = (0.03 * 255) ** 2


def unit_energy(a: torch.Tensor) -> torch.Tensor:
    """Нормировка каждой плоскости к энергии поля ДОЭ (N² при |H| = 1 и унитарном FFT)."""
    n2 = a.shape[-1] * a.shape[-2]
    return a * torch.sqrt(n2 / (a.square().sum(dim=(-2, -1), keepdim=True) + 1e-12))


def superposition_init(targets: torch.Tensor, chirp: torch.Tensor, seed: int) -> torch.Tensor:
    """Фаза суммы обратно распространённых целей со случайной фазой (неитерационный старт)."""
    a = unit_energy(targets)
    g = torch.Generator().manual_seed(seed)
    rnd = ((torch.rand(a.shape, generator=g) * 2 - 1) * torch.pi).to(a.device)
    return backpropagate(torch.polar(a, rnd), chirp).sum(0).angle()


def gs_global(targets, chirp, iters: int, phase0: torch.Tensor) -> torch.Tensor:
    """Многоплоскостной GS с усреднением (global GS): все плоскости проецируются параллельно,
    поля в плоскости ДОЭ суммируются, от суммы берётся фаза."""
    a = unit_energy(targets)
    phase = phase0.clone()
    for _ in range(iters):
        k = propagate(torch.polar(torch.ones_like(phase), phase), chirp)
        back = backpropagate(torch.polar(a, k.angle()), chirp)
        phase = back.sum(0).angle()
    return phase


def gs_sequential(targets, chirp, iters: int, phase0: torch.Tensor) -> torch.Tensor:
    """Последовательный GS (ping-pong по плоскостям); итерация — проход по всем P плоскостям."""
    a = unit_energy(targets)
    phase = phase0.clone()
    for _ in range(iters):
        for p in range(a.shape[0]):
            h = torch.polar(torch.ones_like(phase), phase)
            k = propagate(h, chirp[p : p + 1])[0]
            phase = backpropagate(torch.polar(a[p], k.angle())[None], chirp[p : p + 1])[0].angle()
    return phase


def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    y = y - y.mean(dim=(-2, -1), keepdim=True)
    num = (x * y).sum(dim=(-2, -1))
    return num / torch.sqrt(x.square().sum(dim=(-2, -1)) * y.square().sum(dim=(-2, -1)) + 1e-12)


def _ssim_global(img: torch.Tensor, targets: torch.Tensor, soft: float = 0.0) -> torch.Tensor:
    """SSIM условия по плоскостям, как его считает оценка: восстановление нормируется
    «максимум → 255», цель — T·255. Дифференцируем (без округления до 8 бит).
    soft > 0 — значение максимума точное, а градиент идёт через p-норму (p = soft): его получают
    все яркие пиксели, а не один."""
    peak = img.amax(dim=(-2, -1), keepdim=True)
    if soft > 0:
        ref = peak.detach().clamp_min(1e-12)  # p-норма от img/max — без переполнения float32
        pn = ref * (img / ref).pow(soft).mean(dim=(-2, -1), keepdim=True).pow(1 / soft)
        scale = ref / pn.detach().clamp_min(1e-12)
        peak = peak.detach() + (pn - pn.detach()) * scale
    b = img / peak.clamp_min(1e-12) * 255
    a = targets * 255
    ma, mb = a.mean(dim=(-2, -1)), b.mean(dim=(-2, -1))
    da = a - ma[..., None, None]
    db = b - mb[..., None, None]
    va, vb = da.square().mean(dim=(-2, -1)), db.square().mean(dim=(-2, -1))
    cov = (da * db).mean(dim=(-2, -1))
    return (2 * ma * mb + C1) * (2 * cov + C2) / ((ma**2 + mb**2 + C1) * (va + vb + C2))


def plane_losses(
    amp: torch.Tensor,
    targets: torch.Tensor,
    kind: str,
    power: int,
    ssim_weight: float = 1.0,
    soft: float = 0.0,
) -> torch.Tensor:
    """Loss каждой плоскости: amp — |K| [..., P, N, N] → [..., P]. power: 1 — изображение = |K|,
    2 — изображение = |K|²; targets — изображения T, не амплитуды.

    kind: «pearson» — 1 − Пирсон (как при обучении сетей); «pearson_ssim» — плюс
    ssim_weight·(1 − SSIM условия); «ssim» — только 1 − SSIM условия."""
    img = amp if power == 1 else amp.square()
    if kind == "pearson":
        return 1 - _pearson(img, targets)
    if kind == "pearson_ssim":
        ssim = _ssim_global(img, targets, soft)
        return (1 - _pearson(img, targets)) + ssim_weight * (1 - ssim)
    if kind == "ssim":
        return 1 - _ssim_global(img, targets, soft)
    raise ValueError(kind)


def adam(
    images,
    chirp,
    iters: int,
    phase0: torch.Tensor,
    *,
    kind: str = "pearson",
    power: int = IMAGE_POWER,
    lr: float = 0.05,
    ssim_weight: float = 1.0,
    soft: float = 0.0,
    callback=None,
) -> torch.Tensor:
    """Градиентная оптимизация фазы (Adam) через прямую модель; images — T в [0, 1].

    callback(step, phase) — для кривых качества в бенчмарке (вызывается после шага)."""
    phase = phase0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=lr)
    for step in range(iters):
        opt.zero_grad(set_to_none=True)
        amp = propagate(torch.polar(torch.ones_like(phase), phase), chirp).abs()
        plane_losses(amp, images, kind, power, ssim_weight, soft).sum().backward()
        opt.step()
        if callback is not None:
            callback(step + 1, phase.detach())
    return phase.detach()


# Дооптимизация по умолчанию (подобрана на выборке настройки, benchmark/README.md): первая
# часть шагов — Пирсон (структура изображений), остальные — Пирсон + SSIM условия с градиентом
# максимума через p-норму: SSIM нормирует восстановление на максимум, и яркие пики спекла
# делают восстановление тёмным; второй этап их гасит.
FINETUNE = {"split": 0.5, "lr1": 0.4, "lr2": 0.05, "ssim_weight": 1.0, "soft": 8.0}


def finetune(
    images,
    chirp,
    steps: int,
    phase0: torch.Tensor,
    *,
    power: int = IMAGE_POWER,
    split: float = FINETUNE["split"],
    lr1: float = FINETUNE["lr1"],
    lr2: float = FINETUNE["lr2"],
    ssim_weight: float = FINETUNE["ssim_weight"],
    soft: float = FINETUNE["soft"],
    callback=None,
) -> torch.Tensor:
    """Двухэтапная дооптимизация фазы: steps·split шагов Adam по Пирсону, затем остальные —
    по Пирсону + SSIM. callback(step, phase) получает сквозной номер шага."""
    k1 = round(steps * split)
    k2 = steps - k1
    phase = adam(images, chirp, k1, phase0, power=power, lr=lr1, callback=callback)
    if k2 > 0:
        cb = None if callback is None else (lambda k, p: callback(k1 + k, p))
        phase = adam(
            images, chirp, k2, phase, kind="pearson_ssim", power=power, lr=lr2,
            ssim_weight=ssim_weight, soft=soft, callback=cb,
        )  # fmt: skip
    return phase
