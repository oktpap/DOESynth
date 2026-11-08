"""Итерационные методы синтеза одной фазы под P плоскостей (baseline для нейросети).

targets — целевые АМПЛИТУДЫ [P, N, N] (≥ 0). Как из 8-битной картинки получить амплитуду,
решает вызывающий код: при изображении-интенсивности A = sqrt(T), при изображении-амплитуде A = T.
Все методы возвращают непрерывную фазу [N, N]; квантование в 8 бит — отдельный шаг.
"""

import torch

from doetrain.core.optics import IMAGE_POWER, backpropagate, propagate


def unit_energy(a: torch.Tensor) -> torch.Tensor:
    """Нормировка каждой плоскости к энергии поля ДОЭ (N² при |H| = 1 и унитарном FFT)."""
    n2 = a.shape[-1] * a.shape[-2]
    return a * torch.sqrt(n2 / (a.square().sum(dim=(-2, -1), keepdim=True) + 1e-12))


def random_phase(n: int, seed: int, device=None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return ((torch.rand(n, n, generator=g) * 2 - 1) * torch.pi).to(device)


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
    """Последовательный GS (ping-pong по плоскостям); одна итерация — проход по всем P плоскостям."""
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


def plane_losses(amp: torch.Tensor, targets: torch.Tensor, kind: str, power: int) -> torch.Tensor:
    """Loss каждой плоскости: amp — |K| [..., P, N, N] → [..., P]. power: 1 — изображение = |K|,
    2 — изображение = |K|²; targets в той же величине, что изображение (T), не амплитуда."""
    img = amp if power == 1 else amp.square()
    if kind == "pearson":
        return 1 - _pearson(img, targets)
    if kind == "mse_scaled":
        s = (img * targets).sum(dim=(-2, -1), keepdim=True) / (
            img.square().sum(dim=(-2, -1), keepdim=True) + 1e-12
        )
        err = (s * img - targets).square().sum(dim=(-2, -1))
        return err / targets.square().sum(dim=(-2, -1))
    raise ValueError(kind)


def loss_fn(amp: torch.Tensor, targets: torch.Tensor, kind: str, power: int) -> torch.Tensor:
    """Сумма plane_losses по всем плоскостям (и по батчу, если он есть)."""
    return plane_losses(amp, targets, kind, power).sum()


def adam(
    images, chirp, iters: int, phase0: torch.Tensor, *, kind="pearson", power=IMAGE_POWER, lr=0.05
) -> torch.Tensor:
    """Прямая градиентная оптимизация фазы (Adam) через прямую модель; images — T в [0, 1]."""
    phase = phase0.clone().requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        amp = propagate(torch.polar(torch.ones_like(phase), phase), chirp).abs()
        loss_fn(amp, images, kind, power).backward()
        opt.step()
    return phase.detach()
