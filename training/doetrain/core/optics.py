"""Прямая модель по условию (task.pdf, с. 2 и 5): Френель с одним БПФ.

K(x, y) = exp(jkz)/(jλz) · exp(jπ(x²+y²)/λz) · FFT[H(u, v) · exp(jπ(u²+v²)/λz)].

Внешний чирп exp(jπ(x²+y²)/λz) и exp(jkz) по модулю единица, а 1/(λz) и нормировка FFT меняют
|K| только постоянным по плоскости масштабом (модуль физического поля Френеля равен
NΔx²/(λz) · наш |K|), который уходит при нормировке «максимум → 255», — поэтому для синтеза
и метрик всё это опускается. FFT центрированный и унитарный (norm="ortho"), координаты отсчётов
u = (n − N/2)Δx. Шаг сетки восстановления λz/(NΔx) свой у каждой плоскости (10,0–11,5 мкм),
но в пикселях изображение всегда N×N — так и требует формула с одним БПФ. Оператор сверен
с реализациями других участников команды (вручную, разово: модуль лаборатории FresnelPhysics — бит-в-бит,
включая побайтное совпадение эталонных PNG; полная модель Holography — с точностью до
перечисленных множителей, ~1e-14; см. docs/decisions.md, 2026-11-07); ту же математику
постоянно закрепляет независимая NumPy-реализация в tests/test_optics.py.
"""

from dataclasses import dataclass

import torch

WAVELENGTH = 532e-9
PIXEL = 10e-6
N = 128
DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)

# Что считаем восстановленным изображением: |K|^IMAGE_POWER. Условие этого не уточняет.
# 2 — интенсивность |K|² (решение владельца 2026-11-06 22:09), 1 — модуль |K|. Переключатель
# один на весь проект: от него берут оценку, GS, Adam и loss обучения; веса сети запоминают,
# под какую конвенцию обучены. Метрики всегда пишутся в обеих конвенциях.
IMAGE_POWER = 2
POWERS = (1, 2)
POWER_LABEL = {1: "|K|", 2: "|K|²"}
POWER_KEY = {1: "K", 2: "K2"}  # для имён полей и столбцов


def check_power(power: int) -> int:
    if power not in POWERS:
        raise ValueError(f"изображение — |K| (1) или |K|² (2), получено {power}")
    return power


def target_amplitude(images: torch.Tensor, power: int = IMAGE_POWER) -> torch.Tensor:
    """Целевая амплитуда поля для изображения T: √T, если изображение — |K|², иначе T."""
    return images.sqrt() if check_power(power) == 2 else images


@dataclass(frozen=True)
class Setup:
    n: int = N
    wavelength: float = WAVELENGTH
    pixel: float = PIXEL
    distances: tuple[float, ...] = DISTANCES

    def fresnel_numbers(self) -> list[float]:
        """λz/(NΔx²): 1 — чирп дискретизован точно на пределе Найквиста у края апертуры."""
        return [self.wavelength * z / (self.n * self.pixel**2) for z in self.distances]


TASK = Setup()


def chirps(setup: Setup = TASK, device=None, dtype=torch.complex64) -> torch.Tensor:
    """Входные чирпы exp(jπ(u²+v²)/λz), форма [P, N, N]."""
    n = setup.n
    u = (torch.arange(n, dtype=torch.float64) - n // 2) * setup.pixel
    r2 = u[:, None] ** 2 + u[None, :] ** 2
    z = torch.tensor(setup.distances, dtype=torch.float64)[:, None, None]
    phase = torch.pi * r2[None] / (setup.wavelength * z)
    return torch.polar(torch.ones_like(phase), phase).to(device=device, dtype=dtype)


def cfft2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )


def cifft2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )


def propagate(field: torch.Tensor, chirp: torch.Tensor) -> torch.Tensor:
    """Поле ДОЭ [..., N, N] → поля в плоскостях [..., P, N, N] (без внешних фазовых множителей)."""
    return cfft2(field.unsqueeze(-3) * chirp)


def backpropagate(fields: torch.Tensor, chirp: torch.Tensor) -> torch.Tensor:
    """Обратный к propagate для каждой плоскости: [..., P, N, N] → [..., P, N, N] в плоскости ДОЭ."""
    return cifft2(fields) * chirp.conj()


def reconstruct(phase: torch.Tensor, chirp: torch.Tensor) -> torch.Tensor:
    """Фаза ДОЭ [..., N, N] → амплитуды |K| [..., P, N, N] при освещении плоской волной."""
    return propagate(torch.polar(torch.ones_like(phase), phase), chirp).abs()
