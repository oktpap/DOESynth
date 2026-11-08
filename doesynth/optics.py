"""Прямая модель по условию задачи: дифракция Френеля с одним БПФ.

K(x, y) = exp(jkz)/(jλz) · exp(jπ(x²+y²)/λz) · FFT[H(u, v) · exp(jπ(u²+v²)/λz)].

Внешний чирп exp(jπ(x²+y²)/λz) и exp(jkz) по модулю равны единице, а 1/(λz) и нормировка FFT
меняют |K| только на постоянный по плоскости множитель, который уходит при нормировке
«максимум → 255». Поэтому для синтеза и метрик эти множители опускаются. FFT центрированный
и унитарный (norm="ortho"), координаты отсчётов u = (n − N/2)·Δx.

Восстановленное изображение — интенсивность |K|² (IMAGE_POWER = 2), под неё обучены все сети.
"""

from dataclasses import dataclass

import torch

WAVELENGTH = 532e-9
PIXEL = 10e-6
N = 128
DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)

# Восстановленное изображение = |K|^IMAGE_POWER. 2 — интенсивность (решение команды).
IMAGE_POWER = 2


def check_power(power: int) -> int:
    if power not in (1, 2):
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
    """Поле ДОЭ [..., N, N] → поля в плоскостях [..., P, N, N] (без внешних множителей)."""
    return cfft2(field.unsqueeze(-3) * chirp)


def backpropagate(fields: torch.Tensor, chirp: torch.Tensor) -> torch.Tensor:
    """Обратный к propagate для каждой плоскости: [..., P, N, N] → [..., P, N, N] на ДОЭ."""
    return cifft2(fields) * chirp.conj()


def reconstruct(phase: torch.Tensor, chirp: torch.Tensor) -> torch.Tensor:
    """Фаза ДОЭ [..., N, N] → амплитуды |K| [..., P, N, N] при освещении плоской волной."""
    return propagate(torch.polar(torch.ones_like(phase), phase), chirp).abs()


def reconstruct_at(phase: torch.Tensor, z: float, setup: Setup = TASK) -> torch.Tensor:
    """|K| на произвольном расстоянии z (м) — для просмотра объёма между плоскостями."""
    one = Setup(setup.n, setup.wavelength, setup.pixel, (z,))
    return reconstruct(phase, chirps(one, device=phase.device))[0]
