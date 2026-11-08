"""Самопроверка решения: аналитические эталоны физики, метрики, формат ДОЭ, сквозной синтез.

    python -m doesynth selftest

Проверки физики построены на независимых от кода ответах (явные константы условия, прямая
сумма Френеля, NumPy-декодер), поэтому ловят ошибки знака чирпа, начала координат, порядка
плоскостей, единиц и нормировки БПФ. Сквозная проверка сети идёт без дооптимизации: шаги Adam
исправили бы и сломанную модель.
"""

import io
from collections.abc import Callable

import numpy as np
import torch

from doesynth import metrics
from doesynth.optics import DISTANCES, TASK, backpropagate, chirps, propagate

LAM, DX, N = 532e-9, 10e-6, 128  # константы условия, записанные независимо от optics


def _explicit_lens(z: float) -> np.ndarray:
    """Собирающая линза f = z из явных констант условия, без chirps()."""
    u = (np.arange(N) - N // 2) * DX
    return np.exp(-1j * np.pi * (u[:, None] ** 2 + u[None, :] ** 2) / (LAM * z))


def check_constants() -> tuple[bool, str]:
    ok = (TASK.wavelength, TASK.pixel, TASK.n) == (LAM, DX, N) and DISTANCES == (
        24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3,
    )  # fmt: skip
    return ok, "λ = 532 нм, Δx = 10 мкм, N = 128, z = 24,1 / 25,3 / 26,5 / 27,7 мм"


def check_unitary() -> tuple[bool, str]:
    chirp = chirps(dtype=torch.complex128)
    g = torch.Generator().manual_seed(0)
    h = torch.polar(torch.ones(N, N, dtype=torch.float64), torch.rand(N, N, generator=g).double())
    k = propagate(h, chirp)
    energy = k.abs().square().sum(dim=(-2, -1))
    back = backpropagate(k, chirp)
    err = (back - h.expand_as(back)).abs().max().item()
    ok = torch.allclose(energy, torch.tensor(float(N * N)).double()) and err < 1e-10
    return ok, f"энергия сохраняется, ошибка обратного переноса {err:.1e}"


def check_matched_lens() -> tuple[bool, str]:
    """Линза f = z_p фокусирует в центр только своей плоскости p (порядок плоскостей)."""
    k = [propagate(torch.from_numpy(_explicit_lens(z)), chirps(dtype=torch.complex128))
         .abs().square() for z in DISTANCES]  # fmt: skip
    fracs = np.array([[(kp[q, 64, 64] / kp[q].sum()).item() for q in range(4)] for kp in k])
    ok = all(fracs[p, p] > 0.999 for p in range(4)) and all(
        fracs[p, q] < 0.1 for p in range(4) for q in range(4) if q != p
    )
    return ok, f"доля энергии в фокусе своей плоскости ≥ {fracs.diagonal().min():.4f}"


def check_lens_and_tilt() -> tuple[bool, str]:
    """Линза f = z₂ + наклон на 5 периодов по столбцам → пик в (64, 69) плоскости 2."""
    tilt = np.exp(1j * 2 * np.pi * 5 * np.arange(N) / N)[None, :]
    h = torch.from_numpy(_explicit_lens(DISTANCES[1]) * tilt)
    k = propagate(h, chirps(dtype=torch.complex128)).abs().square()[1]
    peak = divmod(int(torch.argmax(k)), N)
    return peak == (64, 69) and (k[64, 69] / k.sum()).item() > 0.999, f"пик в {peak}"


def check_direct_fresnel_sum() -> tuple[bool, str]:
    """|K| модели совпадает с прямой суммой Френеля Σ H·exp(jπ|x−u|²/λz)/N."""
    z = DISTANCES[0]
    rng = np.random.default_rng(1)
    h = np.exp(1j * rng.uniform(-np.pi, np.pi, (N, N)))
    k = propagate(torch.from_numpy(h), chirps(dtype=torch.complex128)).abs().numpy()[0]
    u = (np.arange(N) - N // 2) * DX
    err = 0.0
    for m1, m2 in ((64, 64), (70, 50), (10, 100)):
        x1, x2 = (m1 - N // 2) * LAM * z / (N * DX), (m2 - N // 2) * LAM * z / (N * DX)
        kern = np.exp(1j * np.pi * ((x1 - u)[:, None] ** 2 + (x2 - u)[None, :] ** 2) / (LAM * z))
        err = max(err, abs(abs((h * kern).sum()) / N - k[m1, m2]))
    return err < 1e-9, f"расхождение {err:.1e}"


def check_numpy_decoder() -> tuple[bool, str]:
    """Независимый NumPy-декодер (вход без центрирования) на 8-битном ДОЭ даёт тот же |K|."""
    rng = np.random.default_rng(3)
    h = np.exp(1j * metrics.u8_to_phase(rng.integers(0, 256, (N, N))))
    ours = propagate(torch.from_numpy(h), chirps(dtype=torch.complex128)).abs().numpy()
    u = (np.arange(N) - N // 2) * DX
    r2 = u[:, None] ** 2 + u[None, :] ** 2
    err = max(
        np.abs(np.abs(np.fft.fftshift(np.fft.fft2(h * np.exp(1j * np.pi * r2 / (LAM * z))))) / N
               - ours[p]).max()
        for p, z in enumerate(DISTANCES)
    )  # fmt: skip
    return err < 1e-12, f"расхождение {err:.1e}"


def check_metrics() -> tuple[bool, str]:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (N, N)).astype(np.float64)
    b = np.arange(0, 256, dtype=np.float64)
    mb, vb = b.mean(), b.var()
    want = (
        (mb**2 + metrics.C1)
        * (vb + metrics.C2)
        / ((1.25 * mb**2 + metrics.C1) * (1.25 * vb + metrics.C2))
    )  # SSIM пары (b, b/2) в замкнутой форме
    ok = (
        abs(metrics.pearson(a, a) - 1) < 1e-12
        and abs(metrics.ssim(a, a) - 1) < 1e-12
        and abs(metrics.pearson(a, 3 * a + 7) - 1) < 1e-12
        and abs(metrics.ssim(b, b / 2) - want) < 1e-12
    )
    return ok, "Пирсон и SSIM: тождества и значение в замкнутой форме"


def check_phase_codes() -> tuple[bool, str]:
    codes = np.arange(256, dtype=np.uint8)
    roundtrip = np.array_equal(metrics.phase_to_u8(metrics.u8_to_phase(codes)), codes)
    ends = (
        metrics.phase_to_u8(np.array([-np.pi]))[0] == 0
        and metrics.phase_to_u8(np.array([0.0]))[0] == 128
        and metrics.u8_to_phase(np.array([255]))[0] == np.pi
    )
    return roundtrip and ends, "0 ↔ −π, 255 ↔ +π, коды 0..255 без потерь"


def check_end_to_end() -> tuple[bool, str]:
    """Сеть по умолчанию без дооптимизации на примере; формат сдаваемого ДОЭ."""
    from PIL import Image

    from doesynth import catalog, pipeline
    from doesynth.imageio import is_image_name, load_image, png_bytes

    kits = sorted(p for p in (catalog.ROOT / "examples").iterdir() if p.is_dir())
    files = sorted(p for p in kits[0].iterdir() if is_image_name(p.name))[:4]
    targets = np.stack([load_image(p)[0] for p in files])
    pure = pipeline.run(targets, pipeline.NetSpec(finetune=False))
    full = pipeline.run(targets, pipeline.NetSpec())
    img = Image.open(io.BytesIO(png_bytes(full.doe_u8)))
    fmt = img.format == "PNG" and img.mode == "L" and img.size == (N, N)
    # одна сеть на этом примере (фото) по бенчмарку даёт Пирсон ≈ 0,3; случайная фаза — ≈ 0
    ok = fmt and pure.metrics["pearson_mean"] > 0.15 and full.metrics["ssim_mean"] > 0.3
    return ok, (
        f"{kits[0].name}: сеть SSIM {pure.metrics['ssim_mean']:.3f} / Пирсон "
        f"{pure.metrics['pearson_mean']:.3f} за {pure.seconds:.2f} с; с дооптимизацией "
        f"{full.metrics['ssim_mean']:.3f} / {full.metrics['pearson_mean']:.3f} за "
        f"{full.seconds:.2f} с; ДОЭ — PNG 128×128, 8 бит, серый"
    )


CHECKS: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
    ("параметры условия", check_constants),
    ("унитарность и обратный перенос", check_unitary),
    ("линза фокусирует только в свою плоскость", check_matched_lens),
    ("линза + наклон: пик в ожидаемом пикселе", check_lens_and_tilt),
    ("прямая сумма Френеля", check_direct_fresnel_sum),
    ("независимый NumPy-декодер 8-битного ДОЭ", check_numpy_decoder),
    ("метрики", check_metrics),
    ("коды ДОЭ", check_phase_codes),
    ("сквозной синтез", check_end_to_end),
]


def run() -> bool:
    results = []
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:  # самопроверка сообщает, а не падает
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append(ok)
        print(f"[{'OK' if ok else 'ОШИБКА'}] {name}: {detail}", flush=True)
    print("Самопроверка пройдена" if all(results) else "Самопроверка НЕ пройдена", flush=True)
    return all(results)
