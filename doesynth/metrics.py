"""Метрики условия и 8-битные представления ДОЭ и восстановлений.

Пирсон — коэффициент корреляции по всему изображению 128×128.
SSIM — формула со слайда 3 условия (одно окно на всё изображение), c₁ = (0,01·255)²,
c₂ = (0,03·255)². Оконный SSIM (гауссово окно σ = 1,5, как в scikit-image) — справочно.
Метрики считаются по 8-битным изображениям: исходное ↔ восстановление из 8-битного ДОЭ.
"""

import numpy as np

C1 = (0.01 * 255) ** 2
C2 = (0.03 * 255) ** 2


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def ssim(a: np.ndarray, b: np.ndarray, c1: float = C1, c2: float = C2) -> float:
    """SSIM по формуле условия (глобальный: средние и дисперсии по всему изображению)."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float((2 * ma * mb + c1) * (2 * cov + c2) / ((ma**2 + mb**2 + c1) * (va + vb + c2)))


def ssim_windowed(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            np.asarray(a, np.float64),
            np.asarray(b, np.float64),
            data_range=255,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def phase_to_u8(phase: np.ndarray) -> np.ndarray:
    """Фаза (любая) → код 0..255, где 0 ↔ −π, 255 ↔ +π (условие, с. 6)."""
    wrapped = np.angle(np.exp(1j * np.asarray(phase, np.float64)))
    return np.clip(np.rint((wrapped + np.pi) / (2 * np.pi) * 255), 0, 255).astype(np.uint8)


def u8_to_phase(code: np.ndarray) -> np.ndarray:
    return np.asarray(code, np.float64) / 255 * 2 * np.pi - np.pi


def image_to_u8(values: np.ndarray) -> np.ndarray:
    """Восстановление → 8 бит: линейно, максимум → 255 (без подгонки под цель)."""
    v = np.asarray(values, np.float64)
    peak = v.max()
    if peak <= 0:
        return np.zeros(v.shape, np.uint8)
    return np.clip(np.rint(v / peak * 255), 0, 255).astype(np.uint8)


def score(targets_u8: np.ndarray, recons_u8: np.ndarray) -> dict:
    """Пары «исходное k ↔ восстановление k» и их средние; матрица Пирсона 4×4 — для разделения
    плоскостей (диагональ минус среднее вне диагонали)."""
    p = len(targets_u8)
    cc = np.array([[pearson(targets_u8[i], recons_u8[j]) for j in range(p)] for i in range(p)])
    ss = [ssim(targets_u8[i], recons_u8[i]) for i in range(p)]
    sw = [ssim_windowed(targets_u8[i], recons_u8[i]) for i in range(p)]
    diag = np.diag(cc)
    off = cc[~np.eye(p, dtype=bool)]
    return {
        "ssim": [float(v) for v in ss],
        "pearson": [float(v) for v in diag],
        "ssim_windowed": [float(v) for v in sw],
        "ssim_mean": float(np.mean(ss)),
        "pearson_mean": float(np.mean(diag)),
        "ssim_windowed_mean": float(np.mean(sw)),
        "separation": float(diag.mean() - off.mean()),
    }
