"""Метрики по условию (task.pdf, с. 3) и 8-битное представление ДОЭ и восстановлений (с. 6).

КК — коэффициент Пирсона по всему изображению. ИСС — глобальная формула со слайда
(одно окно на всё изображение); константы не заданы, берём общепринятые для 8 бит:
c₁ = (0,01·255)², c₂ = (0,03·255)². Оконный SSIM (skimage, гаусс σ = 1,5) — справочно.
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


def ssim_global(a: np.ndarray, b: np.ndarray, c1: float = C1, c2: float = C2) -> float:
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
    return (
        np.clip(np.rint(v / peak * 255), 0, 255).astype(np.uint8)
        if peak > 0
        else v.astype(np.uint8)
    )


def score(targets_u8: np.ndarray, recons_u8: np.ndarray) -> dict:
    """Матрицы 4×4 (исходное i × восстановленное j); диагональ — сдаваемые пары."""
    p = len(targets_u8)
    cc = np.array([[pearson(targets_u8[i], recons_u8[j]) for j in range(p)] for i in range(p)])
    ss = np.array([[ssim_global(targets_u8[i], recons_u8[j]) for j in range(p)] for i in range(p)])
    return {
        "cc": np.diag(cc).tolist(),
        "ssim": np.diag(ss).tolist(),
        "ssim_win": [ssim_windowed(targets_u8[i], recons_u8[i]) for i in range(p)],
        "cc_matrix": cc.tolist(),
        "ssim_matrix": ss.tolist(),
    }
