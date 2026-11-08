"""Чтение входных изображений и запись 8-битных PNG.

Вход по условию — 8-битный серый PNG 128×128. Остальное (RGB, альфа, 16 бит, другой размер)
не отвергаем, а приводим к условию и сообщаем об этом предупреждением.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from doetrain.core.data import to_u8_square
from doetrain.core.optics import N

NUMERIC_MODES = ("1", "L", "LA", "RGB", "RGBA", "I;16", "I;16B", "I;16L", "I", "F")
ALPHA_MODES = ("LA", "RGBA", "PA", "La", "RGBa")


def _read(img: Image.Image) -> np.ndarray:
    """Массив, который понимает to_u8_square: целые — в полном диапазоне типа, float ≤ 1."""
    if img.mode == "I":  # int32; так PIL открывает 16-битные TIFF и PNG
        x = np.asarray(img).astype(np.float64)
        return x / (65535.0 if 0 <= x.min() and x.max() <= 65535 else max(x.max(), 1.0))
    if img.mode in NUMERIC_MODES:
        return np.asarray(img)
    if img.mode in ALPHA_MODES or "transparency" in img.info:  # палитра или иное с альфой
        return np.asarray(img.convert("RGBA"))
    return np.asarray(img.convert("RGB"))  # P, CMYK, YCbCr, LAB, HSV


def load_image(path: str | Path, n: int = N) -> tuple[np.ndarray, list[str]]:
    """PNG/JPEG/TIFF → uint8 [n, n] и список предупреждений о приведении к условию."""
    path = Path(path)
    with Image.open(path) as img:
        mode, size = img.mode, img.size
        transparent = mode in ALPHA_MODES or "transparency" in img.info
        x = _read(img)
    warnings = []
    if mode not in ("L", "1"):
        warnings.append(f"{path.name}: режим {mode} переведён в 8-битный оттенок серого")
    if transparent:
        warnings.append(f"{path.name}: прозрачные места считаем чёрными (нет света)")
    if size != (n, n):
        w, h = size
        crop = " (центральный квадрат)" if w != h else ""
        warnings.append(f"{path.name}: размер {w}×{h} приведён к {n}×{n}{crop}")
    if mode == "L" and size == (n, n) and x.dtype == np.uint8:
        return x.copy(), warnings
    return to_u8_square(x, n), warnings


def load_images(paths, n: int = N) -> tuple[np.ndarray, list[str]]:
    """Ровно 4 пути (по числу плоскостей) → uint8 [4, n, n] и предупреждения."""
    paths = list(paths)
    if len(paths) != 4:
        raise ValueError(f"нужно 4 изображения, по одному на плоскость, получено {len(paths)}")
    images, warnings = [], []
    for p in paths:
        img, w = load_image(p, n)
        images.append(img)
        warnings += w
    return np.stack(images), warnings


def save_png(path: str | Path, image_u8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image_u8, np.uint8)).save(path)
