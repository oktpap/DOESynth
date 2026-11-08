"""Чтение входных изображений и запись 8-битных серых PNG.

Вход по условию — 8-битный серый PNG 128×128. Остальное (RGB, альфа, 16 бит, другой размер)
не отвергаем, а приводим к условию и сообщаем об этом предупреждением.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image

from doesynth.optics import N

NUMERIC_MODES = ("1", "L", "LA", "RGB", "RGBA", "I;16", "I;16B", "I;16L", "I", "F")
ALPHA_MODES = ("LA", "RGBA", "PA", "La", "RGBa")
IMAGE_SUFFIXES = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".gif", ".webp")


def to_u8_square(img: np.ndarray, n: int = N) -> np.ndarray:
    """Любое изображение → 8-битный оттенок серого n×n: центральный квадрат и сглаженное сжатие.
    Прозрачность накладываем на чёрный фон: для голограммы прозрачное место — «нет света»."""
    from skimage.color import rgb2gray
    from skimage.transform import resize

    x = np.asarray(img)
    if x.dtype == np.bool_:
        scale = 1.0
    elif np.issubdtype(x.dtype, np.integer):
        scale = float(np.iinfo(x.dtype).max)
    else:
        scale = 1.0 if x.max() <= 1.0 else float(x.max())
    x = x.astype(np.float64) / scale
    if x.ndim == 3 and x.shape[-1] in (2, 4):
        x = x[..., :-1] * x[..., -1:]
        if x.shape[-1] == 1:
            x = x[..., 0]
    if x.ndim == 3:
        x = rgb2gray(x)
    h, w = x.shape
    s = min(h, w)
    x = x[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    if x.shape != (n, n):
        x = resize(x, (n, n), anti_aliasing=True)
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)


def _read(img: Image.Image) -> np.ndarray:
    if img.mode == "I":  # int32; так PIL открывает 16-битные PNG и TIFF (и 8-битные TIFF)
        x = np.asarray(img).astype(np.float64)
        if 0 <= x.min() and x.max() <= 255:
            return x / 255.0
        return x / (65535.0 if 0 <= x.min() and x.max() <= 65535 else max(x.max(), 1.0))
    if img.mode in NUMERIC_MODES:
        return np.asarray(img)
    if img.mode in ALPHA_MODES or "transparency" in img.info:
        return np.asarray(img.convert("RGBA"))
    return np.asarray(img.convert("RGB"))


def load_image(source: str | Path | bytes, name: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Файл или байты → uint8 [N, N] и список предупреждений о приведении к условию."""
    if isinstance(source, bytes):
        stream, label = io.BytesIO(source), name or "изображение"
    else:
        stream, label = Path(source), name or Path(source).name
    try:
        with Image.open(stream) as img:
            img.seek(0)
            mode, size = img.mode, img.size
            transparent = mode in ALPHA_MODES or "transparency" in img.info
            x = _read(img)
    except Exception as e:  # любой нечитаемый файл
        raise ValueError(f"{label}: не удалось прочитать изображение ({e})") from e
    warnings = []
    if mode not in ("L", "1"):
        warnings.append(f"{label}: режим {mode} переведён в 8-битный оттенок серого")
    if transparent:
        warnings.append(f"{label}: прозрачные места считаются чёрными")
    if size != (N, N):
        w, h = size
        crop = " (центральный квадрат)" if w != h else ""
        warnings.append(f"{label}: размер {w}×{h} приведён к {N}×{N}{crop}")
    if mode == "L" and size == (N, N) and x.dtype == np.uint8:
        return x.copy(), warnings
    return to_u8_square(x, N), warnings


def png_bytes(image_u8: np.ndarray) -> bytes:
    """uint8 [H, W] → PNG: 8 бит, один канал (режим L)."""
    buf = io.BytesIO()
    image = Image.fromarray(np.ascontiguousarray(image_u8, np.uint8))  # 2D uint8 → режим L
    if image.mode != "L":
        raise ValueError(f"ожидалось одноканальное 8-битное изображение, получено {image.mode}")
    image.save(buf, format="PNG")
    return buf.getvalue()


def save_png(path: str | Path, image_u8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(image_u8))


def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_SUFFIXES and not Path(name).name.startswith(".")
