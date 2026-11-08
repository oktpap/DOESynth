"""Наборы из 4 целевых изображений (skimage.data и буквы).

VALIDATION — для выбора методов и гиперпараметров. HELD_OUT_TEST — имена отложенных картинок
для финальной проверки обобщения; при выборе методов их не загружаем. Набор временный.
"""

import numpy as np

VALIDATION = {
    "photos": ("camera", "astronaut", "coffee", "chelsea"),
    "docs": ("page", "text", "coins", "moon"),
    "letters": ("letter:A", "letter:B", "letter:C", "letter:D"),
    "mixed": ("shepp_logan_phantom", "horse", "clock", "cell"),
}
HELD_OUT_TEST = (
    "rocket", "brick", "grass", "gravel", "hubble_deep_field", "retina",
    "immunohistochemistry", "logo", "binary_blobs", "checkerboard", "colorwheel", "microaneurysms",
)  # fmt: skip
# skimage.data.cat() — та же картинка, что chelsea(): в наборы её не включать.


def to_u8_square(img: np.ndarray, n: int = 128) -> np.ndarray:
    """Любое изображение → 8-битный оттенок серого n×n: центральный квадрат и сглаженное сжатие.

    Прозрачность (серый + альфа и RGBA) одинаково накладываем на чёрный фон: для голограммы
    прозрачное место — «нет света».
    """
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
    if x.ndim == 3 and x.shape[-1] in (2, 4):  # серый + альфа, RGBA → на чёрном фоне
        x = x[..., :-1] * x[..., -1:]
        if x.shape[-1] == 1:
            x = x[..., 0]
    if x.ndim == 3:
        x = rgb2gray(x)
    h, w = x.shape
    s = min(h, w)
    x = x[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    x = resize(x, (n, n), anti_aliasing=True)
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)


def letter(ch: str, n: int = 128) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=int(n * 0.8))
    left, top, right, bottom = draw.textbbox((0, 0), ch, font=font)
    draw.text(((n - right - left) / 2, (n - bottom - top) / 2), ch, fill=255, font=font)
    return np.asarray(img, np.uint8)


def load(name: str, n: int = 128) -> np.ndarray:
    if name.startswith("letter:"):
        return letter(name.split(":", 1)[1], n)
    import skimage.data

    if name == "binary_blobs":  # без сида картинка при каждой загрузке новая
        return to_u8_square(skimage.data.binary_blobs(rng=0), n)
    return to_u8_square(getattr(skimage.data, name)(), n)


def validation_set(key: str, n: int = 128) -> np.ndarray:
    """[4, n, n] uint8."""
    return np.stack([load(name, n) for name in VALIDATION[key]])
