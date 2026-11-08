"""Наборы для обучения, генерируемые на лету: 4 изображения 128×128 на пример.

Правильные фазы не нужны (обучение через прямую модель), поэтому данных бесконечно много.
Какие картинки дадут на защите, заранее неизвестно, поэтому генератор покрывает как можно
больше видов изображений и их сочетаний в четвёрке:

- виды (KINDS): фото из каталога (`photos_dir`, у нас — COCO val2017); текст — латиница,
  кириллица и цифры разными шрифтами: одиночный знак, слова, страница документа; фигуры
  (заливка и контур, сглаженные и «пиксельные»); тонкие линии (ломаные, кривые, окружности,
  спирали, штриховка); точки и массивы пятен; силуэты; двоичные коды (QR-подобные блоки,
  штрихкоды, шахматка); текстуры; градиенты и гладкий шум со степенным спектром;
- режимы четвёрки (MODES): независимые плоскости; все четыре одного вида, как наборы
  validation; «сцена» — одна картинка со сдвигом, масштабом и поворотом по плоскостям или копиями.
  По умолчанию — только независимые: в R02 режимы «все одного вида» и «сцена» не дали выигрыша
  ни на одном наборе (experiments/R02-data/report.md), их включают через `modes`;
  «объём» (volume) — сечения одного трёхмерного объекта на глубинах плоскостей
  (doetrain.train.volume, параметры — `volume`); по умолчанию выключен;
- аугментации: отражения и повороты на 90° (кроме текста), инверсия, гамма, лёгкое размытие,
  яркость и контраст. Пирсон в loss не меняется при T → aT + b (a > 0), поэтому яркость
  и контраст задачу не меняют — только учат сеть не зависеть от них на входе.

Почти однотонные картинки (размах < MIN_RANGE) перегенерируем: плоскость с такой целью
ничему не учит. Порог — по размаху, а не по σ: у одной точки на чёрном σ ≈ 2/255.
Доли видов — `weights` (по умолчанию DEFAULT_WEIGHTS и PHOTO_SHARE фото, если задан каталог),
доли режимов — `modes`. Картинки из doetrain.core.data (validation и отложенный test)
сюда не попадают.
"""

import importlib.util
import string
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from doetrain.core.optics import TASK, N
from doetrain.train.volume import VolumeGenerator

KINDS = (
    "photo", "text", "shapes", "lines", "dots", "blobs", "code", "texture", "gradient", "noise",
)  # fmt: skip
MODES = ("independent", "themed", "scene", "volume")
# Смесь по умолчанию без фото; с каталогом фото добавляется долей PHOTO_SHARE
DEFAULT_WEIGHTS = {
    "text": 1.5, "shapes": 1.0, "lines": 1.0, "dots": 0.7, "blobs": 0.8,
    "code": 0.6, "texture": 0.8, "gradient": 0.4, "noise": 0.6,
}  # fmt: skip
PHOTO_SHARE = 0.35
# Выбор R02 по правилу, зафиксированному до оценки: wide_indep (только независимые четвёрки)
DEFAULT_MODES = {"independent": 1.0}
MIN_RANGE = 8  # минимальный размах яркости картинки, из 255
NO_FLIP = {"text"}  # зеркальные буквы на защите не встретятся
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LATIN = string.ascii_letters
CYRILLIC = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
DIGITS = string.digits
SYMBOLS = "#@&%?!+=№"
# Шрифты из matplotlib (уже в зависимостях обучения): DejaVu и STIX — с кириллицей
FONT_FILES = (
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSerif.ttf",
    "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf", "DejaVuSansMono.ttf",
    "DejaVuSansMono-Bold.ttf", "STIXGeneral.ttf", "STIXGeneralBol.ttf", "STIXGeneralItalic.ttf",
    "cmr10.ttf", "cmss10.ttf",
)  # fmt: skip


def _to_u8(v: np.ndarray) -> np.ndarray:
    """[0, 1] → uint8."""
    return np.clip(np.rint(v * 255), 0, 255).astype(np.uint8)


def _stretch(v: np.ndarray) -> np.ndarray:
    v = v - v.min()
    return v / (v.max() + 1e-12)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _power_noise(rng: np.random.Generator, n: int, alpha: float) -> np.ndarray:
    """Шум со спектром амплитуды ∝ f^−alpha, [0, 1]; у естественных изображений alpha ≈ 1."""
    f = np.fft.fftfreq(n)
    r = np.hypot(f[:, None], f[None, :])
    r[0, 0] = 1.0
    spectrum = r**-alpha * np.exp(2j * np.pi * rng.random((n, n)))
    spectrum[0, 0] = 0
    return _stretch(np.fft.ifft2(spectrum).real)


# ---------- Шрифты и текст ----------


@cache
def _fonts() -> tuple[tuple[str | None, bool, str], ...]:
    """(путь к TTF, есть ли кириллица, доступные символы из SYMBOLS); None — встроенный
    шрифт PIL (только латиница и цифры).

    Шрифт берём, только если в нём есть вся латиница и цифры: иначе вместо букв — пустые
    квадраты. Кириллицу и символы проверяем по таблице символов шрифта."""
    fonts: list[tuple[str | None, bool, str]] = [(None, False, "")]
    spec = importlib.util.find_spec("matplotlib")
    if spec is None or not spec.submodule_search_locations:
        return tuple(fonts)
    from fontTools.ttLib import TTFont

    root = Path(spec.submodule_search_locations[0]) / "mpl-data" / "fonts" / "ttf"
    for name in FONT_FILES:
        path = root / name
        if not path.is_file():
            continue
        cmap = TTFont(path, lazy=True).getBestCmap() or {}
        if all(ord(c) in cmap for c in LATIN + DIGITS):
            cyrillic = all(ord(c) in cmap for c in CYRILLIC)
            fonts.append((str(path), cyrillic, "".join(c for c in SYMBOLS if ord(c) in cmap)))
    return tuple(fonts)


@cache
def _font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size) if path is None else ImageFont.truetype(path, size)


def _font_and_alphabet(rng: np.random.Generator) -> tuple[str | None, str]:
    fonts = _fonts()
    path, cyrillic, symbols = fonts[int(rng.integers(len(fonts)))]
    if cyrillic and rng.random() < 0.6:
        return path, CYRILLIC + DIGITS + symbols
    return path, LATIN + DIGITS + symbols


def _words(rng: np.random.Generator, alphabet: str, count: int, longest: int) -> str:
    chars = list(alphabet)
    return " ".join(
        "".join(rng.choice(chars, int(rng.integers(1, longest + 1)))) for _ in range(count)
    )


def _text(rng: np.random.Generator, n: int) -> np.ndarray:
    path, alphabet = _font_and_alphabet(rng)
    style = rng.choice(3, p=[0.35, 0.4, 0.25])  # одиночный знак, слова, страница документа
    if style == 2:
        bg, fg = int(rng.integers(170, 256)), int(rng.integers(0, 100))
    else:
        bg, fg = int(rng.integers(0, 70)), int(rng.integers(130, 256))
    img = Image.new("L", (n, n), bg)
    draw = ImageDraw.Draw(img)
    if style == 0:
        font = _font(path, int(n * rng.uniform(0.45, 0.95)))
        ch = str(rng.choice(list(alphabet)))
        x0, y0, x1, y1 = draw.textbbox((0, 0), ch, font=font)
        x = (n - (x1 - x0)) / 2 - x0 + rng.normal(0, 0.08 * n)
        y = (n - (y1 - y0)) / 2 - y0 + rng.normal(0, 0.08 * n)
        draw.text((x, y), ch, fill=fg, font=font)
    elif style == 1:
        for _ in range(int(rng.integers(1, 5))):
            font = _font(path, int(_log_uniform(rng, n / 10, n / 2.2)))
            text = _words(rng, alphabet, 1, 8)
            x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
            w, h = x1 - x0, y1 - y0
            x = rng.uniform(-0.15 * w, max(n - 0.85 * w, -0.15 * w + 1)) - x0
            y = rng.uniform(-0.15 * h, max(n - 0.85 * h, -0.15 * h + 1)) - y0
            draw.text((x, y), text, fill=int(rng.integers(130, 256)), font=font)
    else:  # строки мелкого текста, как скан страницы
        size = int(rng.integers(7, 17))
        font = _font(path, size)
        step = size * rng.uniform(1.15, 1.7)
        x, y = rng.uniform(1, 12), rng.uniform(-size / 2, size)
        while y < n:
            line = _words(rng, alphabet, int(rng.integers(2, 12)), 9)
            draw.text((x, y), line, fill=fg, font=font)
            y += step * (2 if rng.random() < 0.1 else 1)  # иногда пустая строка
    v = np.asarray(img, np.float64) / 255
    if style == 2 and rng.random() < 0.6:  # неравномерная подсветка, как у скана
        yy, xx = np.mgrid[0:n, 0:n] / n
        a = rng.uniform(0, 2 * np.pi)
        v = v * (1 - rng.uniform(0.1, 0.5) * _stretch(xx * np.cos(a) + yy * np.sin(a)))
    if rng.random() < 0.2:  # небольшой поворот надписи
        img = Image.fromarray(_to_u8(v)).rotate(
            rng.uniform(-20, 20), Image.Resampling.BILINEAR, fillcolor=bg
        )
        return np.asarray(img, np.uint8)
    return _to_u8(v)


# ---------- Фигуры и линии (рисуем в 4× и уменьшаем: сглаженные края) ----------


def _polygon(rng: np.random.Generator, cx: float, cy: float, r: float) -> list[tuple]:
    """Многоугольник: правильный, звезда или «клякса» с неровными радиусами."""
    k = int(rng.integers(3, 10))
    style = rng.integers(3)
    if style == 0:
        angles = np.arange(k) * 2 * np.pi / k + rng.uniform(0, 2 * np.pi)
        radii = np.full(k, r)
    elif style == 1:
        k = int(rng.integers(4, 9))
        angles = np.arange(2 * k) * np.pi / k + rng.uniform(0, 2 * np.pi)
        radii = np.where(np.arange(2 * k) % 2 == 0, r, r * rng.uniform(0.3, 0.7))
    else:
        angles = np.sort(rng.uniform(0, 2 * np.pi, k + 2))
        radii = r * rng.uniform(0.5, 1, k + 2)
    return list(zip(cx + radii * np.cos(angles), cy + radii * np.sin(angles), strict=True))


def _shapes(rng: np.random.Generator, n: int) -> np.ndarray:
    s = 4 if rng.random() < 0.6 else 1  # сглаженные края или «пиксельные»
    m = n * s
    img = Image.new("L", (m, m), int(rng.integers(0, 80)))
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(1, 8))):
        fill = int(rng.integers(60, 256))
        cx, cy = rng.uniform(-0.1, 1.1, 2) * m
        r = m * _log_uniform(rng, 0.04, 0.45)
        outline = rng.random() < 0.25
        width = max(1, int(s * rng.uniform(1, 4)))
        style = rng.integers(3)
        if style == 0:  # эллипс
            a = np.exp(rng.uniform(-0.7, 0.7))
            box = [cx - r * a, cy - r / a, cx + r * a, cy + r / a]
            if outline:
                draw.ellipse(box, outline=fill, width=width)
            else:
                draw.ellipse(box, fill=fill)
            continue
        if style == 1:  # прямоугольник с поворотом
            w, h = r * np.exp(rng.uniform(-0.7, 0.7, 2))
            a = rng.uniform(0, np.pi) if rng.random() < 0.5 else 0.0
            corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            pts = [tuple(p) for p in corners @ rot.T + [cx, cy]]
        else:
            pts = _polygon(rng, cx, cy, r)
        if outline:
            draw.polygon(pts, outline=fill, width=width)
        else:
            draw.polygon(pts, fill=fill)
    return np.asarray(img.resize((n, n), Image.Resampling.BOX) if s > 1 else img, np.uint8)


def _lines(rng: np.random.Generator, n: int) -> np.ndarray:
    s = 4
    m = n * s
    img = Image.new("L", (m, m), int(rng.integers(0, 60)))
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(1, 10))):
        fill = int(rng.integers(120, 256))
        width = max(1, round(s * _log_uniform(rng, 1, 5)))  # 1–5 пикселей после уменьшения
        style = rng.choice(5, p=[0.25, 0.3, 0.2, 0.1, 0.15])
        if style == 0:  # ломаная
            pts = rng.uniform(-0.05, 1.05, (int(rng.integers(2, 8)), 2)) * m
            draw.line([tuple(p) for p in pts], fill=fill, width=width, joint="curve")
        elif style == 1:  # кубическая кривая Безье — похоже на штрих от руки
            p = rng.uniform(-0.1, 1.1, (4, 2)) * m
            t = np.linspace(0, 1, 64)[:, None]
            pts = (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1]
            pts = pts + 3 * (1 - t) * t**2 * p[2] + t**3 * p[3]
            draw.line([tuple(q) for q in pts], fill=fill, width=width, joint="curve")
        elif style == 2:  # окружность, эллипс или дуга
            cx, cy = rng.uniform(0.1, 0.9, 2) * m
            r, a = m * rng.uniform(0.05, 0.45), np.exp(rng.uniform(-0.5, 0.5))
            box = [cx - r * a, cy - r / a, cx + r * a, cy + r / a]
            if rng.random() < 0.5:
                draw.ellipse(box, outline=fill, width=width)
            else:
                start = rng.uniform(0, 360)
                draw.arc(box, start, start + rng.uniform(60, 300), fill=fill, width=width)
        elif style == 3:  # спираль
            t = np.linspace(0, 2 * np.pi * rng.uniform(1.5, 6), 300)
            r = m * rng.uniform(0.15, 0.5) * t / t[-1]
            c = rng.uniform(0.3, 0.7, 2) * m
            pts = np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1)
            draw.line([tuple(q) for q in pts], fill=fill, width=width, joint="curve")
        else:  # штриховка параллельными линиями
            a, period = rng.uniform(0, np.pi), s * rng.uniform(4, 24)
            d, normal = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            for k in np.arange(-m, m, period) + rng.uniform(0, period):
                p = m / 2 + normal * k
                draw.line([tuple(p - d * m), tuple(p + d * m)], fill=fill, width=width)
    return np.asarray(img.resize((n, n), Image.Resampling.BOX), np.uint8)


# ---------- Точки, силуэты, коды, текстуры ----------


def _dots(rng: np.random.Generator, n: int) -> np.ndarray:
    """Массивы пятен — классические цели ДОЭ: регулярные, случайные и «звёздное поле»."""
    style = rng.integers(3)
    if style == 0:
        k = int(rng.integers(2, 17))
        period = rng.uniform(0.3, 0.95) * n / max(k - 1, 1)
        i, j = np.meshgrid(np.arange(k) - (k - 1) / 2, np.arange(k) - (k - 1) / 2)
        pts = np.stack([i.ravel(), j.ravel()], 1) * period
        a = rng.uniform(0, np.pi / 2) if rng.random() < 0.5 else 0.0
        pts = pts @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
        jitter = rng.normal(0, rng.uniform(0, 0.15) * period, pts.shape)
        pts += n / 2 + rng.normal(0, 0.05 * n, 2) + jitter
        amp = np.ones(len(pts)) if rng.random() < 0.5 else rng.uniform(0.3, 1, len(pts))
    elif style == 1:
        count = int(_log_uniform(rng, 1, 400))
        pts = rng.uniform(0, n, (count, 2))
        amp = rng.uniform(0.3, 1, count)
    else:
        count = int(rng.integers(20, 400))  # «звёздное поле»: яркости в пределах 1:10
        pts = rng.uniform(0, n, (count, 2))
        amp = np.exp(rng.uniform(np.log(0.1), 0, count))
    field = np.zeros((n, n))
    inside = (pts >= 0).all(1) & (pts <= n - 1).all(1)
    pts, amp = pts[inside], amp[inside]
    if len(pts):
        base = np.floor(pts).astype(int)
        frac = pts - base
        x1, y1 = np.minimum(base[:, 0] + 1, n - 1), np.minimum(base[:, 1] + 1, n - 1)
        for xi, yi, wgt in (
            (base[:, 0], base[:, 1], (1 - frac[:, 0]) * (1 - frac[:, 1])),
            (x1, base[:, 1], frac[:, 0] * (1 - frac[:, 1])),
            (base[:, 0], y1, (1 - frac[:, 0]) * frac[:, 1]),
            (x1, y1, frac[:, 0] * frac[:, 1]),
        ):
            np.add.at(field, (yi, xi), amp * wgt)
    field = ndimage.gaussian_filter(field, _log_uniform(rng, 0.5, 3.5))
    bg = rng.uniform(0, 0.15)
    return _to_u8(bg + (1 - bg) * _stretch(field))


def _blobs(rng: np.random.Generator, n: int) -> np.ndarray:
    """Силуэты: порог гладкого шума — органические пятна."""
    v = _power_noise(rng, n, rng.uniform(2.0, 3.5))
    mask = v > np.quantile(v, rng.uniform(0.45, 0.9))
    if rng.random() < 0.2:  # только контур
        mask &= ~ndimage.binary_erosion(mask, iterations=int(rng.integers(1, 4)))
    fill = mask.astype(np.float64)
    if rng.random() < 0.3:  # фактура внутри силуэта
        fill *= 0.4 + 0.6 * _power_noise(rng, n, rng.uniform(1.0, 2.5))
    if rng.random() < 0.5:
        fill = ndimage.gaussian_filter(fill, rng.uniform(0.5, 1.5))
    bg, fg = rng.uniform(0, 0.25), rng.uniform(0.6, 1)
    return _to_u8(bg + (fg - bg) * fill)


FINDER = np.ones((7, 7), bool)
FINDER[1:6, 1:6] = False
FINDER[2:5, 2:5] = True


def _code(rng: np.random.Generator, n: int) -> np.ndarray:
    """Двоичные коды: QR-подобные блоки (страницы данных), штрихкоды, шахматка и решётки."""
    style = rng.choice(3, p=[0.6, 0.2, 0.2])
    if style == 0:
        b = int(rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 16]))
        cells = -(-n // b)
        bits = rng.random((cells, cells)) < rng.uniform(0.3, 0.7)
        if cells >= 21 and rng.random() < 0.5:  # поисковые квадраты QR в трёх углах
            for r0, c0 in ((0, 0), (0, cells - 7), (cells - 7, 0)):
                bits[r0 : r0 + 7, c0 : c0 + 7] = FINDER
        v = np.kron(bits, np.ones((b, b)))[:n, :n]
        if rng.random() < 0.5:  # код занимает часть поля — вокруг «тихая зона»
            size = int(n * rng.uniform(0.45, 0.9))
            y0, x0 = rng.integers(0, n - size + 1, 2)
            frame = np.zeros((n, n))
            frame[y0 : y0 + size, x0 : x0 + size] = v[:size, :size]
            v = frame
    elif style == 1:
        widths = rng.integers(1, 5, 2 * n) * int(rng.integers(1, 3))
        bars = np.repeat(np.arange(len(widths)) % 2, widths)[:n].astype(np.float64)
        v = np.tile(bars, (n, 1))
        if rng.random() < 0.6:  # штрихкод — полоса, а не всё поле
            y0, y1 = sorted(rng.integers(0, n, 2))
            v[: min(y0, n - 8)] = 0
            v[max(y1, y0 + 8) :] = 0
    else:
        s = 4
        yy, xx = (np.mgrid[0 : n * s, 0 : n * s] + 0.5) / s
        a, period = rng.uniform(0, np.pi), rng.uniform(4, 40)
        u = (xx * np.cos(a) + yy * np.sin(a)) / period
        w = (-xx * np.sin(a) + yy * np.cos(a)) / period
        v = (np.floor(u) + np.floor(w)) % 2 if rng.random() < 0.6 else np.floor(u) % 2
        v = v.reshape(n, s, n, s).mean((1, 3))
    lo, hi = rng.uniform(0, 0.3), rng.uniform(0.7, 1)
    return _to_u8(lo + (hi - lo) * v)


def _texture(rng: np.random.Generator, n: int) -> np.ndarray:
    """Текстуры: полосовой и ориентированный шум, ячейки Вороного, сумма синусоид."""
    style = rng.integers(4)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    if style == 2:
        seeds = rng.uniform(0, n, (int(rng.integers(6, 80)), 2))
        d2 = (yy.ravel()[:, None] - seeds[:, 0]) ** 2 + (xx.ravel()[:, None] - seeds[:, 1]) ** 2
        variant = rng.integers(3)
        if variant == 0:  # расстояние до центра ячейки
            v = np.sqrt(d2.min(1))
        elif variant == 1:  # границы ячеек
            two = np.sqrt(np.partition(d2, 1, axis=1)[:, :2])
            v = -np.minimum(two[:, 1] - two[:, 0], rng.uniform(2, 8))
        else:  # мозаика: своя яркость у каждой ячейки
            v = rng.random(len(seeds))[d2.argmin(1)]
        v = _stretch(v.reshape(n, n))
    elif style == 3:
        v = np.zeros((n, n))
        for _ in range(int(rng.integers(2, 5))):
            f, a = _log_uniform(rng, 0.02, 0.25), rng.uniform(0, np.pi)
            v += np.cos(2 * np.pi * f * (xx * np.cos(a) + yy * np.sin(a)) + rng.uniform(0, 7))
        v = _stretch(v)
    else:
        f = np.fft.fftfreq(n)
        fy, fx = f[:, None], f[None, :]
        if style == 0:  # полосовой шум: детали одного масштаба
            f0 = _log_uniform(rng, 0.03, 0.3)
            width = f0 * rng.uniform(0.1, 0.5)
            spectrum = np.exp(-0.5 * ((np.hypot(fx, fy) - f0) / width) ** 2)
        else:  # ориентированный шум: волокна, дерево, ткань
            a = rng.uniform(0, np.pi)
            u, w = fx * np.cos(a) + fy * np.sin(a), -fx * np.sin(a) + fy * np.cos(a)
            su = _log_uniform(rng, 0.005, 0.04)
            sw = su * _log_uniform(rng, 3, 15)
            spectrum = np.exp(-0.5 * ((u / su) ** 2 + (w / sw) ** 2))
        spectrum[0, 0] = 0
        v = _stretch(np.fft.ifft2(spectrum * np.exp(2j * np.pi * rng.random((n, n)))).real)
    if rng.random() < 0.25:  # двоичная фактура
        v = (v > np.quantile(v, rng.uniform(0.3, 0.7))).astype(np.float64)
    return _to_u8(v)


def _gradient(rng: np.random.Generator, n: int) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n] / n - 0.5
    if rng.random() < 0.5:
        a = rng.uniform(0, 2 * np.pi)
        v = x * np.cos(a) + y * np.sin(a)
    else:
        cx, cy = rng.uniform(-0.4, 0.4, 2)
        v = -np.hypot(x - cx, y - cy)
    v = (v - v.min()) / (np.ptp(v) + 1e-12)
    if rng.random() < 0.5:  # периодическая модуляция: полосы и кольца
        v = 0.5 + 0.5 * np.cos(2 * np.pi * rng.uniform(1, 12) * v)
    return np.rint(v * 255).astype(np.uint8)


def _noise(rng: np.random.Generator, n: int) -> np.ndarray:
    # alpha 0,6–2: от «зернистого» до гладкого, у фото alpha ≈ 1–1,5. Раньше было 1,5–3,5:
    # почти вся дисперсия в деталях крупнее 16 пикселей, глаже любых фото validation.
    # Совсем гладкие цели остаются в силуэтах и градиентах.
    return _to_u8(_power_noise(rng, n, rng.uniform(0.6, 2.0)))


MAKERS = {
    "text": _text, "shapes": _shapes, "lines": _lines, "dots": _dots, "blobs": _blobs,
    "code": _code, "texture": _texture, "gradient": _gradient, "noise": _noise,
}  # fmt: skip


class PhotoFolder:
    """Случайные квадратные вырезки из фотографий каталога → серый n×n."""

    def __init__(self, root: str | Path):
        self.files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in PHOTO_SUFFIXES)
        if not self.files:
            raise FileNotFoundError(f"в {root} нет изображений")

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        with Image.open(self.files[rng.integers(len(self.files))]) as img:
            img = img.convert("L")
            w, h = img.size
            # Размер вырезки лог-равномерный: деталь и общий план одинаково часты
            s = min(round(_log_uniform(rng, min(w, h, n), min(w, h))), w, h)
            x0, y0 = int(rng.integers(0, w - s + 1)), int(rng.integers(0, h - s + 1))
            img = img.resize((n, n), Image.Resampling.LANCZOS, box=(x0, y0, x0 + s, y0 + s))
            return np.asarray(img, np.uint8)


def _augment(rng: np.random.Generator, img: np.ndarray, kind: str) -> np.ndarray:
    v = img.astype(np.float64) / 255
    if kind not in NO_FLIP:
        if rng.random() < 0.5:
            v = v[:, ::-1]
        v = np.rot90(v, int(rng.integers(4)))
    if rng.random() < 0.3:  # инверсия: светлый фон тоже встречается
        v = 1 - v
    if rng.random() < 0.3:
        v = v ** _log_uniform(rng, 0.5, 2)
    if rng.random() < 0.15:
        v = ndimage.gaussian_filter(v, rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:  # яркость и контраст: T → b + c·T, контраст от 0,15 до 1
        c = _log_uniform(rng, 0.15, 1)
        v = rng.uniform(0, 1 - c) + c * _stretch(v)
    return _to_u8(v)


def _transform(img: np.ndarray, angle: float, scale: float, shift: np.ndarray) -> np.ndarray:
    """Поворот на angle (градусы), масштаб и сдвиг вокруг центра; фон — медиана края."""
    n = img.shape[-1]
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    a = np.deg2rad(angle)
    inv = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) / scale  # выход → вход
    c = np.array([n / 2, n / 2])
    offset = c - inv @ (c + shift)
    data = (inv[0, 0], inv[0, 1], offset[0], inv[1, 0], inv[1, 1], offset[1])
    out = Image.fromarray(img).transform(
        (n, n), Image.Transform.AFFINE, data, Image.Resampling.BILINEAR,
        fillcolor=int(np.median(edge)),
    )  # fmt: skip
    return np.asarray(out, np.uint8)


class SampleGenerator:
    """Один пример — uint8 [4, n, n]: четвёрка в одном из режимов MODES."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        photos_dir=None,
        n: int = N,
        modes: dict[str, float] | None = None,
        volume: dict | None = None,
    ):
        if weights is None:
            weights = dict(DEFAULT_WEIGHTS)
            if photos_dir is not None:
                weights["photo"] = PHOTO_SHARE / (1 - PHOTO_SHARE) * sum(DEFAULT_WEIGHTS.values())
        weights = {k: float(v) for k, v in weights.items() if v > 0}
        if "photo" in weights and photos_dir is None:
            raise ValueError("в weights есть photo, а photos_dir не задан: укажите каталог фото")
        unknown = set(weights) - set(KINDS)
        if unknown or not weights:
            raise ValueError(f"неизвестные виды изображений: {unknown}; есть: {KINDS}")
        modes = {k: float(v) for k, v in (modes or DEFAULT_MODES).items() if v > 0}
        if set(modes) - set(MODES) or not modes:
            raise ValueError(f"режимы четвёрки — из {MODES}, получено {sorted(modes)}")
        self.kinds = list(weights)
        self.p = np.array([weights[k] for k in self.kinds]) / sum(weights.values())
        self.modes = list(modes)
        self.mode_p = np.array([modes[k] for k in self.modes]) / sum(modes.values())
        self.photos = PhotoFolder(photos_dir) if photos_dir is not None else None
        self.n = n
        # volume — аргументы VolumeGenerator (families, supersample); крайние случаи (edge)
        # дают однотонные цели, на которых Пирсон не определён, — в обучение они не идут
        volume = dict(volume or {})
        if volume.get("edge"):
            raise ValueError("volume.edge — крайние случаи только для оценки, не для обучения")
        self.volume = None
        if "volume" in self.modes:
            setup = TASK if n == TASK.n else replace(TASK, n=n)
            self.volume = VolumeGenerator(**volume, setup=setup)

    def _kind(self, rng: np.random.Generator) -> str:
        return self.kinds[rng.choice(len(self.kinds), p=self.p)]

    def image(self, rng: np.random.Generator, kind: str | None = None) -> np.ndarray:
        """Одна картинка вида kind (по умолчанию — случайного) с аугментациями."""
        kind = kind or self._kind(rng)
        for _ in range(20):
            raw = self.photos.sample(rng, self.n) if kind == "photo" else MAKERS[kind](rng, self.n)
            if int(raw.max()) - int(raw.min()) < MIN_RANGE:  # до аугментаций: контраст
                continue  # растянул бы 2–3 уровня шума до полного размаха
            img = _augment(rng, raw, kind)
            if int(img.max()) - int(img.min()) >= MIN_RANGE:
                return img
        raise RuntimeError(f"вид {kind}: 20 почти однотонных картинок подряд")

    def _scene(self, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
        """Одна картинка по всем плоскостям: копии или сдвиг, масштаб и поворот —
        плавно по глубине (объект «движется») или случайно."""
        base = self.image(rng, kind)
        copies = rng.uniform(0, 0.5)
        progressive = rng.random() < 0.5
        lim = np.array([rng.uniform(0, 30), rng.uniform(0, 0.35), rng.uniform(0, 0.2 * self.n)])
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction) + 1e-12
        planes = []
        for t in np.linspace(-1, 1, 4):
            if rng.random() < copies:
                planes.append(base)
                continue
            if progressive:
                angle, log_scale, shift = t * lim[0], t * lim[1], t * lim[2] * direction
            else:
                angle, log_scale = rng.uniform(-1, 1, 2) * lim[:2]
                shift = rng.normal(0, lim[2] / 2, 2)
            img = _transform(base, angle, np.exp(log_scale), shift)
            planes.append(img if int(img.max()) - int(img.min()) >= MIN_RANGE else base)
        return planes

    def sample_with_meta(self, rng: np.random.Generator) -> tuple[np.ndarray, list[str], str]:
        """(uint8 [4, n, n], вид каждой плоскости, режим четвёрки)."""
        mode = self.modes[rng.choice(len(self.modes), p=self.mode_p)]
        if mode == "independent":
            kinds = [self._kind(rng) for _ in range(4)]
            planes = [self.image(rng, k) for k in kinds]
        elif mode == "themed":
            kinds = [self._kind(rng)] * 4
            planes = [self.image(rng, k) for k in kinds]
        elif mode == "volume":  # вид плоскостей — семейство объекта
            volume, desc = self.volume.sample(rng)
            return volume, [desc["family"]] * 4, mode
        else:
            kinds = [self._kind(rng)] * 4
            planes = self._scene(rng, kinds[0])
        return np.stack(planes), kinds, mode

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.sample_with_meta(rng)[0]


class TrainStream(torch.utils.data.IterableDataset):
    """Бесконечный поток T ∈ [0, 1] float32 [4, n, n]; у каждого воркера свой seed."""

    def __init__(self, generator: SampleGenerator, seed: int = 0):
        self.generator = generator
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            yield torch.from_numpy(self.generator.sample(rng).astype(np.float32) / 255)
