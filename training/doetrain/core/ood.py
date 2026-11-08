"""Категории картинок, которых нет в генераторе обучения, — проверка обобщения.

Генератор обучения (doetrain.train.data) покрывает фото, текст латиницей и кириллицей,
фигуры, линии, точки, силуэты, двоичные коды, текстуры, градиенты и шум, поэтому отложенный
test (core.data.HELD_OUT_TEST) проверяет в основном знакомые категории. Здесь — категории,
которых генератор не рисует: другие алфавиты (греческий, армянский, грузинский, иврит),
пиктограммы, научные графики, формулы, фракталы и оптические моды (Эрмит–Гаусс,
Лагерр–Гаусс, Эйри). Это новое содержание из знакомых примитивов (штрихи, пятна, заливки),
а не картинки иной природы: по попиксельному сходству такие категории от генератора не
отличить (experiments/R02-data/ood_novelty.py), новизну показываем по качеству сети
относительно Adam.

Набор детерминирован (сид — категория и номер четвёрки) и задан кодом. Варианты внутри
категории (алфавит, вид графика, фрактала, моды, формула) идут латинским квадратом: каждый
вариант проходит все четыре плоскости, иначе вклад варианта не отделить от вклада плоскости.
В обучение и выбор гиперпараметров набор не входит: на нём считаем только итоговые числа.
"""

import importlib.util
from functools import cache
from pathlib import Path

import numpy as np

CATEGORIES = ("alphabets", "pictograms", "plots", "formulas", "fractals", "modes")
QUADS = 4  # четвёрок на категорию
SEED = 2026

# Буквы, не похожие на латиницу и кириллицу (без Α, Β, Ο, Ρ, Γ, Π, Φ и подобных)
SCRIPTS = {
    "greek": "αβγδζηθλμξπσφψωΔΘΛΞΣΨΩ",
    "armenian": "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ",
    "georgian": "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ",
    "hebrew": "אבגדהוזחטיכלמנסעפצקרשת",
}
PICTOGRAMS = "♔♕♖♗♘♙♚♛♜♝♞♟☀☁☂☃☎☕☘☠☢☣☮☯☺☻♠♣♥♦♨♪♫♬⚓⚔⚕⚖⚙⚛⚠⚡✂✈✉✌✍✎❄❤☚☛☜☝☞☟♀♂⚧⛄⛅⛪⛵⛺"
FONT_FILES = (
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSerif.ttf",
    "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf", "DejaVuSansMono.ttf",
    "DejaVuSansMono-Bold.ttf",
)  # fmt: skip
FORMULAS = (
    r"$E = mc^2$",
    r"$e^{i\pi} + 1 = 0$",
    r"$\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}$",
    r"$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$",
    r"$\nabla \cdot \vec{E} = \frac{\rho}{\varepsilon_0}$",
    r"$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$",
    r"$K = \frac{e^{jkz}}{j\lambda z}\,\mathcal{F}[H]$",
    r"$\Delta x\, \Delta p \geq \frac{\hbar}{2}$",
    r"$\sqrt{a^2 + b^2}$",
    r"$\lim_{x \to 0} \frac{\sin x}{x} = 1$",
    r"$\oint_C \vec{B} \cdot d\vec{l} = \mu_0 I$",
    r"$f(x) = \sum_k c_k\, e^{2\pi i k x}$",
    r"$\frac{\lambda z}{N \Delta x^2}$",
    r"$i\hbar \frac{\partial \psi}{\partial t} = \hat{H} \psi$",
    r"$a^2 + b^2 = c^2$",
    r"$\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}$",
)  # 16 формул на 4 × QUADS = 16 картинок — без повторов


# Формулы без повторов в пределах категории: фиксированная перестановка, по номеру картинки
FORMULA_ORDER = np.random.default_rng(SEED).permutation(len(FORMULAS))


def _to_u8(v: np.ndarray) -> np.ndarray:
    v = v - v.min()
    return np.clip(np.rint(v / (v.max() + 1e-12) * 255), 0, 255).astype(np.uint8)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


@cache
def _cmaps() -> dict[str, frozenset[int]]:
    """{путь к шрифту DejaVu из matplotlib: коды символов в нём}."""
    from fontTools.ttLib import TTFont

    spec = importlib.util.find_spec("matplotlib")
    if spec is None or not spec.submodule_search_locations:
        return {}
    root = Path(spec.submodule_search_locations[0]) / "mpl-data" / "fonts" / "ttf"
    return {
        str(root / name): frozenset(TTFont(root / name, lazy=True).getBestCmap() or {})
        for name in FONT_FILES
        if (root / name).is_file()
    }


def _fonts_with(chars: str) -> tuple[str, ...]:
    """Шрифты, в которых есть все символы chars."""
    return tuple(path for path, cmap in _cmaps().items() if all(ord(c) in cmap for c in chars))


@cache
def _pictograms() -> str:
    """Пиктограммы, которые есть в DejaVu Sans."""
    sans = [cmap for path, cmap in _cmaps().items() if path.endswith("DejaVuSans.ttf")]
    return "".join(c for c in PICTOGRAMS if sans and ord(c) in sans[0])


def _glyphs(rng: np.random.Generator, n: int, chars: str, fonts: tuple[str, ...]) -> np.ndarray:
    """Один крупный знак по центру или слово из 2–4 знаков; светлое на тёмном (30 % — наоборот)."""
    from PIL import Image, ImageDraw, ImageFont

    if not fonts:
        raise RuntimeError("нет шрифтов DejaVu из matplotlib — набор OOD не построить")
    path = fonts[int(rng.integers(len(fonts)))]
    img = Image.new("L", (n, n), int(rng.integers(0, 50)))
    draw = ImageDraw.Draw(img)
    if rng.random() < 0.6:
        text, size = str(rng.choice(list(chars))), int(n * rng.uniform(0.5, 0.9))
    else:
        text = "".join(rng.choice(list(chars), int(rng.integers(2, 5))))
        size = int(n * rng.uniform(0.2, 0.32))
    # Простая раскладка: с raqm (есть в Pillow на Linux при libfribidi) иврит шёл бы справа
    # налево и набор на разных машинах различался бы
    basic = ImageFont.Layout.BASIC
    font = ImageFont.truetype(path, size, layout_engine=basic)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    if x1 - x0 > 0.95 * n or y1 - y0 > 0.95 * n:  # не влезло — уменьшаем
        size = int(size * 0.9 * n / max(x1 - x0, y1 - y0))
        font = ImageFont.truetype(path, size, layout_engine=basic)
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    xy = ((n - (x1 - x0)) / 2 - x0, (n - (y1 - y0)) / 2 - y0)
    draw.text(xy, text, fill=int(rng.integers(170, 256)), font=font)
    v = np.asarray(img, np.uint8)
    return 255 - v if rng.random() < 0.3 else v


def _alphabet(rng: np.random.Generator, n: int, slot: int) -> np.ndarray:
    chars = SCRIPTS[sorted(SCRIPTS)[slot % len(SCRIPTS)]]  # алфавиты поровну
    return _glyphs(rng, n, chars, _fonts_with(chars))


def _pictogram(rng: np.random.Generator, n: int, slot: int) -> np.ndarray:
    chars = _pictograms()
    return _glyphs(rng, n, chars, _fonts_with(chars))


def _figure(n: int):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(n / 100, n / 100), dpi=100)
    return fig, FigureCanvasAgg(fig)


def _rendered(canvas) -> np.ndarray:
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].mean(-1)


def _plot(rng: np.random.Generator, n: int, slot: int) -> np.ndarray:
    """Научный график с осями: кривые, столбцы, точки или гистограмма."""
    fig, canvas = _figure(n)
    ax = fig.add_axes((0.2, 0.15, 0.75, 0.78))
    style, lw = slot % 4, rng.uniform(0.8, 2.0)  # виды графиков поровну
    if style == 0:
        x = np.linspace(0, rng.uniform(2, 10), 200)
        for _ in range(int(rng.integers(1, 4))):
            a, f, d = rng.uniform(0.5, 2), rng.uniform(0.5, 3), rng.uniform(0, 0.5)
            curve = int(rng.integers(3))
            if curve == 0:
                y = a * np.exp(-d * x) * np.sin(f * x + rng.uniform(0, 2 * np.pi))
            elif curve == 1:
                y = a * np.exp(-2 * d * x)
            else:
                y = np.polyval(rng.normal(0, 1, 3), x / x[-1])
            ax.plot(x, y, color="k", lw=lw)
    elif style == 1:
        k = int(rng.integers(4, 13))
        ax.bar(np.arange(k), rng.uniform(0.2, 1, k), color=str(round(rng.uniform(0, 0.6), 2)))
    elif style == 2:
        m = int(rng.integers(20, 200))
        centers = rng.normal(0, 1, (int(rng.integers(1, 4)), 2))
        pts = centers[rng.integers(len(centers), size=m)] + rng.normal(0, 0.3, (m, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=rng.uniform(2, 12), c="k")
    else:
        ax.hist(rng.normal(0, 1, 500), bins=int(rng.integers(8, 30)), color="0.3")
    ax.tick_params(labelsize=rng.uniform(4, 6), length=2, width=0.5, pad=1)
    if rng.random() < 0.4:
        ax.grid(True, lw=0.4)
    v = _to_u8(_rendered(canvas))
    return 255 - v if rng.random() < 0.3 else v


def _formula(rng: np.random.Generator, n: int, slot: int) -> np.ndarray:
    """Формула (mathtext matplotlib) по центру, размер — по ширине поля."""
    fig, canvas = _figure(n)
    fig.patch.set_facecolor("black")
    family = str(rng.choice(["dejavusans", "dejavuserif", "cm", "stix"]))
    text = fig.text(
        0.5, 0.5, FORMULAS[FORMULA_ORDER[slot % len(FORMULAS)]], ha="center", va="center", fontsize=20,
        color="white", math_fontfamily=family,
    )  # fmt: skip
    canvas.draw()
    box = text.get_window_extent(renderer=canvas.get_renderer())
    text.set_fontsize(20 * min(0.92 * n / box.width, 0.8 * n / box.height) * rng.uniform(0.75, 1))
    v = _to_u8(_rendered(canvas))
    return 255 - v if rng.random() < 0.3 else v


def _escape_time(z: np.ndarray, c: np.ndarray, iters: int = 80) -> np.ndarray:
    count = np.zeros(z.shape)
    alive = np.ones(z.shape, bool)
    for _ in range(iters):
        z[alive] = z[alive] ** 2 + c[alive]
        alive &= np.abs(z) <= 2
        count += alive
    return count


def _points_image(pts: np.ndarray, n: int, margin: float = 0.06) -> np.ndarray:
    """Облако точек → картинка n×n (вписываем с полями, сохраняя пропорции)."""
    lo, hi = pts.min(0), pts.max(0)
    scale = (1 - 2 * margin) * n / max(hi - lo)
    ij = (pts - (lo + hi) / 2) * scale + n / 2
    img = np.zeros((n, n))
    ij = np.clip(ij.astype(int), 0, n - 1)
    np.add.at(img, (n - 1 - ij[:, 1], ij[:, 0]), 1)
    return np.minimum(img, np.quantile(img[img > 0], 0.9))


def _fractal(rng: np.random.Generator, n: int, slot: int) -> np.ndarray:
    """Мандельброт, Жюлиа, треугольник Серпинского, снежинка Коха, дерево, папоротник."""
    from PIL import Image, ImageDraw

    style = slot % 6  # виды фракталов поровну
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 - 1
    if style == 0:
        if rng.random() < 0.4:
            cx, cy, half = -0.6, 0.0, 1.3
        else:
            spots = ((-0.745, 0.113), (-0.1615, 1.0405), (-1.2507, 0.0201), (0.2549, 0.0005))
            cx, cy = spots[int(rng.integers(len(spots)))]
            half = _log_uniform(rng, 0.01, 0.3)
        c = cx + half * xx + 1j * (cy + half * yy)
        v = _escape_time(np.zeros_like(c), c)
        v = np.log1p(v) if rng.random() < 0.7 else (v == v.max()).astype(float)
    elif style == 1:
        seeds = (-0.8 + 0.156j, 0.285 + 0.01j, -0.4 + 0.6j, -0.70176 - 0.3842j, 0.355 + 0.355j)
        c0 = seeds[int(rng.integers(len(seeds)))] + complex(*rng.normal(0, 0.01, 2))
        half, shift = rng.uniform(0.9, 1.6), rng.normal(0, 0.15, 2)
        z = half * xx + shift[0] + 1j * (half * yy + shift[1])
        v = np.log1p(_escape_time(z, np.full(xx.shape, c0)))
    elif style == 2:  # треугольник Серпинского: «игра в хаос»
        angles = rng.uniform(0, 2 * np.pi) + np.arange(3) * 2 * np.pi / 3 + rng.normal(0, 0.15, 3)
        vertices = np.stack([np.cos(angles), np.sin(angles)], 1)  # почти правильный треугольник
        pts = rng.uniform(-1, 1, (3000, 2))
        cloud = []
        for k in range(25):
            pts = (pts + vertices[rng.integers(3, size=len(pts))]) / 2
            if k > 5:
                cloud.append(pts)
        v = _points_image(np.concatenate(cloud), n)
    elif style == 5:  # папоротник Барнсли
        maps = np.array([
            [0, 0, 0, 0.16, 0, 0], [0.85, 0.04, -0.04, 0.85, 0, 1.6],
            [0.2, -0.26, 0.23, 0.22, 0, 1.6], [-0.15, 0.28, 0.26, 0.24, 0, 0.44],
        ])  # fmt: skip
        pts, cloud = np.zeros((3000, 2)), []
        for k in range(40):
            m = maps[rng.choice(4, size=len(pts), p=[0.01, 0.85, 0.07, 0.07])]
            x, y = pts[:, 0], pts[:, 1]
            pts = np.stack(
                [m[:, 0] * x + m[:, 1] * y + m[:, 4], m[:, 2] * x + m[:, 3] * y + m[:, 5]], 1
            )
            if k > 10:
                cloud.append(pts)
        cloud = np.concatenate(cloud) - [0.25, 5]  # центр папоротника
        a = rng.uniform(-0.5, 0.5)  # наклон, отражение и вытянутость — у каждого свои
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        flip = [rng.choice([-1, 1]) * np.exp(rng.uniform(-0.3, 0.3)), 1]
        v = _points_image(cloud * flip @ rot.T, n)
    else:  # снежинка Коха или дерево: рисуем в 4× и уменьшаем
        s = 4
        img = Image.new("L", (n * s, n * s), 0)
        draw = ImageDraw.Draw(img)
        if style == 3:
            a0 = rng.uniform(0, 2 * np.pi)
            pts = n * s / 2 * (1 + 1j) + 0.38 * n * s * np.exp(
                1j * (a0 + np.arange(3) * 2 * np.pi / 3)
            )
            for _ in range(int(rng.integers(2, 5))):
                p, q = pts, np.roll(pts, -1)
                d = (q - p) / 3
                pts = np.stack(
                    [p, p + d, p + d + d * np.exp(-1j * np.pi / 3), p + 2 * d], 1
                ).ravel()
            poly = [(z.real, z.imag) for z in pts]
            if rng.random() < 0.5:
                draw.polygon(poly, fill=255)
            else:
                draw.line([*poly, poly[0]], fill=255, width=int(s * rng.uniform(1, 2.5)))
        else:
            spread, ratio = rng.uniform(0.3, 0.7), rng.uniform(0.62, 0.78)
            depth = int(rng.integers(6, 10))

            def branch(x, y, angle, length, level):
                x2, y2 = x + length * np.cos(angle), y - length * np.sin(angle)
                draw.line([(x, y), (x2, y2)], fill=255, width=max(1, int(s * 0.4 * level)))
                if level > 1:
                    for sign in (-1, 1):
                        branch(x2, y2, angle + sign * spread, length * ratio, level - 1)

            branch(n * s / 2, n * s * 0.95, np.pi / 2, n * s * 0.28, depth)
        v = np.asarray(img.resize((n, n), Image.Resampling.BOX), np.float64)
    v = _to_u8(v)
    return 255 - v if rng.random() < 0.3 else v


def _mode(rng: np.random.Generator, n: int, slot: int) -> np.ndarray:
    """Интенсивность оптических мод: Эрмит–Гаусс, Лагерр–Гаусс, «лепестки» и дифракция
    (Эйри — круглая апертура, sinc² — квадратная, J0² — бесселев пучок)."""
    from scipy.special import eval_genlaguerre, eval_hermite, j0, j1

    w = _log_uniform(rng, 8, 22)  # перетяжка, пиксели
    cy, cx = n / 2 + rng.normal(0, 3, 2)
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    a = rng.uniform(0, np.pi)
    x = ((xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)) / w
    y = (-(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)) / w
    r2 = x**2 + y**2
    style = slot % 4  # виды мод поровну
    if style == 0:
        m, k = rng.integers(0, 5, 2)
        v = (eval_hermite(m, np.sqrt(2) * x) * eval_hermite(k, np.sqrt(2) * y)) ** 2 * np.exp(
            -2 * r2
        )
    elif style in (1, 2):
        p, ell = int(rng.integers(0, 3)), int(rng.integers(1, 7))
        v = (2 * r2) ** ell * eval_genlaguerre(p, ell, 2 * r2) ** 2 * np.exp(-2 * r2)
        if style == 2:  # LG(p, l) + LG(p, −l): 2l лепестков
            v = v * np.cos(ell * np.arctan2(y, x)) ** 2
    else:  # дифракция: четыре одинаковых кольца Эйри разного масштаба почти не различались
        k = rng.uniform(2, 5)
        aperture = int(rng.integers(3))
        if aperture == 0:
            rr = np.sqrt(r2) * k
            v = np.where(rr > 1e-9, 2 * j1(rr) / np.maximum(rr, 1e-9), 1.0) ** 2
        elif aperture == 1:
            v = (
                np.sinc(k * x / np.pi) * np.sinc(k * y / np.pi * np.exp(rng.uniform(-0.4, 0.4)))
            ) ** 2
        else:
            v = j0(np.sqrt(r2) * k) ** 2
    v = v / v.max()
    return _to_u8(v ** (0.5 if rng.random() < 0.5 else 1.0))


MAKERS = {
    "alphabets": _alphabet, "pictograms": _pictogram, "plots": _plot,
    "formulas": _formula, "fractals": _fractal, "modes": _mode,
}  # fmt: skip


def ood_set(category: str, index: int, n: int = 128) -> np.ndarray:
    """Четвёрка uint8 [4, n, n] категории category; index ∈ [0, QUADS)."""
    if category not in MAKERS or not 0 <= index < QUADS:
        raise ValueError(f"категория — из {CATEGORIES}, номер — от 0 до {QUADS - 1}")
    rng = np.random.default_rng([SEED, CATEGORIES.index(category), index])
    # Латинский квадрат: в четвёрке index плоскость plane получает вариант (plane + index) % 4
    slots = [4 * index + (plane + index) % 4 for plane in range(4)]
    return np.stack([MAKERS[category](rng, n, slot) for slot in slots])


def ood_sets(n: int = 128) -> dict[str, np.ndarray]:
    """Все четвёрки: {"категория/номер": uint8 [4, n, n]}."""
    return {f"{c}/{i}": ood_set(c, i, n) for c in CATEGORIES for i in range(QUADS)}
