"""Трёхмерные объекты: четвёрка целей — сечения одного объекта V(x, y, z) плоскостями задачи.

Рабочая гипотеза, а не требование условия: прикладной смысл четырёх плоскостей — слои одной
3D-сцены (тестовые изображения организаторы не раскрывают). В R02 менялись категории картинок,
а связанные геометрией сечения не проверялись: режим «сцена» в train/data.py — преобразования
одной 2D-картинки. Здесь все четыре сечения следуют из одной геометрии.

Геометрия. Объект задан в физических координатах, мм; ось z — оптическая, z = 0 — ДОЭ.
Сечение k — значения V на сетке восстановления плоскости z_k: отсчёт (строка i, столбец j) —
точка x = (j − N/2)·p_k, y = (i − N/2)·p_k, где p_k = λz_k/(NΔx) = 10,0–11,5 мкм — шаг формулы
Френеля с одним БПФ (core/optics.py, сама модель не меняется). Поэтому один и тот же объект
в дальней плоскости в пикселях на 13 % меньше, чем в ближней. Соответствие пикселя и точки
закреплено тестом: ДОЭ, фокусирующий в точку (x, y, z_k), по optics.reconstruct даёт максимум
ровно в пикселе, который здесь отвечает (x, y). Пиксель усредняет V по своей площади
(SUPERSAMPLE² подотсчётов).

Это геометрические сечения объекта, а не распределения света, достижимые одним фазовым ДОЭ:
дефокус соседних слоёв не добавляется, и физическая реализуемость набора не гарантируется.

Объекты (FAMILIES): эллипсоиды и сферы, оболочки, наклонные цилиндры, конусы и трубки, торы,
плавно меняющиеся по глубине формы (скрутка, сужение, смена профиля, винтовые пряди), изоповерхности
гладкого 3D-шума, облака частиц и простые составные объекты (объединение, пучок трубок,
сквозное отверстие). Плоскости разнесены на 3,6 мм при поле зрения 1,3–1,5 мм, поэтому объект,
пересекающий все четыре плоскости, обычно вытянут по z (вытяжка — в params каждой части);
изотропны сферы и сферические оболочки.

Яркость. V ∈ [0, 1] задаёт объект: уровни частей, градиент, спад к краю, объёмная текстура, фон.
В 8 бит — round(255·V) сразу для всех плоскостей, без нормировки отдельных сечений: относительные
яркости плоскостей сохраняются (Пирсон в loss к ним не чувствителен, но цели честные).

Аугментации — только для объекта целиком, одни на все плоскости: тон (гамма, инверсия), фон
и размытие краёв в физических единицах; у близких вариантов (vary) ещё поворот, отражение и сдвиг
всего объекта (transform). Отдельные сечения не рисуются и не аугментируются.

Отбор. В основной набор идут объекты, у которых каждое сечение непустое и неоднотонное: размах
≥ MIN_RANGE и не меньше MIN_PIXELS пикселей отличаются от медианы сечения на MIN_RANGE.
Крайние случаи (пустые, сплошные и почти пустые сечения) — отдельное разбиение edge: объекты
там не обязаны пересекать все плоскости, часть растянута или сжата поперёк оси. Пирсон с однотонной
целью не определён, в обучение они не идут.

Описание объекта — словарь из чисел, строк и списков (JSON): render(desc) строит сечения из него,
VolumeGenerator.sample(rng) — из сида. Разбиения SPLITS: объект k разбиения s строится из сида
(seed, код s, k), его близкий вариант v — из (seed, код s, k, v), поэтому варианты всегда
в разбиении своего объекта, а разные разбиения не делят ни одного объекта. Семейство объекта k
задаёт последовательность золотого сечения: даже в малых validation и test доли семейств близки
к весам.
"""

import copy
import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from doetrain.core.optics import TASK, Setup

VERSION = 1
FAMILIES = ("ellipsoid", "shell", "cylinder", "torus", "morph", "blobs", "particles", "composite")
DEFAULT_FAMILIES = {
    "ellipsoid": 1.0, "shell": 1.0, "cylinder": 1.2, "torus": 1.0, "morph": 1.4,
    "blobs": 1.0, "particles": 0.6, "composite": 1.2,
}  # fmt: skip
SUPERSAMPLE = 4  # подотсчётов на пиксель по каждой оси
MIN_RANGE = 8  # как train.data.MIN_RANGE: минимальный размах сечения, из 255
MIN_PIXELS = 16  # столько пикселей должно отличаться от медианы сечения (≈ круг радиуса 2 px)
MAX_TRIES = 200
# Крайние случаи: целевой вид вырожденного сечения и сколько попыток его добиваться
EDGE_KINDS = {"empty": 0.4, "uniform": 0.3, "sparse": 0.3}
EDGE_TARGET_TRIES = 60
SEED = 3101
SPLITS = {"train": 1, "validation": 2, "test": 3, "edge": 4}  # код входит в сид объекта
GOLDEN = (math.sqrt(5) - 1) / 2

# Геометрия задачи в мм — по ней подобраны размеры объектов (рендер берёт любой Setup)
Z_PLANES = np.array(TASK.distances) * 1e3
Z_MID = float(Z_PLANES.mean())
Z_LO, Z_HI = float(Z_PLANES[0] - 0.6), float(Z_PLANES[-1] + 0.6)
IDENTITY = {"rotation": np.eye(3).tolist(), "shift": [0.0, 0.0, 0.0], "pivot": [0.0, 0.0, Z_MID]}


# ---------- Сетки плоскостей ----------


@dataclass(frozen=True)
class Plane:
    z: float  # мм
    pitch: float  # мм, λz/(NΔx)
    n: int

    @property
    def fov(self) -> float:
        return self.n * self.pitch


def planes(setup: Setup = TASK) -> list[Plane]:
    """Плоскости задачи: z и шаг сетки восстановления λz/(NΔx), мм."""
    return [
        Plane(z * 1e3, setup.wavelength * z / (setup.n * setup.pixel) * 1e3, setup.n)
        for z in setup.distances
    ]


def pixel_axis(n: int, pitch: float, supersample: int = 1, pad: int = 0) -> np.ndarray:
    """Координаты (мм) подотсчётов вдоль строки или столбца: центр пикселя j — (j − n/2)·pitch,
    как у центрированного БПФ в optics.py; pad пикселей с каждой стороны — запас для размытия."""
    idx = np.arange((n + 2 * pad) * supersample)
    return ((idx + 0.5) / supersample - 0.5 - pad - n / 2) * pitch


@dataclass(frozen=True)
class _Grid:
    """Сетка в плоскости, параллельной двум осям: точка (r, c) — rows[r] по оси row_axis,
    cols[c] по оси col_axis и fixed по третьей (оси: 0 — x, 1 — y, 2 — z)."""

    fixed_axis: int
    fixed: float
    row_axis: int
    rows: np.ndarray
    col_axis: int
    cols: np.ndarray


@dataclass(frozen=True)
class _Window:
    """Часть сетки, где может лежать примитив; локальные координаты q = q0 + qr·r + qc·c."""

    rows: slice
    cols: slice
    r: np.ndarray
    c: np.ndarray
    q0: np.ndarray
    qr: np.ndarray
    qc: np.ndarray

    def local(self) -> list[np.ndarray]:
        """q1, q2, q3 на окне, float32: точности хватает с запасом (≈ 1e-7 мм), а вдвое быстрее."""
        r, c = self.r.astype(np.float32)[:, None], self.c.astype(np.float32)[None, :]
        f32 = np.float32
        return [f32(self.q0[m]) + f32(self.qr[m]) * r + f32(self.qc[m]) * c for m in range(3)]


# ---------- Формы в локальных координатах ----------
# Примитив: точка p (мм) ↔ локальные q = M⁻¹(p − center); форма задана в q, M — матрица 3×3
# (поворот, размеры, вытяжка). Каждая форма даёт маску «внутри» и rho — нормированное
# расстояние от сердцевины (0 — центр, ось или осевая линия, 1 — граница) для спада яркости.


def _field(spec: dict, w: _Window) -> np.ndarray:
    """Σ a·cos(2π k·q + φ) на окне: плоские волны по осям сетки раскладываются в два
    матричных произведения, без косинуса в каждой точке."""
    k, amp = np.asarray(spec["k"]), np.asarray(spec["amp"])
    beta = 2 * np.pi * ((k @ w.q0)[:, None] + (k @ w.qr)[:, None] * w.r[None, :])
    beta += np.asarray(spec["phase"])[:, None]
    alpha = 2 * np.pi * (k @ w.qc)[:, None] * w.c[None, :]
    cb, sb = (
        (amp[:, None] * np.cos(beta)).astype(np.float32),
        (amp[:, None] * np.sin(beta)).astype(np.float32),
    )
    return cb.T @ np.cos(alpha).astype(np.float32) - sb.T @ np.sin(alpha).astype(np.float32)


def _ellipsoid(prim: dict, q: list, w: _Window):
    rho = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
    return rho <= 1, rho


def _cylinder(prim: dict, q: list, w: _Window):
    """Цилиндр по оси q3 (|q3| ≤ 1), радиус 1 + taper·q3 — конус или усечённый конус."""
    # |taper| < 1: радиус положителен на всей длине; за торцами (|q3| > 1) точки и так вне
    rho = np.hypot(q[0], q[1]) / np.maximum(1 + prim["taper"] * q[2], 1e-3)
    return (np.abs(q[2]) <= 1) & (rho <= 1), rho


def _torus(prim: dict, q: list, w: _Window):
    rho = np.hypot(np.hypot(q[0], q[1]) - 1, q[2]) / prim["minor"]
    return rho <= 1, rho


def _path(path: dict, t: np.ndarray, strand: int, strands: int) -> tuple:
    """Смещение осевой линии профиля на глубине t: нет, винтовая линия или «змейка»."""
    if path["kind"] == "helix":
        a = path["freq"] * t + path["phase"] + 2 * np.pi * strand / strands
        return path["radius"] * np.cos(a), path["radius"] * np.sin(a)
    if path["kind"] == "wiggle":
        (ax, ay), (fx, fy), (px, py) = path["amp"], path["freq"], path["phase"]
        return ax * np.sin(fx * t + px), ay * np.sin(fy * t + py)
    return 0.0, 0.0


def _morph(prim: dict, q: list, w: _Window):
    """Профиль вдоль оси q3 ∈ [−1, 1], плавно меняющийся по глубине: масштаб exp(s₁t + s₂t²),
    скрутка, вытянутость, переход формы (суперэллипс |u|ⁿ + |v|ⁿ ≤ 1 с n(t) или звезда с
    амплитудой лучей A(t)) и смещение осевой линии; несколько прядей — копии со сдвигом фазы."""
    m = prim["morph"]
    t = q[2]
    s = (np.clip(t, -1, 1) + 1) / 2
    f = np.exp(m["scale"][0] * t + m["scale"][1] * t**2)
    th = m["twist"][0] + m["twist"][1] * t
    e = np.exp(m["aspect"][0] + m["aspect"][1] * t)
    cos, sin = np.cos(th), np.sin(th)
    rho = np.full(t.shape, np.inf, t.dtype)
    for strand in range(m["strands"]):
        cx, cy = _path(m["path"], t, strand, m["strands"])
        u0, v0 = (q[0] - cx) / f, (q[1] - cy) / f
        u, v = (cos * u0 + sin * v0) / e, (cos * v0 - sin * u0) * e
        if m["profile"] == "superellipse":
            n = np.exp(np.log(m["n"][0]) + np.log(m["n"][1] / m["n"][0]) * s)
            r = (np.abs(u) ** n + np.abs(v) ** n) ** (1 / n)
        else:
            amp = m["amp"][0] + (m["amp"][1] - m["amp"][0]) * s
            r = np.hypot(u, v) * (1 + amp) / (1 + amp * np.cos(m["lobes"] * np.arctan2(v, u)))
        rho = np.minimum(rho, r)
    return (np.abs(t) <= 1) & (rho <= 1), rho


def _blob(prim: dict, q: list, w: _Window):
    """Изоповерхность гладкого 3D-шума g = поле − β|q|² ≥ порог; β держит объект компактным,
    membrane > 0 оставляет только стенку толщиной membrane в единицах поля."""
    r2 = q[0] ** 2 + q[1] ** 2 + q[2] ** 2
    g = _field(prim["field"], w) - prim["envelope"] * r2 - prim["threshold"]
    inside = g >= 0
    if prim["membrane"]:
        inside &= g <= prim["membrane"]
    return inside, np.minimum(np.sqrt(r2), 1)


def _morph_extent(prim: dict) -> tuple:
    m = prim["morph"]
    path = m["path"]
    offset = {"helix": lambda: path.get("radius", 0.0), "wiggle": lambda: max(path["amp"])}
    lateral = math.exp(abs(m["scale"][0]) + max(m["scale"][1], 0.0))
    lateral *= math.exp(abs(m["aspect"][0]) + abs(m["aspect"][1])) * math.sqrt(2)
    lateral += offset.get(path["kind"], lambda: 0.0)()
    return lateral, lateral, 1.0


def _blob_extent(prim: dict) -> tuple:
    reach = sum(abs(a) for a in prim["field"]["amp"]) - prim["threshold"]
    e = math.sqrt(max(reach, 0.0) / prim["envelope"])
    return e, e, e


SHAPES = {
    "ellipsoid": _ellipsoid, "cylinder": _cylinder, "torus": _torus, "morph": _morph,
    "blob": _blob,
}  # fmt: skip
# Полуширины локального бокса, вне которого формы нет: по ним рендер отсекает пустые окна
EXTENTS = {
    "ellipsoid": lambda p: (1.0, 1.0, 1.0),
    "cylinder": lambda p: (1 + abs(p["taper"]), 1 + abs(p["taper"]), 1.0),
    "torus": lambda p: (1 + p["minor"], 1 + p["minor"], p["minor"]),
    "morph": _morph_extent,
    "blob": _blob_extent,
}


def _profile(prim: dict, q: list, rho: np.ndarray, w: _Window):
    """Распределение яркости внутри примитива, множитель в [0, 1]."""
    p = prim["profile"]
    if p["kind"] == "gradient":  # линейно вдоль направления в локальных координатах
        d = p["dir"]
        s = np.clip(d[0] * q[0] + d[1] * q[1] + d[2] * q[2], -1, 1)
        return 1 - p["depth"] / 2 + p["depth"] / 2 * s
    if p["kind"] == "radial":  # спад от сердцевины к границе
        return 1 - p["depth"] * np.clip(rho, 0, 1) ** p["power"]
    if p["kind"] == "texture":  # объёмная текстура, движется вместе с объектом
        return 1 - p["depth"] + p["depth"] * (0.5 + 0.5 * np.tanh(_field(p["field"], w)))
    return 1.0


# ---------- Рендер ----------


def _affine(prim: dict, transform: dict) -> tuple[np.ndarray, np.ndarray]:
    """q = B·p + b: физическая точка → локальные координаты примитива с учётом transform —
    движения всего объекта p ↦ pivot + Q(p_obj − pivot) + shift."""
    a = np.linalg.inv(np.asarray(prim["matrix"]))
    c = np.asarray(prim["center"])
    q_inv = np.linalg.inv(np.asarray(transform["rotation"]))
    o, t = np.asarray(transform["pivot"]), np.asarray(transform["shift"])
    b_mat = a @ q_inv
    return b_mat, a @ (o - c) - b_mat @ (o + t)


def _span(coords: np.ndarray, center: float, half: float) -> slice | None:
    lo = int(np.searchsorted(coords, center - half, "left"))
    hi = int(np.searchsorted(coords, center + half, "right"))
    return slice(lo, hi) if hi > lo else None


def _window(prim: dict, b_mat: np.ndarray, b: np.ndarray, grid: _Grid) -> _Window | None:
    inv = np.linalg.inv(b_mat)
    p0 = -inv @ b
    half = np.abs(inv) @ np.asarray(EXTENTS[prim["shape"]](prim))
    if abs(grid.fixed - p0[grid.fixed_axis]) > half[grid.fixed_axis]:
        return None
    rows = _span(grid.rows, p0[grid.row_axis], half[grid.row_axis])
    cols = _span(grid.cols, p0[grid.col_axis], half[grid.col_axis])
    if rows is None or cols is None:
        return None
    q0 = b_mat[:, grid.fixed_axis] * grid.fixed + b
    return _Window(
        rows, cols, grid.rows[rows], grid.cols[cols], q0,
        b_mat[:, grid.row_axis], b_mat[:, grid.col_axis],
    )  # fmt: skip


def _evaluate(desc: dict, grid: _Grid) -> np.ndarray:
    """V на точках сетки до тона: фон, объединение частей по максимуму, затем вырезы."""
    bg = float(desc["background"])
    v = np.full((grid.rows.size, grid.cols.size), bg, np.float32)
    for op in ("add", "cut"):
        for prim in desc["primitives"]:
            if prim["op"] != op:
                continue
            b_mat, b = _affine(prim, desc["transform"])
            w = _window(prim, b_mat, b, grid)
            if w is None:
                continue
            q = w.local()
            inside, rho = SHAPES[prim["shape"]](prim, q, w)
            if prim["hollow"]:
                inside &= rho >= 1 - prim["hollow"]
            region = v[w.rows, w.cols]  # вид на v: запись идёт в v
            if op == "cut":
                region[inside] = bg
            else:
                value = prim["level"] * _profile(prim, q, rho, w)
                np.maximum(region, np.where(inside, value, 0.0), out=region)
    return v


def tone(v: np.ndarray, spec: dict) -> np.ndarray:
    """Тон объекта целиком: гамма, затем инверсия — одинаково во всех плоскостях."""
    v = np.clip(v, 0.0, 1.0) ** spec["gamma"]
    return 1 - v if spec["invert"] else v


def to_u8(v: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(v) * 255), 0, 255).astype(np.uint8)


def render(desc: dict, setup: Setup = TASK, supersample: int = SUPERSAMPLE) -> np.ndarray:
    """Сечения объекта float [P, n, n] ∈ [0, 1]: плоскость k — на своей сетке шага λz_k/(NΔx),
    пиксель усредняет supersample² подотсчётов; размытие краёв — гауссово с σ в мм
    (в пикселях плоскости k это σ/p_k), считается на сетке с запасом и обрезается."""
    blur = float(desc["blur"])
    out = []
    for pl in planes(setup):
        pad = math.ceil(3 * blur / pl.pitch) if blur else 0
        axis = pixel_axis(pl.n, pl.pitch, supersample, pad)
        v = _evaluate(desc, _Grid(2, pl.z, 1, axis, 0, axis))
        m = pl.n + 2 * pad
        v = v.reshape(m, supersample, m, supersample).mean((1, 3))
        if blur:
            v = ndimage.gaussian_filter(v, blur / pl.pitch, mode="nearest")
            v = v[pad : pad + pl.n, pad : pad + pl.n]
        out.append(v)
    return tone(np.stack(out), desc["tone"])


def side_view(
    desc: dict, setup: Setup = TASK, rows: int = 128, half_width: float = 0.78, ys: int = 160
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вид сбоку для проверки геометрии: максимум V по y ∈ [−half_width, half_width] на сетке
    (x — строки, z — столбцы) с одинаковым шагом по x и z; без тона. → (вид, x, z), мм."""
    x = (np.arange(rows) + 0.5 - rows / 2) * (2 * half_width / rows)
    step = x[1] - x[0]
    z = np.arange(Z_LO - 0.3, Z_HI + 0.3, step)
    out = np.zeros((rows, z.size))
    for y in np.linspace(-half_width, half_width, ys):
        out = np.maximum(out, _evaluate(desc, _Grid(1, float(y), 0, x, 2, z)))
    return out, x, z


def background_u8(desc: dict) -> int:
    return int(to_u8(tone(np.array(float(desc["background"])), desc["tone"])))


def plane_stats(img_u8: np.ndarray, background: int) -> list[dict]:
    """По сечению: размах, доля пикселей объекта (отличие от фона ≥ MIN_RANGE) и статус —
    ok, sparse (почти пусто: < MIN_PIXELS пикселей отличаются от медианы), empty (однотонно
    и равно фону) или uniform (однотонно, плоскость внутри объекта)."""
    stats = []
    for s in np.asarray(img_u8).astype(np.int64):
        span = int(s.max() - s.min())
        deviating = int((np.abs(s - np.median(s)) >= MIN_RANGE).sum())
        if span >= MIN_RANGE:
            status = "ok" if deviating >= MIN_PIXELS else "sparse"
        else:
            status = "empty" if abs(np.median(s) - background) < MIN_RANGE else "uniform"
        coverage = float((np.abs(s - background) >= MIN_RANGE).mean())
        stats.append(
            {"status": status, "range": span, "coverage": round(coverage, 4),
             "mean": round(float(s.mean()), 2)}
        )  # fmt: skip
    return stats


# ---------- Случайные объекты ----------


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _orientation(rng: np.random.Generator, max_tilt: float, min_tilt: float = 0.0):
    """Поворот, наклоняющий локальную ось q3 от z на угол из [min_tilt, max_tilt] градусов."""
    tilt = math.radians(rng.uniform(min_tilt, max_tilt))
    azimuth, spin = rng.uniform(0, 2 * np.pi, 2)
    rot = _rot_z(azimuth) @ _rot_y(tilt) @ _rot_z(spin)
    return rot, {
        "tilt_deg": round(math.degrees(tilt), 2),
        "azimuth_deg": round(math.degrees(azimuth), 1),
    }


def _modes(rng: np.random.Generator, count: int, std) -> dict:
    """Гладкое поле из count плоских волн, дисперсия ≈ 1; std — разброс частот по осям
    (периодов на единицу локальной координаты)."""
    k = rng.normal(0, 1, (count, 3)) * np.asarray(std, float)
    return {
        "k": k.tolist(),
        "phase": rng.uniform(0, 2 * np.pi, count).tolist(),
        "amp": [math.sqrt(2 / count)] * count,
    }


def _brightness(rng: np.random.Generator) -> tuple[float, dict]:
    level = float(rng.uniform(0.35, 1.0))
    kind = str(rng.choice(["flat", "gradient", "radial", "texture"], p=[0.4, 0.2, 0.2, 0.2]))
    if kind == "gradient":  # в половине случаев — вдоль оси объекта: яркость меняется по глубине
        d = (
            np.array([0.0, 0.0, rng.choice([-1.0, 1.0])])
            if rng.random() < 0.5
            else rng.normal(size=3)
        )
        return level, {"kind": kind, "dir": (d / np.linalg.norm(d)).tolist(),
                       "depth": float(rng.uniform(0.3, 0.9))}  # fmt: skip
    if kind == "radial":
        return level, {"kind": kind, "depth": float(rng.uniform(0.3, 0.9)),
                       "power": float(rng.uniform(0.7, 3.0))}  # fmt: skip
    if kind == "texture":
        f = _log_uniform(rng, 1.0, 4.0)
        return level, {"kind": kind, "depth": float(rng.uniform(0.3, 0.8)),
                       "field": _modes(rng, 16, (f, f, f))}  # fmt: skip
    return level, {"kind": kind}


def _prim(rng, shape: str, center, matrix: np.ndarray, params: dict, **fields) -> dict:
    level, profile = _brightness(rng)
    return {
        "shape": shape, "op": "add", "center": [float(c) for c in center],
        "matrix": np.asarray(matrix, float).tolist(), "level": level, "profile": profile,
        "hollow": 0.0, **fields, "params": params,
    }  # fmt: skip


def _z_center(rng: np.random.Generator, span: bool) -> float:
    """span — объект должен пересечь все плоскости: центр у середины; иначе — где угодно."""
    return Z_MID + rng.uniform(-0.35, 0.35) if span else rng.uniform(Z_LO, Z_HI)


def _ellipsoids(rng: np.random.Generator, span: bool, hollow: bool = False) -> list[dict]:
    if rng.random() < 0.25:  # изотропная сфера; при span граница сечений проходит по полю
        radius = rng.uniform(1.9, 3.2) if span else _log_uniform(rng, 0.15, 1.5)
        zc = _z_center(rng, span)
        r = np.sqrt(np.maximum(radius**2 - (Z_PLANES - zc) ** 2, 0))
        d = rng.uniform(max(r.min() - 0.45, 0), r.max() + 0.45) if span else rng.uniform(0, 0.4)
        a = rng.uniform(0, 2 * np.pi)
        center, matrix = (d * math.cos(a), d * math.sin(a), zc), radius * np.eye(3)
        params = {"sphere": True, "radius_mm": round(radius, 4)}
        shell = _log_uniform(rng, 0.02, 0.2) / radius
    else:  # эллипсоид: поперечные полуоси a, b, по z полуширина H
        a = _log_uniform(rng, 0.15, 0.95)
        b = a * math.exp(rng.uniform(-0.5, 0.5))
        rot, params = _orientation(rng, 30)
        h = rng.uniform(1.95, 3.0) if span else _log_uniform(rng, 0.2, 2.5)
        c = math.sqrt(max(h**2 - (rot[2, 0] * a) ** 2 - (rot[2, 1] * b) ** 2, 0.05**2))
        c /= abs(rot[2, 2])
        center = (*rng.uniform(-0.35, 0.35, 2), _z_center(rng, span))
        matrix = rot @ np.diag([a, b, c])
        params |= {"sphere": False, "radii_mm": [round(a, 4), round(b, 4), round(c, 4)]}
        shell = rng.uniform(0.06, 0.3)
    prim = _prim(rng, "ellipsoid", center, matrix, params)
    if hollow:
        prim["hollow"] = float(shell)
    return [prim]


def _shells(rng: np.random.Generator, span: bool) -> list[dict]:
    return _ellipsoids(rng, span, hollow=True)


def _cylinders(rng: np.random.Generator, span: bool) -> list[dict]:
    """Наклонный цилиндр, конус или трубка эллиптического сечения."""
    a = _log_uniform(rng, 0.05, 0.45)
    b = a * math.exp(rng.uniform(-0.4, 0.4))
    rot, params = _orientation(rng, 25) if rng.random() < 0.8 else _orientation(rng, 50, 25)
    through = span and rng.random() < 0.85  # торцы за крайними плоскостями
    half = rng.uniform(2.6, 4.0) if through else _log_uniform(rng, 0.4, 2.4)
    half /= math.cos(math.radians(params["tilt_deg"]))
    taper = float(rng.uniform(-0.6, 0.6)) if rng.random() < 0.4 else 0.0
    center = (*rng.uniform(-0.3, 0.3, 2), _z_center(rng, span))
    params |= {"radii_mm": [round(a, 4), round(b, 4)], "half_length_mm": round(half, 4)}
    prim = _prim(rng, "cylinder", center, rot @ np.diag([a, b, half]), params, taper=taper)
    if rng.random() < 0.5:
        prim["hollow"] = float(rng.uniform(0.2, 0.7))
    return [prim]


def _tori(rng: np.random.Generator, span: bool) -> list[dict]:
    """Тор любой ориентации, вытянутый по z до полуширины H (вытяжка — depth_stretch)."""
    major = rng.uniform(0.25, 0.6)
    minor = float(rng.uniform(0.15, 0.45))  # в долях большого радиуса
    tilt = math.degrees(math.acos(rng.uniform(0, 1)))  # ось тора равномерно по сфере
    rot, params = _orientation(rng, tilt, tilt)
    reach = major * (math.hypot(rot[0, 2], rot[1, 2]) + minor)  # полуширина по z без вытяжки
    h = rng.uniform(1.9, 2.8) if span else _log_uniform(rng, 0.3, 2.5)
    stretch = min(max(h / reach, 1.0), 12.0)
    matrix = np.diag([1.0, 1.0, stretch]) @ rot * major
    center = (*rng.uniform(-0.25, 0.25, 2), _z_center(rng, span))
    params |= {"major_mm": round(major, 4), "minor_ratio": round(minor, 4),
               "depth_stretch": round(stretch, 4)}  # fmt: skip
    prim = _prim(rng, "torus", center, matrix, params, minor=minor)
    if rng.random() < 0.2:
        prim["hollow"] = float(rng.uniform(0.3, 0.6))
    return [prim]


def _morphs(rng: np.random.Generator, span: bool) -> list[dict]:
    """Форма, плавно меняющаяся по глубине: профиль вдоль почти продольной оси."""
    r0 = _log_uniform(rng, 0.12, 0.5)
    rot, params = _orientation(rng, 15)
    kind = str(rng.choice(["none", "helix", "wiggle"], p=[0.45, 0.35, 0.2]))
    strands = 1
    path: dict = {"kind": kind}
    if kind == "helix":  # винтовая линия радиуса rad мм, freq рад на локальную единицу длины
        rad = rng.uniform(0.15, 0.45)
        strands = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
        r0 = min(r0, rad / 1.3, rad * math.sin(math.pi / strands) / 1.1 if strands > 1 else rad)
        sign = rng.choice([-1.0, 1.0])
        path |= {"radius": rad / r0, "freq": float(sign * 2 * np.pi * rng.uniform(0.1, 0.45)),
                 "phase": float(rng.uniform(0, 2 * np.pi))}  # fmt: skip
    elif kind == "wiggle":
        path |= {"amp": (rng.uniform(0, 0.25, 2) / r0).tolist(),
                 "freq": (2 * np.pi * rng.uniform(0.1, 0.4, 2)).tolist(),
                 "phase": rng.uniform(0, 2 * np.pi, 2).tolist()}  # fmt: skip
    through = span and rng.random() < 0.85
    half = rng.uniform(2.4, 3.5) if through else _log_uniform(rng, 0.4, 2.2)
    half /= math.cos(math.radians(params["tilt_deg"]))
    profile = str(rng.choice(["superellipse", "star"]))
    morph = {
        "profile": profile, "strands": strands, "path": path,
        "scale": rng.uniform([-0.7, -0.5], [0.7, 0.5]).tolist() if rng.random() < 0.6 else [0.0, 0.0],
        "twist": [float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(-2.0, 2.0))],
        "aspect": rng.uniform([-0.4, -0.3], [0.4, 0.3]).tolist(),
    }  # fmt: skip
    if profile == "superellipse":  # n: 0,6 — астроида, 1 — ромб, 2 — круг, 8 — почти квадрат
        morph["n"] = [_log_uniform(rng, 0.6, 8.0), _log_uniform(rng, 0.6, 8.0)]
    else:
        morph |= {"lobes": int(rng.integers(3, 8)), "amp": rng.uniform(0, 0.45, 2).tolist()}
    center = (*rng.uniform(-0.25, 0.25, 2), _z_center(rng, span))
    params |= {"radius_mm": round(r0, 4), "half_length_mm": round(half, 4)}
    prim = _prim(rng, "morph", center, rot @ np.diag([r0, r0, half]), params, morph=morph)
    if rng.random() < 0.3:
        prim["hollow"] = float(rng.uniform(0.2, 0.6))
    return [prim]


def _blobs(rng: np.random.Generator, span: bool) -> list[dict]:
    """Органический объект: изоповерхность гладкого 3D-шума в эллипсоидальной оболочке.
    Масштаб деталей поперёк 0,25–0,7 мм, вдоль z 1–4 мм: соседние сечения похожи, но меняются."""
    ex = rng.uniform(0.5, 1.0)
    ey = ex * math.exp(rng.uniform(-0.3, 0.3))
    ez = rng.uniform(2.0, 3.2) if span else _log_uniform(rng, 0.4, 2.5)
    rot, params = _orientation(rng, 15)
    lateral, depth = _log_uniform(rng, 0.25, 0.7), _log_uniform(rng, 1.0, 4.0)
    field = _modes(rng, 32, (ex / lateral, ey / lateral, ez / depth))
    membrane = float(rng.uniform(0.15, 0.4)) if rng.random() < 0.2 else 0.0
    center = (*rng.uniform(-0.15, 0.15, 2), _z_center(rng, span))
    params |= {"envelope_mm": [round(ex, 4), round(ey, 4), round(ez, 4)],
               "detail_mm": [round(lateral, 4), round(depth, 4)]}  # fmt: skip
    return [
        _prim(
            rng, "blob", center, rot @ np.diag([ex, ey, ez]), params, field=field,
            envelope=float(rng.uniform(0.5, 1.6)), threshold=float(rng.uniform(-1.3, 0.4)),
            membrane=membrane,
        )
    ]  # fmt: skip


def _particles(rng: np.random.Generator, span: bool) -> list[dict]:
    """Облако мелких вытянутых частиц по всему объёму: разреженные сечения."""
    prims = []
    for _ in range(int(_log_uniform(rng, 10, 60))):
        r = _log_uniform(rng, 0.015, 0.06)
        c = _log_uniform(rng, 0.25, 2.0)
        rot, params = _orientation(rng, 25)
        center = (*rng.uniform(-0.68, 0.68, 2), rng.uniform(Z_LO - 0.5, Z_HI + 0.5))
        matrix = rot @ np.diag([r, r * math.exp(rng.uniform(-0.3, 0.3)), c])
        prim = _prim(
            rng, "ellipsoid", center, matrix, params | {"radii_mm": [round(r, 4), round(c, 4)]}
        )
        if rng.random() < 0.5:  # «бусина» — спад яркости к краю
            prim["profile"] = {"kind": "radial", "depth": float(rng.uniform(0.5, 1.0)),
                               "power": float(rng.uniform(1.0, 3.0))}  # fmt: skip
        prims.append(prim)
    return prims


PARTS = ("ellipsoid", "shell", "cylinder", "torus", "morph")


def _composites(rng: np.random.Generator, span: bool) -> tuple[list[dict], str]:
    """Простые составные объекты: объединение 2–3 частей, пучок параллельных трубок или
    часть со сквозным отверстием."""
    layout = str(rng.choice(["union", "bundle", "cut"], p=[0.5, 0.25, 0.25]))
    if layout == "bundle":
        rot, params = _orientation(rng, 25)
        half = rng.uniform(2.6, 4.0) / math.cos(math.radians(params["tilt_deg"]))
        base = np.array([*rng.uniform(-0.2, 0.2, 2), _z_center(rng, span)])
        prims = []
        for _ in range(int(rng.integers(3, 8))):
            r = _log_uniform(rng, 0.03, 0.12)
            ang, dist = rng.uniform(0, 2 * np.pi), 0.45 * math.sqrt(rng.uniform())
            center = base + rot @ np.array([dist * math.cos(ang), dist * math.sin(ang), 0.0])
            prim = _prim(rng, "cylinder", center, rot @ np.diag([r, r, half]),
                         params | {"radius_mm": round(r, 4)}, taper=0.0)  # fmt: skip
            if rng.random() < 0.3:
                prim["hollow"] = float(rng.uniform(0.3, 0.6))
            prims.append(prim)
        return prims, layout
    if layout == "cut":
        prims = SAMPLERS[str(rng.choice(PARTS))](rng, span)
        r = _log_uniform(rng, 0.06, 0.25)
        rot, params = _orientation(rng, 90)
        center = np.array(prims[0]["center"]) + np.array([*rng.uniform(-0.2, 0.2, 2), 0.0])
        cut = _prim(rng, "cylinder", center, rot @ np.diag([r, r, 6.0]),
                    params | {"radius_mm": round(r, 4)}, taper=0.0)  # fmt: skip
        cut["op"] = "cut"
        return [*prims, cut], layout
    prims = []
    for _ in range(int(rng.integers(2, 4))):  # части уже и разнесены по полю
        part = SAMPLERS[str(rng.choice(PARTS))](rng, span)[0]
        s = rng.uniform(0.45, 0.75)
        part["matrix"] = (np.diag([s, s, 1.0]) @ np.asarray(part["matrix"])).tolist()
        part["center"][:2] = rng.uniform(-0.4, 0.4, 2).tolist()
        part["params"]["lateral_scale"] = round(s, 4)
        prims.append(part)
    return prims, layout


SAMPLERS = {
    "ellipsoid": _ellipsoids, "shell": _shells, "cylinder": _cylinders, "torus": _tori,
    "morph": _morphs, "blobs": _blobs, "particles": _particles, "composite": _composites,
}  # fmt: skip


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.bool_ | bool):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def random_object(rng: np.random.Generator, family: str, span: bool = True) -> dict:
    """Описание случайного объекта семейства family (без отбора по сечениям)."""
    out = SAMPLERS[family](rng, span)
    prims, layout = out if isinstance(out, tuple) else (out, None)
    desc = {"version": VERSION, "family": family}
    if layout:
        desc["layout"] = layout
    desc |= {
        "primitives": prims,
        "background": 0.0 if rng.random() < 0.7 else float(rng.uniform(0.02, 0.2)),
        "tone": {"gamma": _log_uniform(rng, 0.6, 1.6) if rng.random() < 0.3 else 1.0,
                 "invert": bool(rng.random() < 0.12)},
        "blur": float(rng.uniform(0.004, 0.015)) if rng.random() < 0.25 else 0.0,  # σ, мм
        "transform": copy.deepcopy(IDENTITY),
    }  # fmt: skip
    return _jsonable(desc)


def scale_lateral(desc: dict, s: float, about_center: bool = False) -> dict:
    """Тот же объект, растянутый поперёк в s раз: (x, y) → c + s·((x, y) − c), z без изменений;
    c — ось z или (about_center) средний центр частей объекта."""
    new = copy.deepcopy(desc)
    c = np.zeros(3)
    if about_center:
        c[:2] = np.mean([p["center"][:2] for p in desc["primitives"]], axis=0)
    d = np.diag([s, s, 1.0])
    for prim in new["primitives"]:
        prim["center"] = (c + d @ (np.asarray(prim["center"]) - c)).tolist()
        prim["matrix"] = (d @ np.asarray(prim["matrix"])).tolist()
    new["lateral_scale"] = {"factor": float(s), "about_mm": c[:2].tolist()}
    return new


def vary(desc: dict, rng: np.random.Generator) -> dict:
    """Близкий вариант: тот же объект, повёрнутый вокруг оси z (с отражением или без), чуть
    наклонённый и сдвинутый как целое, с немного другой гаммой."""
    q = _rot_z(rng.uniform(0, 2 * np.pi))
    if rng.random() < 0.5:
        q = q @ np.diag([-1.0, 1.0, 1.0])
    q = q @ _orientation(rng, 4)[0]
    shift = [*rng.normal(0, 0.06, 2), rng.uniform(-0.2, 0.2)]
    new = copy.deepcopy(desc)
    new["transform"] = {"rotation": q.tolist(), "shift": shift, "pivot": [0.0, 0.0, Z_MID]}
    new["tone"]["gamma"] = desc["tone"]["gamma"] * _log_uniform(rng, 0.85, 1.18)
    return _jsonable(new)


class VolumeGenerator:
    """Объект → uint8 [4, n, n]: сечения, прошедшие отбор (edge=True — крайние случаи:
    хотя бы одно сечение пустое, сплошное или почти пустое и хотя бы одно обычное)."""

    def __init__(
        self,
        families: dict[str, float] | None = None,
        setup: Setup = TASK,
        supersample: int = SUPERSAMPLE,
        edge: bool = False,
    ):
        weights = {k: float(v) for k, v in (families or DEFAULT_FAMILIES).items() if v > 0}
        unknown = set(weights) - set(FAMILIES)
        if unknown or not weights:
            raise ValueError(f"неизвестные семейства объектов: {unknown}; есть: {FAMILIES}")
        self.families = list(weights)
        self.p = np.array([weights[k] for k in self.families]) / sum(weights.values())
        self.weights = weights
        self.setup, self.supersample, self.edge = setup, int(supersample), bool(edge)

    def render(self, desc: dict) -> np.ndarray:
        return to_u8(render(desc, self.setup, self.supersample))

    def accepts(self, img: np.ndarray, desc: dict, need: str | None = None) -> bool:
        """Основной набор — все сечения ok; крайние случаи — есть ok и есть вырожденное
        (need — какое именно)."""
        status = [s["status"] for s in plane_stats(img, background_u8(desc))]
        if not self.edge:
            return set(status) == {"ok"}
        return "ok" in status and set(status) != {"ok"} and (need is None or need in status)

    def split_family(self, split: str, obj: int) -> str:
        """Семейство объекта obj разбиения split: последовательность золотого сечения по
        накопленным долям — в любом начальном отрезке разбиения доли близки к весам."""
        u = ((obj + 1) * GOLDEN + 0.25 * SPLITS[split]) % 1.0
        k = int(np.searchsorted(np.cumsum(self.p), u, side="right"))
        return self.families[min(k, len(self.families) - 1)]

    def sample(
        self, rng: np.random.Generator, family: str | None = None
    ) -> tuple[np.ndarray, dict]:
        """(uint8 [4, n, n], описание). Семейство — заданное или случайное, выбирается один раз:
        отбор не меняет доли. Крайний случай сначала выбирает вид вырожденного сечения
        (EDGE_KINDS) и строит попытки под него: пустое — объект не обязан пересекать все
        плоскости; сплошное — объект растянут поперёк с однородной яркостью; почти пустое —
        сжат. Если вид за EDGE_TARGET_TRIES попыток не вышел, годится любой крайний случай."""
        if family is None:
            family = self.families[rng.choice(len(self.families), p=self.p)]
        elif family not in self.families:
            raise ValueError(f"семейство {family!r} не из {self.families}")
        target = None
        if self.edge:
            target = str(rng.choice(list(EDGE_KINDS), p=list(EDGE_KINDS.values())))
        for attempt in range(MAX_TRIES):
            desc = random_object(rng, family, span=target != "empty")
            if target == "uniform":
                desc = scale_lateral(desc, _log_uniform(rng, 2.5, 5.0), about_center=True)
                for prim in desc["primitives"]:
                    prim["profile"] = {"kind": "flat"}
            elif target == "sparse":
                desc = scale_lateral(desc, _log_uniform(rng, 0.06, 0.2))
            img = self.render(desc)
            if self.accepts(img, desc, target if attempt < EDGE_TARGET_TRIES else None):
                return img, desc
        raise RuntimeError(f"семейство {family}: {MAX_TRIES} объектов подряд не прошли отбор")

    def vary(self, desc: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        """Близкий вариант, прошедший тот же отбор; если не вышло — сам объект."""
        for _ in range(MAX_TRIES):
            new = vary(desc, rng)
            img = self.render(new)
            if self.accepts(img, new):
                return img, new
        return self.render(desc), desc


# ---------- Разбиения ----------


def object_key(split: str, obj: int, variant: int = 0, seed: int = SEED) -> list[int]:
    """Сид объекта obj разбиения split (и его варианта ≥ 1) для np.random.default_rng."""
    if split not in SPLITS:
        raise ValueError(f"разбиение — одно из {tuple(SPLITS)}, получено {split!r}")
    key = [int(seed), SPLITS[split], int(obj)]
    return key + [int(variant)] if variant else key


def item_meta(
    gen: VolumeGenerator, split: str, index: int, variants: int, seed: int, img: np.ndarray,
    desc: dict,
) -> dict:  # fmt: skip
    """Метаданные примера: всё, чтобы воспроизвести его из сида или из описания объекта."""
    obj, var = divmod(index, variants)
    stats = plane_stats(img, background_u8(desc))
    setup = gen.setup
    return {
        "id": f"{split}/{index:06d}", "split": split, "index": index, "object_index": obj,
        "variant": var, "family": desc["family"],
        "seed": {"seed": seed, "variants": variants, "object_key": object_key(split, obj, 0, seed),
                 "variant_key": object_key(split, obj, var, seed) if var else None},
        "planes": [
            {"plane": k + 1, "z_mm": round(pl.z, 6), "pitch_um": round(pl.pitch * 1e3, 6),
             "fov_mm": round(pl.fov, 6), "file": f"plane_{k + 1}.png", **stats[k]}
            for k, pl in enumerate(planes(setup))
        ],
        "grid": {"pixel": "строка i, столбец j ↔ x = (j − N/2)·p, y = (i − N/2)·p, z = z_k",
                 "units": "мм", "n": setup.n, "supersample": gen.supersample},
        "optics": {"wavelength_m": setup.wavelength, "pixel_m": setup.pixel,
                   "distances_m": list(setup.distances)},
        "generator": {"module": __name__, "version": VERSION, "families": gen.weights,
                      "edge": gen.edge},
        "object": desc,
        "sha256": hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest(),
    }  # fmt: skip


def dataset_item(
    split: str,
    index: int,
    variants: int = 1,
    seed: int = SEED,
    families: dict[str, float] | None = None,
    setup: Setup = TASK,
    supersample: int = SUPERSAMPLE,
) -> tuple[np.ndarray, dict]:
    """Пример index разбиения split: объект index // variants, вариант index % variants
    (0 — сам объект). → (uint8 [4, n, n], метаданные)."""
    if variants < 1 or index < 0:
        raise ValueError("variants ≥ 1, index ≥ 0")
    gen = VolumeGenerator(families, setup, supersample, edge=split == "edge")
    obj, var = divmod(index, variants)
    rng = np.random.default_rng(object_key(split, obj, 0, seed))
    img, desc = gen.sample(rng, gen.split_family(split, obj))
    if var:
        img, desc = gen.vary(desc, np.random.default_rng(object_key(split, obj, var, seed)))
    return img, item_meta(gen, split, index, variants, seed, img, desc)
