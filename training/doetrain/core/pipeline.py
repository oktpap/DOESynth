"""Сквозной сценарий: 4 цели → фаза ДОЭ каждым методом → сдаваемые PNG и метрики.

CLI и окно вызывают только этот модуль, поэтому числа у них совпадают. Метрики всегда
считаются по сдаваемому пути: фаза → 8-битный код PNG → фаза из кода → |K|^power → 8 бит
(максимум → 255) → КК и ИСС против исходных 8-битных целей. power — конвенция изображения
(optics.IMAGE_POWER); сдаваемые PNG — в ней, метрики дополнительно и в другой (by_power).
"""

import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from doetrain.core import metrics, net
from doetrain.core.device import get_device
from doetrain.core.io import save_png
from doetrain.core.methods import adam, gs_global, superposition_init
from doetrain.core.optics import (
    IMAGE_POWER,
    POWER_KEY,
    POWER_LABEL,
    POWERS,
    TASK,
    Setup,
    check_power,
    chirps,
    reconstruct,
    target_amplitude,
)

# Фаза: (цели T ∈ [0, 1] [4, N, N], чирпы [4, N, N], seed, power) → непрерывная фаза [N, N]
PhaseFn = Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor]


@dataclass(frozen=True)
class Method:
    name: str
    title: str
    run: PhaseFn
    available: Callable[[int], bool] = field(default=lambda power: True)  # power → доступен?
    # Разовая подготовка вне замера времени (например, чтение весов с диска)
    prepare: Callable[[torch.device], None] = field(default=lambda device: None)


def _gs(images, chirp, seed, power):
    amp = target_amplitude(images, power)
    return gs_global(amp, chirp, 200, superposition_init(amp, chirp, seed))


def _adam(images, chirp, seed, power):
    p0 = superposition_init(target_amplitude(images, power), chirp, seed)
    return adam(images, chirp, 300, p0, power=power)


def _net(images, chirp, seed, power):
    return net.predict(images, power=power)


# Гибрид: фаза сети — старт того же Adam, что у метода adam (loss, конвенция, lr); отличаются
# только начальная фаза и число шагов. Веса сети не меняются (docs/decisions.md, гибрид).
HYBRID_STEPS = 25


def _hybrid(images, chirp, seed, power):
    return adam(images, chirp, HYBRID_STEPS, net.predict(images, power=power), power=power)


METHODS: dict[str, Method] = {
    m.name: m
    for m in (
        Method("gs", "GS (многоплоскостной, 200 итераций)", _gs),
        Method("adam", "Adam по фазе (Пирсон, 300 шагов)", _adam),
        Method("net", "Нейросеть", _net, net.available_for, net.prepare),
        Method(
            "hybrid",
            f"Сеть + Adam ({HYBRID_STEPS} шагов)",
            _hybrid,
            net.available_for,
            net.prepare,
        ),
    )
}


@dataclass
class Result:
    method: str
    doe_u8: np.ndarray  # [N, N] — сдаваемый ДОЭ, 0 ↔ −π, 255 ↔ +π
    recons_u8: np.ndarray  # [4, N, N] — сдаваемые восстановления
    metrics: dict
    seconds: float  # время синтеза фазы (медиана повторов), после прогрева
    total_seconds: float  # синтез + квантование, восстановление и метрики (без записи PNG)

    def summary(self) -> dict:
        return summary_row(self.method, self.seconds, self.metrics, self.total_seconds)


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def summary_row(method: str, seconds: float | None, m: dict, total: float | None = None) -> dict:
    """Строка сводной таблицы: конвенция, средние по плоскостям, разделение, пары по плоскостям
    и средние КК/ИСС в каждой конвенции (столбцы *_K — по |K|, *_K2 — по |K|²)."""
    by_power = {
        f"{name}_mean_{POWER_KEY[p]}": round(m["by_power"][POWER_KEY[p]][f"{name}_mean"], 4)
        for p in POWERS
        for name in ("cc", "ssim")
    }
    return {
        "method": method,
        "image": POWER_LABEL[m["power"]],
        "seconds": _round(seconds),
        "total_seconds": _round(total),
        "cc_mean": round(m["cc_mean"], 4),
        "ssim_mean": round(m["ssim_mean"], 4),
        "separation": round(m["separation"], 4),
        **{f"cc_{i}": round(v, 4) for i, v in enumerate(m["cc"])},
        **{f"ssim_{i}": round(v, 4) for i, v in enumerate(m["ssim"])},
        **by_power,
    }


def _score(targets_u8: np.ndarray, recons_u8: np.ndarray) -> dict:
    s = metrics.score(targets_u8, recons_u8)
    cc = np.array(s["cc_matrix"])
    off = cc[~np.eye(len(cc), dtype=bool)]
    s["cc_mean"] = float(np.mean(s["cc"]))
    s["ssim_mean"] = float(np.mean(s["ssim"]))
    s["separation"] = float(np.mean(s["cc"]) - off.mean())  # диагональ минус вне диагонали
    return s


def _brief(s: dict) -> dict:
    keys = ("cc", "ssim", "cc_mean", "ssim_mean", "separation")
    return {k: s[k] for k in keys}


def _recons_u8(doe_u8: np.ndarray, power: int, setup: Setup) -> np.ndarray:
    """Сдаваемый путь от 8-битного кода ДОЭ: фаза из кода → |K|^power → 8 бит (максимум → 255)."""
    quantized = torch.from_numpy(metrics.u8_to_phase(doe_u8)).float()
    amp = reconstruct(quantized, chirps(setup)).numpy()
    return np.stack([metrics.image_to_u8(a**power) for a in amp])


def add_other_convention(
    score: dict, doe_u8: np.ndarray, targets_u8: np.ndarray, setup: Setup = TASK
) -> dict:
    """Дополняет метрики score (в конвенции score["power"]) справочными метриками другой
    конвенции: score["by_power"]["K" | "K2"]. Вынесено отдельно, чтобы не входить в замер
    полного времени получения сдаваемых файлов."""
    power, other = score["power"], 3 - score["power"]
    other_score = _score(targets_u8, _recons_u8(doe_u8, other, setup))
    by_power = {POWER_KEY[power]: _brief(score), POWER_KEY[other]: _brief(other_score)}
    return score | {"by_power": {POWER_KEY[p]: by_power[POWER_KEY[p]] for p in POWERS}}


def evaluate(
    phase: np.ndarray,
    targets_u8: np.ndarray,
    setup: Setup = TASK,
    power: int = IMAGE_POWER,
    *,
    other: bool = True,
) -> tuple:
    """Непрерывная фаза → (ДОЭ uint8, восстановления uint8 [4, N, N] в конвенции power, метрики).

    Метрики верхнего уровня — в конвенции power; при other = True ещё
    metrics["by_power"]["K" | "K2"] — краткие метрики в каждой конвенции, чтобы сравнение
    не зависело от спорного выбора.
    """
    doe_u8 = metrics.phase_to_u8(phase)
    recons_u8 = _recons_u8(doe_u8, check_power(power), setup)
    score = _score(targets_u8, recons_u8) | {"power": power}
    if other:
        score = add_other_convention(score, doe_u8, targets_u8, setup)
    return doe_u8, recons_u8, score


def control(targets_u8: np.ndarray, power: int = IMAGE_POWER) -> dict:
    """Контроль без голограммы: в каждой плоскости среднее четырёх целей.

    Пирсон к нему нечувствителен к разделению плоскостей, поэтому метод, не превзошедший
    контроль, ничего не доказывает, даже если обходит GS. От конвенции не зависит.
    """
    avg = np.rint(targets_u8.astype(np.float64).mean(0)).astype(np.uint8)
    s = _score(targets_u8, np.stack([avg] * len(targets_u8)))
    return s | {"power": check_power(power), "by_power": {POWER_KEY[p]: _brief(s) for p in POWERS}}


def available_methods(power: int = IMAGE_POWER) -> list[str]:
    return [name for name, m in METHODS.items() if m.available(power)]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


_WARM: set[tuple[str, str]] = set()  # (метод, устройство), уже прогретые в этом процессе


def synthesize(
    targets_u8: np.ndarray,
    methods: list[str],
    *,
    device: torch.device | None = None,
    seed: int = 0,
    warmup: bool = True,
    repeats: int = 1,
    power: int = IMAGE_POWER,
    on_result: Callable[[Result], None] | None = None,
) -> list[Result]:
    """Запускает методы по очереди на одном устройстве и возвращает сдаваемые результаты.

    power — конвенция изображения (|K| или |K|²): под неё оптимизируют GS и Adam, в ней
    пишутся PNG; сеть отказывается работать, если обучена под другую.
    Время честное для таблицы «качество против времени»: подготовка (чтение весов) — вне
    замера; первый вызов метода на устройстве в процессе — прогон без замера (разовые
    накладные расходы PyTorch завышают первый Adam примерно втрое); seconds — медиана
    repeats прогонов. Для таблицы защиты: repeats = 5, как в R01.
    """
    if repeats < 1:
        raise ValueError("repeats ≥ 1")
    check_power(power)
    targets_u8 = np.asarray(targets_u8, np.uint8)
    if targets_u8.shape != (len(TASK.distances), TASK.n, TASK.n):
        raise ValueError(f"цели должны быть {len(TASK.distances)}×{TASK.n}×{TASK.n} uint8")
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"неизвестные методы: {', '.join(unknown)}; есть: {', '.join(METHODS)}")
    device = device or get_device()
    images = torch.from_numpy(targets_u8.astype(np.float32) / 255).to(device)
    chirp = chirps(device=device)
    results = []
    if warmup and ("evaluate", "cpu") not in _WARM:  # первый вызов тянет импорт skimage и др.
        evaluate(np.zeros((TASK.n, TASK.n)), targets_u8)
        _WARM.add(("evaluate", "cpu"))
    for name in methods:
        method = METHODS[name]
        method.prepare(images.device)
        if warmup and (name, str(images.device)) not in _WARM:
            method.run(images, chirp, seed, power)
            _sync(device)
            _WARM.add((name, str(images.device)))
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            phase = method.run(images, chirp, seed, power)
            _sync(device)
            times.append(time.perf_counter() - t0)
        seconds = float(np.median(times))
        t0 = time.perf_counter()
        phase_np = phase.detach().cpu().double().numpy()
        doe_u8, recons_u8, score = evaluate(phase_np, targets_u8, power=power, other=False)
        total = seconds + time.perf_counter() - t0
        score = add_other_convention(score, doe_u8, targets_u8)  # справочно, вне замера
        result = Result(name, doe_u8, recons_u8, score, seconds, total)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def save(results: list[Result], targets_u8: np.ndarray, out: str | Path) -> Path:
    """Сдаваемые файлы: <out>/<метод>/{doe.png, recon_0..3.png, metrics.json},
    <out>/inputs/target_0..3.png (цели после приведения к условию), <out>/summary.{csv,json}."""
    out = Path(out)
    for i, t in enumerate(targets_u8):
        save_png(out / "inputs" / f"target_{i}.png", t)
    rows = []
    for r in results:
        d = out / r.method
        save_png(d / "doe.png", r.doe_u8)
        for i, rec in enumerate(r.recons_u8):
            save_png(d / f"recon_{i}.png", rec)
        (d / "metrics.json").write_text(
            json.dumps(
                {"seconds": r.seconds, "total_seconds": r.total_seconds, **r.metrics}, indent=1
            ),
            encoding="utf-8",
        )
        rows.append(r.summary())
    power = results[0].metrics["power"] if results else IMAGE_POWER
    rows.append(summary_row("control", None, control(targets_u8, power)))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return out
