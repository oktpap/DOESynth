"""Сквозной сценарий: 4 изображения → фаза ДОЭ выбранным методом → сдаваемые PNG и метрики.

Интерфейс и командная строка вызывают только этот модуль, поэтому числа у них совпадают.
Метрики считаются по сдаваемому пути: фаза → 8-битный ДОЭ (0 ↔ −π, 255 ↔ +π) → фаза из кода →
|K|² по формуле Френеля → 8 бит (максимум → 255) → SSIM условия и Пирсон против исходных
8-битных изображений.

Время: чтение весов и первый (прогревочный) вызов метода — вне замера. «Синтез» — время
получения фазы: сеть плюс дооптимизация или итерации; «полное» — ещё квантование,
восстановление и метрики.
"""

import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from doesynth import catalog, methods, metrics, models
from doesynth.catalog import DEFAULTS
from doesynth.optics import IMAGE_POWER, TASK, chirps, reconstruct, reconstruct_at, target_amplitude

LOCK = threading.Lock()  # один расчёт за раз: параллельные запросы портят замер времени
SEED = 0


def get_device() -> torch.device:
    want = os.environ.get("DOESYNTH_DEVICE", "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if want in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "cuda":
        raise RuntimeError("DOESYNTH_DEVICE=cuda, но CUDA недоступна")
    return torch.device("cpu")


DEVICE = get_device()


def device_label() -> str:
    if DEVICE.type == "cuda":
        return f"GPU · {torch.cuda.get_device_name(DEVICE)}"
    return f"CPU · {torch.get_num_threads()} потоков"


@dataclass(frozen=True)
class NetSpec:
    model: str = DEFAULTS.model
    finetune: bool = DEFAULTS.finetune
    steps: int = DEFAULTS.finetune_steps

    @property
    def label(self) -> str:
        title = catalog.net_model(self.model).title
        return f"{title} + дооптимизация, {self.steps} шагов" if self.finetune else title


@dataclass(frozen=True)
class IterSpec:
    method: str = DEFAULTS.iter_method
    iters: int | None = None

    @property
    def n(self) -> int:
        return self.iters or catalog.iter_method(self.method).default_iters

    @property
    def label(self) -> str:
        m = catalog.iter_method(self.method)
        return f"{m.title}, {self.n} {m.unit}"


@dataclass
class Result:
    kind: str  # "net" | "iter"
    label: str
    params: dict
    phase: np.ndarray  # непрерывная фаза [N, N] (до квантования)
    doe_u8: np.ndarray  # сдаваемый ДОЭ [N, N]
    recons_u8: np.ndarray  # сдаваемые восстановления [4, N, N]
    metrics: dict
    seconds: float  # синтез фазы
    total_seconds: float  # + квантование, восстановление и метрики
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "params": self.params,
            "seconds": round(self.seconds, 4),
            "total_seconds": round(self.total_seconds, 4),
            "image": "|K|^2 (интенсивность), 8 бит, максимум -> 255",
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.metrics.items()},
        }


def _sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def _chirp() -> torch.Tensor:
    return chirps(device=DEVICE)


def _net_phase(spec: NetSpec, images: torch.Tensor, chirp: torch.Tensor) -> torch.Tensor:
    model = models.load(catalog.net_model(spec.model).path, str(DEVICE))
    with torch.no_grad():
        phase = model(images[None])[0]
    if spec.finetune and spec.steps > 0:
        phase = methods.finetune(images, chirp, spec.steps, phase, power=IMAGE_POWER)
    return phase


def _iter_phase(spec: IterSpec, images: torch.Tensor, chirp: torch.Tensor) -> torch.Tensor:
    amp = target_amplitude(images)
    p0 = methods.superposition_init(amp, chirp, SEED)
    if spec.method == "gs":
        return methods.gs_global(amp, chirp, spec.n, p0)
    if spec.method == "gs_seq":
        return methods.gs_sequential(amp, chirp, spec.n, p0)
    if spec.method == "adam":  # та же схема, что у дооптимизации сети, но со старта суперпозиции
        return methods.finetune(images, chirp, spec.n, p0, power=IMAGE_POWER)
    raise ValueError(f"неизвестный метод {spec.method!r}")


_WARM: set[tuple] = set()


def _warmup(key: tuple, fn) -> None:
    """Первый вызов на устройстве тянет разовые накладные расходы PyTorch — вне замера."""
    if key not in _WARM:
        fn()
        _sync()
        _WARM.add(key)


def _recons(doe_u8: np.ndarray, power: int) -> np.ndarray:
    quantized = torch.from_numpy(metrics.u8_to_phase(doe_u8)).float()
    amp = reconstruct(quantized, chirps()).numpy()
    return np.stack([metrics.image_to_u8(a**power) for a in amp])


def evaluate(phase: np.ndarray, targets_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Непрерывная фаза → (ДОЭ uint8, восстановления uint8 [4, N, N], метрики)."""
    doe_u8 = metrics.phase_to_u8(phase)
    recons = _recons(doe_u8, IMAGE_POWER)
    return doe_u8, recons, metrics.score(targets_u8, recons)


def other_convention(doe_u8: np.ndarray, targets_u8: np.ndarray) -> dict:
    """Справочно: метрики того же ДОЭ, если восстановлением считать модуль |K|, а не |K|²
    (условие этого не уточняет). Вне замера времени."""
    s = metrics.score(targets_u8, _recons(doe_u8, 3 - IMAGE_POWER))
    return {k: s[k] for k in ("ssim", "pearson", "ssim_mean", "pearson_mean")}


def _check(targets_u8: np.ndarray) -> np.ndarray:
    targets_u8 = np.asarray(targets_u8, np.uint8)
    if targets_u8.shape != (len(TASK.distances), TASK.n, TASK.n):
        raise ValueError(f"нужно 4 изображения {TASK.n}×{TASK.n}")
    return targets_u8


def run(targets_u8: np.ndarray, spec: NetSpec | IterSpec) -> Result:
    targets_u8 = _check(targets_u8)
    images = torch.from_numpy(targets_u8.astype(np.float32) / 255).to(DEVICE)
    chirp = _chirp()
    with LOCK:
        if isinstance(spec, NetSpec):
            kind, fn = "net", lambda s: _net_phase(s, images, chirp)
            models.load(catalog.net_model(spec.model).path, str(DEVICE))  # чтение весов вне замера
            warm = NetSpec(spec.model, spec.finetune, min(spec.steps, 6))
            _warmup(("net", spec.model, spec.finetune), lambda: fn(warm))
            params = {
                "model": spec.model,
                "finetune": spec.finetune,
                "steps": spec.steps if spec.finetune else 0,
            }
            if spec.finetune:
                params["finetune_recipe"] = dict(methods.FINETUNE)
        else:
            kind, fn = "iter", lambda s: _iter_phase(s, images, chirp)
            warm = IterSpec(spec.method, min(spec.n, 6))
            _warmup(("iter", spec.method), lambda: fn(warm))
            params = {"method": spec.method, "iters": spec.n}
            if spec.method == "adam":
                params["finetune_recipe"] = dict(methods.FINETUNE)
        t0 = time.perf_counter()
        phase = fn(spec)
        _sync()
        seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        phase_np = phase.detach().cpu().double().numpy()
        doe_u8, recons_u8, score = evaluate(phase_np, targets_u8)
        total = seconds + time.perf_counter() - t1
        score["by_abs_K"] = other_convention(doe_u8, targets_u8)
    return Result(kind, spec.label, params, phase_np, doe_u8, recons_u8, score, seconds, total)


def control(targets_u8: np.ndarray) -> dict:
    """Контроль без голограммы: в каждой плоскости — среднее четырёх изображений."""
    targets_u8 = _check(targets_u8)
    avg = np.rint(targets_u8.astype(np.float64).mean(0)).astype(np.uint8)
    return metrics.score(targets_u8, np.stack([avg] * len(targets_u8)))


def slice_at(phase: np.ndarray, z_mm: float) -> np.ndarray:
    """Восстановление 8 бит из 8-битного ДОЭ на произвольном расстоянии z (мм)."""
    q = torch.from_numpy(metrics.u8_to_phase(metrics.phase_to_u8(phase))).float()
    amp = reconstruct_at(q, z_mm * 1e-3).numpy()
    return metrics.image_to_u8(amp**IMAGE_POWER)


def warm_default() -> None:
    """Прогрев модели по умолчанию и GS при старте сервера, чтобы первый запуск был быстрым."""
    dummy = np.zeros((4, TASK.n, TASK.n), np.uint8)
    dummy[:, 32:96, 32:96] = 255
    try:
        run(dummy, NetSpec())
        run(dummy, IterSpec())
    except Exception as e:  # прогрев не должен ронять сервер
        print(f"прогрев не удался: {e}", flush=True)
