"""Каталог методов: сети с весами и итерационные алгоритмы, их описания и параметры по умолчанию.

Числа качества и времени для подсказок интерфейса берутся из benchmark/summary.json
(его пишет benchmark/summarize.py); без файла каталог работает, просто без чисел.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
SUMMARY = ROOT / "benchmark" / "summary.json"

DIRECT = "Прямые сети: изображения → фаза"
PHYSICS = "Сети с физикой Френеля внутри"


@dataclass(frozen=True)
class NetModel:
    id: str
    title: str
    group: str
    description: str
    experiment: str

    @property
    def path(self) -> Path:
        return WEIGHTS / f"{self.id}.pt"


@dataclass(frozen=True)
class IterMethod:
    id: str
    title: str
    description: str
    default_iters: int
    max_iters: int
    unit: str = "итераций"


NET_MODELS: tuple[NetModel, ...] = (
    NetModel(
        "cghnet_gd8_r08",
        "3D-CGH-Net + 8 обучаемых шагов спуска",
        PHYSICS,
        "3D-CGH-Net с пропуском амплитуды цели и восемью шагами градиентного спуска "
        "по качеству восстановления внутри сети; длина шагов обучена вместе с сетью.",
        "R08",
    ),
    NetModel(
        "cghnet_paper_r08",
        "3D-CGH-Net",
        PHYSICS,
        "Архитектура статьи Rymov et al. (2025): общий кодировщик, ветви амплитуды и фазы "
        "полей четырёх плоскостей, перенос на ДОЭ точным обратным оператором Френеля.",
        "R08",
    ),
    NetModel(
        "cghnet_skip_r08",
        "3D-CGH-Net + пропуск амплитуды",
        PHYSICS,
        "3D-CGH-Net, амплитуда полей плоскостей строится от амплитуды цели (предложение "
        "авторов статьи); обучение с шумом 5 %.",
        "R08",
    ),
    NetModel(
        "cghnet_skip_clean_r08",
        "3D-CGH-Net + пропуск амплитуды, без шума",
        PHYSICS,
        "То же, что «3D-CGH-Net + пропуск амплитуды», обучение без шума.",
        "R08",
    ),
    NetModel(
        "cghnet_planes_r08",
        "3D-CGH-Net, loss по плоскостям",
        PHYSICS,
        "3D-CGH-Net, обученная на сумме ошибок по плоскостям вместо ошибки по всему объёму.",
        "R08",
    ),
    NetModel(
        "cghnet_planes_clean_r08",
        "3D-CGH-Net, loss по плоскостям, без шума",
        PHYSICS,
        "То же, что «3D-CGH-Net, loss по плоскостям», обучение без шума.",
        "R08",
    ),
    NetModel(
        "cghnet_gs4_r08",
        "3D-CGH-Net + 4 итерации GS",
        PHYSICS,
        "3D-CGH-Net с пропуском амплитуды и четырьмя итерациями проекции GS после блока переноса.",
        "R08",
    ),
    NetModel(
        "cghnet_pass2_r08",
        "3D-CGH-Net, второй проход",
        PHYSICS,
        "3D-CGH-Net, вторая сеть видит восстановления первого прохода и правит поля плоскостей.",
        "R08",
    ),
    NetModel(
        "cghnet_refine_r08",
        "3D-CGH-Net + кодировщик в плоскости ДОЭ",
        PHYSICS,
        "3D-CGH-Net с малой сетью, которая доводит комплексную сумму полей до чисто фазового ДОЭ.",
        "R08",
    ),
    NetModel(
        "cghnet_il2_r08",
        "3D-CGH-Net, чередование пикселей ×2",
        PHYSICS,
        "Облегчённая 3D-CGH-Net: вход и выход в виде 4 подрешёток 64×64, быстрее на CPU.",
        "R08",
    ),
    NetModel(
        "unrolled_r06",
        "Развёрнутый GS, 6 обучаемых итераций",
        PHYSICS,
        "Обученная суперпозиция и шесть итераций многоплоскостного GS, в каждой малая сеть "
        "правит поля плоскостей по целям и текущим восстановлениям.",
        "R06",
    ),
    NetModel(
        "superpos_r08",
        "Обученная суперпозиция",
        PHYSICS,
        "Сеть ставит каждой цели фазу и усиление в её плоскости, перенос на ДОЭ — точный "
        "обратный оператор Френеля, сумма полей доводится до фазового ДОЭ.",
        "R08",
    ),
    NetModel(
        "unet_r03",
        "U-Net, 60 тыс. шагов",
        DIRECT,
        "Прямой U-Net «4 изображения → фаза ДОЭ», 4 понижения разрешения, обучение без "
        "учителя через модель Френеля на широком генераторе картинок.",
        "R03",
    ),
    NetModel(
        "unet_r02",
        "U-Net, 20 тыс. шагов",
        DIRECT,
        "Тот же U-Net, более короткое обучение (20 тыс. шагов).",
        "R02",
    ),
    NetModel(
        "unet_volume_r07",
        "U-Net, обучен на 3D-сечениях",
        DIRECT,
        "U-Net, обученный на сечениях трёхмерных объектов в четырёх плоскостях.",
        "R07",
    ),
    NetModel(
        "yresnet_r13",
        "Y-ResNet",
        DIRECT,
        "Y-образная ResNet: общий кодировщик, ветви Re и Im комплексного поля ДОЭ, "
        "проекция на чисто фазовый ДОЭ.",
        "R13",
    ),
    NetModel(
        "phasenet3d_r12",
        "PhaseNet3D",
        DIRECT,
        "Объёмная сеть: Conv3d по четырём плоскостям, спектральные блоки, осевое внимание.",
        "R12",
    ),
)

ITER_METHODS: tuple[IterMethod, ...] = (
    IterMethod(
        "gs",
        "Герчберг–Сэкстон, многоплоскостной",
        "Классический GS для нескольких плоскостей (global GS): все плоскости проецируются "
        "параллельно, поля на ДОЭ складываются.",
        200,
        2000,
    ),
    IterMethod(
        "gs_seq",
        "Герчберг–Сэкстон, последовательный",
        "GS с обходом плоскостей по очереди; итерация — проход по всем четырём плоскостям.",
        50,
        1000,
    ),
    IterMethod(
        "adam",
        "Градиентный спуск по фазе (Adam)",
        "Прямая оптимизация фазы через модель Френеля со старта суперпозиции: сначала по "
        "корреляции восстановлений с целями, затем по корреляции и SSIM (как дооптимизация сети).",
        300,
        3000,
        "шагов",
    ),
)


@dataclass(frozen=True)
class Defaults:
    """Параметры по умолчанию; выбраны по бенчмарку (benchmark/README.md, summary.md)."""

    model: str = "cghnet_paper_r08"
    finetune: bool = True
    finetune_steps: int = 200
    iter_method: str = "gs"
    max_finetune_steps: int = 2000


DEFAULTS = Defaults()


def net_model(model_id: str) -> NetModel:
    for m in NET_MODELS:
        if m.id == model_id:
            return m
    raise KeyError(f"нет модели {model_id!r}")


def iter_method(method_id: str) -> IterMethod:
    for m in ITER_METHODS:
        if m.id == method_id:
            return m
    raise KeyError(f"нет метода {method_id!r}")


def benchmark_summary() -> dict:
    try:
        return json.loads(SUMMARY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def describe() -> dict:
    """Каталог для интерфейса: модели (только с весами), методы, умолчания и числа бенчмарка."""
    bench = benchmark_summary()
    by_model, by_method = bench.get("models", {}), bench.get("iterative", {})
    nets = [
        asdict(m) | {"default": m.id == DEFAULTS.model, "bench": by_model.get(m.id)}
        for m in NET_MODELS
        if m.path.is_file()
    ]

    def order(n: dict) -> tuple:  # группа «с физикой» первой; внутри — по SSIM бенчмарка
        ssim = (n["bench"] or {}).get("finetune", {}).get("ssim", 0.0)
        return (n["group"] != PHYSICS, not n["default"], -ssim)

    nets.sort(key=order)
    iters = [asdict(m) | {"bench": by_method.get(m.id)} for m in ITER_METHODS]
    return {
        "nets": nets,
        "iterative": iters,
        "defaults": asdict(DEFAULTS),
        "bench": bench.get("meta"),
    }
