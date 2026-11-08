"""Подбор критерия дооптимизации под SSIM на отдельной выборке настройки.

SSIM условия (глобальный, восстановление «максимум → 255») штрафует тёмные восстановления:
у спекла яркие пики, после нормировки на максимум среднее восстановления в 2–4 раза ниже,
чем у цели. Дооптимизация по одному Пирсону этого не видит. Здесь сравниваются варианты:
Пирсон; Пирсон + w·(1 − SSIM); два этапа — Пирсон, затем Пирсон + SSIM с малым lr;
градиент максимума через p-норму (soft). Старт — фаза сети.

Выборка настройки — не наборы отчёта: volume_validation (200) и 64 независимые четвёрки
генератора обучения с отдельным сидом (tune.npz, make_tune.py). Итоговый вариант затем
проверяется на наборах отчёта (benchmark.py).

    python benchmark/ssim_tuning.py --tune benchmark/tune.npz --model cghnet_paper_r08
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from doesynth import methods, models
from doesynth.optics import chirps

try:
    from benchmark.benchmark import net_phase, score_batch
except ImportError:  # запуск как скрипта из папки решения
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import net_phase, score_batch

# (имя, [этапы (kind, вес SSIM, lr, шаги, soft)])
VARIANTS = [
    ("pearson lr0.4 x50", [("pearson", 1, 0.4, 50, 0)]),
    ("pearson lr0.4 x100", [("pearson", 1, 0.4, 100, 0)]),
    ("pearson+ssim w1 lr0.05 x100", [("pearson_ssim", 1, 0.05, 100, 0)]),
    ("pearson+ssim w1 lr0.02 x100", [("pearson_ssim", 1, 0.02, 100, 0)]),
    ("pearson+ssim w1 soft8 lr0.05 x100", [("pearson_ssim", 1, 0.05, 100, 8)]),
    ("pearson+ssim w3 soft8 lr0.05 x100", [("pearson_ssim", 3, 0.05, 100, 8)]),
    ("ssim soft8 lr0.05 x100", [("ssim", 1, 0.05, 100, 8)]),
    (
        "2 этапа: P x50 lr0.4 → P+S w1 x50 lr0.02",
        [("pearson", 1, 0.4, 50, 0), ("pearson_ssim", 1, 0.02, 50, 0)],
    ),
    (
        "2 этапа: P x50 lr0.4 → P+S w1 soft8 x50 lr0.02",
        [("pearson", 1, 0.4, 50, 0), ("pearson_ssim", 1, 0.02, 50, 8)],
    ),
    (
        "2 этапа: P x50 lr0.4 → P+S w3 soft8 x50 lr0.02",
        [("pearson", 1, 0.4, 50, 0), ("pearson_ssim", 3, 0.02, 50, 8)],
    ),
    (
        "2 этапа: P x50 lr0.4 → S soft8 x50 lr0.02",
        [("pearson", 1, 0.4, 50, 0), ("ssim", 1, 0.02, 50, 8)],
    ),
    (
        "2 этапа: P x50 lr0.4 → P+S w1 soft8 x50 lr0.05",
        [("pearson", 1, 0.4, 50, 0), ("pearson_ssim", 1, 0.05, 50, 8)],
    ),
    (
        "2 этапа: P x50 lr0.4 → P+S w1 soft32 x50 lr0.02",
        [("pearson", 1, 0.4, 50, 0), ("pearson_ssim", 1, 0.02, 50, 32)],
    ),
]


# Раунд 2: двухэтапная схема methods.finetune, 100 шагов; доля первого этапа, lr второго, p
VARIANTS_R2 = [
    (f"finetune split{split} lr2={lr2} p{soft}", {"split": split, "lr2": lr2, "soft": soft})
    for split in (0.3, 0.5, 0.7)
    for lr2 in (0.05, 0.1)
    for soft in (4.0, 8.0, 16.0)
]


def run_variant(images, chirp, phase0, stages) -> torch.Tensor:
    if isinstance(stages, dict):
        return methods.finetune(images, chirp, 100, phase0, **stages)
    phase = phase0
    for kind, w, lr, k, soft in stages:
        phase = methods.adam(images, chirp, k, phase, kind=kind, lr=lr, ssim_weight=w, soft=soft)
    return phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", type=Path, default=Path("benchmark/tune.npz"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--model", nargs="+", default=["cghnet_paper_r08"])
    parser.add_argument("--out", type=Path, default=Path("benchmark/ssim_tuning.json"))
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.tune)
    targets = data["images"]
    labels = data["set"]
    images = torch.from_numpy(targets.astype(np.float32) / 255).to(device)
    chirp = chirps(device=device)
    out = {}
    t0 = time.time()
    for name in args.model:
        model = models.load((args.weights / f"{name}.pt").resolve(), str(device))
        phase0 = net_phase(model, images)
        for label, stages in VARIANTS if args.round == 1 else VARIANTS_R2:
            arr = score_batch(run_variant(images, chirp, phase0, stages), targets, chirp)
            groups = [arr[labels == g] for g in np.unique(labels)]  # группы — с равным весом
            out[f"{name} | {label}"] = {
                "ssim": float(np.mean([g[:, 0].mean() for g in groups])),
                "pearson": float(np.mean([g[:, 1].mean() for g in groups])),
                "ssim_se": float(np.sqrt(sum(g[:, 0].var(ddof=1) / len(g) for g in groups)) / 2),
                "by_group": {
                    str(n): [float(g[:, 0].mean()), float(g[:, 1].mean())]
                    for n, g in zip(np.unique(labels), groups, strict=True)
                },
            }
            r = out[f"{name} | {label}"]
            print(
                f"{name:20s} {label:52s} SSIM {r['ssim']:.4f} ± {r['ssim_se']:.4f} "
                f"P {r['pearson']:.4f}  ({time.time() - t0:.0f} с)",
                flush=True,
            )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
