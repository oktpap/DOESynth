"""R12: обучение PhaseNet3D (phasenet3d.py) циклом проекта doetrain.train.train.train.

Цикл, данные, loss (Σ по плоскостям 1 − Пирсон по |K|²), валидация по сдаваемому пути и формат
весов — те же, что у U-Net R03. Отличаются модель и ограничение по времени. Число шагов
подбирается по замеру скорости шага (budget.py), чтобы обучение уложилось в 15 минут: так решил
владелец. OneCycle требует числа шагов заранее.

Запуск — через training/train.py:
  python train.py configs/R12-p3d.yaml [--steps N]
"""

import argparse
import sys
from pathlib import Path

from doetrain.train.train import load_config, train

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phasenet3d  # noqa: F401 — регистрирует «phasenet3d» в core.net.MODELS


def main() -> None:
    ap = argparse.ArgumentParser(description="R12: обучение PhaseNet3D в постановке task.pdf")
    ap.add_argument("config")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--name")
    ap.add_argument("--val-every", type=int)
    a = ap.parse_args()
    print(train(load_config(a.config, steps=a.steps, name=a.name, val_every=a.val_every)))


if __name__ == "__main__":
    main()
