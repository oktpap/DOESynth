"""R13: обучение Y-shaped ResNet (yresnet.py) циклом проекта doetrain.train.train.train.

Цикл, данные, loss (Σ по плоскостям 1 − Пирсон по |K|²), валидация по сдаваемому пути и формат
весов — те же, что у U-Net R03 и PhaseNet3D R12. Число шагов подбирается по замеру скорости шага
(experiments/R12-phasenet3d/budget.py), чтобы обучение уложилось в 15 минут, как в R12.

Запуск — через training/train.py:
  python train.py configs/R13-yres.yaml [--steps N]
"""

import argparse
import sys
from pathlib import Path

from doetrain.train.train import load_config, train

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yresnet  # noqa: F401 — регистрирует «yresnet» в core.net.MODELS


def main() -> None:
    ap = argparse.ArgumentParser(description="R13: обучение Y-shaped ResNet в постановке task.pdf")
    ap.add_argument("config")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--name")
    ap.add_argument("--val-every", type=int)
    ap.add_argument("--out", help="каталог прогонов вместо out из конфига (проверка на CPU)")
    ap.add_argument("--photos-dir", help="каталог фото вместо photos_dir (проверка на CPU)")
    a = ap.parse_args()
    cfg = load_config(a.config, steps=a.steps, name=a.name, val_every=a.val_every, out=a.out)
    if a.photos_dir:
        cfg.data = {**cfg.data, "photos_dir": a.photos_dir}
    print(train(cfg))


if __name__ == "__main__":
    main()
