"""Выборка для настройки дооптимизации (не пересекается с наборами отчёта).

volume_validation (200 четвёрок сечений 3D-объектов, разбиение validation) и 64 независимые
четвёрки 2D-генератора обучения (с фото COCO, как в обучении) с сидом 777.
Запуск из корня решения (нужен пакет doetrain из training/):
    PYTHONPATH=training python benchmark/make_tune.py --out benchmark/tune.npz \\
        --photos <каталог COCO val2017>
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tune.npz"))
    parser.add_argument("--quads", type=int, default=64)
    parser.add_argument("--photos", default=None, help="каталог фото COCO val2017")
    args = parser.parse_args()
    from doetrain.train.data import SampleGenerator
    from doetrain.train.volume import dataset_item

    vol = [dataset_item("validation", i, supersample=2)[0] for i in range(200)]
    gen = SampleGenerator(photos_dir=args.photos)  # независимые четвёрки, как в обучении
    rng = np.random.default_rng(777)
    flat = [gen.sample(rng) for _ in range(args.quads)]
    images = np.stack(vol + flat).astype(np.uint8)
    labels = ["volume_validation"] * len(vol) + ["generator"] * len(flat)
    np.savez_compressed(args.out, images=images, set=np.array(labels))
    print(images.shape, "→", args.out)


if __name__ == "__main__":
    main()
