"""Выгрузка наборов для бенчмарка из репозитория экспериментов команды в sets.npz.

Скрипту нужен experiments/R07-volume/evaluate.py репозитория экспериментов, в решение он не
входит; готовые наборы лежат в sets.npz. Запуск:
    python make_sets.py --repo <репозиторий экспериментов> --out sets.npz

Наборы — те же, что в отчётах R02–R13:
- validation — 4 четвёрки картинок scikit-image (фото, документы, буквы, смешанные);
- ood — 24 четвёрки категорий, которых генератор обучения не рисует (другие алфавиты,
  пиктограммы, графики, формулы, фракталы, оптические моды);
- volume_test — 200 четвёрок «сечения одного 3D-объекта» (отложенное разбиение).
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("sets.npz"))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "r07", args.repo / "experiments" / "R07-volume" / "evaluate.py"
    )
    r07 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r07)
    sets = r07.build_sets()
    keys = [k for k in sets if k[0] in ("validation", "ood", "volume_test")]
    images = np.stack([sets[k] for k in keys]).astype(np.uint8)
    np.savez_compressed(
        args.out,
        images=images,
        set=np.array([k[0] for k in keys]),
        category=np.array([k[1] for k in keys]),
        index=np.array([k[2] for k in keys]),
    )
    print(f"{len(keys)} четвёрок → {args.out}")


if __name__ == "__main__":
    main()
