"""Обучение любой из моделей решения по её конфигу (configs/<прогон>.yaml).

    cd training
    pip install -r requirements.txt
    python train.py configs/R08-gd8.yaml [--steps 20] [--workers 0]

Обучение без учителя: генератор целей на лету → сеть → фаза ДОЭ → дифференцируемый Френель →
loss = Σ по плоскостям (1 − Пирсон(|K|², T)); эталонные фазовые маски не нужны. Результат —
runs/<имя>/{best.pt, last.pt, log.jsonl}; best.pt того же формата, что weights/*.pt решения.

Код в doetrain/ и experiments/ — из репозитория экспериментов команды (изменены только имя
пакета и команды запуска в документации): им обучены поставляемые веса. Инференс в приложении
(doesynth) сверен с ним бит в бит.
Конфиги с photos_dir используют фото COCO val2017 (5000 JPEG), по умолчанию из
data/coco2017/val2017 внутри training/: положите их туда, укажите свой путь или удалите
photos_dir и строку photo в data.weights — тогда генератор берёт остальные виды картинок.
"""

import argparse
import runpy
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

R08_KEYS = {"loss", "noise", "noise_target", "val_objects", "val_supersample", "init_from"}
# Модели других участников команды регистрируются в doetrain.core.net.MODELS при импорте модуля
REGISTER = {"phasenet3d": HERE / "experiments" / "R12", "yresnet": HERE / "experiments" / "R13"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", type=int, help="число шагов (для короткой проверки)")
    parser.add_argument("--workers", type=int, help="число процессов генератора данных")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if R08_KEYS & set(raw) or raw.get("model") == "cghnet":
        script = HERE / "experiments" / "R08" / "train_r08.py"
        argv = [str(args.config)]
        argv += ["--steps", str(args.steps)] if args.steps else []
        argv += ["--workers", str(args.workers)] if args.workers is not None else []
    else:
        from doetrain.train.train import load_config, train

        if raw.get("model") in REGISTER:
            sys.path.insert(0, str(REGISTER[raw["model"]]))
            __import__(raw["model"])

        cfg = load_config(args.config, steps=args.steps, workers=args.workers)
        if args.steps:
            cfg.val_every = min(cfg.val_every, args.steps)
            cfg.log_every = min(cfg.log_every, args.steps)
        print(train(cfg))
        return
    sys.argv = [str(script), *argv]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
