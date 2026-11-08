"""R08: обучение 3D-CGH-Net и вариантов — цикл doetrain.train.train.train с опциями статьи.

Отличия от train.train (остальное — тот же код: данные на лету, AdamW + OneCycle, сдаваемый путь
в валидации, формат весов core.net):
- loss: «planes» — как во всём проекте, Σ по плоскостям (1 − Пирсон); «volume» — loss (2) статьи,
  1 − Пирсон по всему объёму сразу (×P, чтобы масштаб совпадал с «planes»);
- noise: σ гауссова шума, который статья (разд. 3.1) добавляет к изображениям обучения (5 %);
  noise_target — шум и в цели loss (как в статье: шум в самих изображениях набора) или только
  во входе сети;
- init_from — дообучение с весов другого прогона той же архитектуры; параметры, которых там нет
  (обучаемые шаги gd_iters), остаются в начальном значении;
- валидация — ещё и на первых val_objects объектах volume-validation (сечения 3D-объектов,
  supersample как в данных); best.pt выбирается по ним (val/vcc), а не по 2D validation:
  обучение идёт на сечениях, а 4 четвёрки core.data — другой вид целей.

Запуск — через training/train.py, например:
  python train.py configs/R08-paper.yaml
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cghnet  # регистрирует «cghnet» в core.net.MODELS

from doetrain.core import net
from doetrain.core.data import VALIDATION, validation_set
from doetrain.core.device import get_device
from doetrain.core.methods import plane_losses
from doetrain.core.optics import chirps, propagate
from doetrain.train.data import SampleGenerator, TrainStream
from doetrain.train.train import Config, model_kwargs, validate
from doetrain.train.volume import dataset_item


@dataclass
class R08Config(Config):
    loss: str = "planes"  # planes | volume
    noise: float = 0.0
    noise_target: bool = True
    val_objects: int = 16
    val_supersample: int = 2
    init_from: str | None = (
        None  # дообучение: веса best.pt другого прогона (новые параметры — с нуля)
    )


def load(path: str | Path) -> R08Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return R08Config(**raw)


def loss_terms(model, images, targets, chirp, cfg: R08Config) -> torch.Tensor:
    """Средний по батчу loss: images — вход сети, targets — цели loss (могут быть без шума)."""
    phase = model(images)
    amp = propagate(torch.polar(torch.ones_like(phase), phase), chirp).abs()
    if cfg.loss == "planes":
        return plane_losses(amp, targets, "pearson", cfg.image_power).sum(-1).mean()
    if cfg.loss == "volume":
        p = targets.shape[1]
        return (p * cghnet.volume_pearson_loss(amp, targets, cfg.image_power)).mean()
    raise ValueError(f"loss — planes или volume, получено {cfg.loss}")


def volume_val_sets(cfg: R08Config) -> dict[str, np.ndarray]:
    return {
        f"vol{i:02d}": dataset_item("validation", i, supersample=cfg.val_supersample)[0]
        for i in range(cfg.val_objects)
    }


def train(cfg: R08Config, device: torch.device | None = None) -> Path:
    if cfg.crosstalk or cfg.crosstalk_end is not None:  # поля Config, которые этот цикл не знает
        raise ValueError(
            "train_r08 не поддерживает crosstalk/crosstalk_end — используйте train.train"
        )
    device = device or get_device()
    torch.manual_seed(cfg.seed)
    run_dir = Path(cfg.out) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), allow_unicode=True), "utf-8")

    kwargs = model_kwargs(cfg)
    model = net.build_model(cfg.model, **kwargs).to(device)
    if cfg.init_from:
        state = torch.load(cfg.init_from, map_location=device, weights_only=True)["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or set(missing) - {"gd_log_step"}:
            raise ValueError(f"init_from: лишние {unexpected}, недостающие {missing}")
        print(f"старт с весов {cfg.init_from}; новые параметры: {missing}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps)
    chirp = chirps(device=device)
    loader = torch.utils.data.DataLoader(
        TrainStream(SampleGenerator(**cfg.data), cfg.seed),
        batch_size=cfg.batch,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.workers > 0,
    )
    val_sets = {key: validation_set(key) for key in VALIDATION}
    vol_sets = volume_val_sets(cfg)
    noise_gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(run_dir)
    except ImportError:
        writer = None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.name}: {cfg.model} {kwargs}, {n_params / 1e6:.2f} M параметров, {device}")
    meta = {"config": asdict(cfg), "params": n_params}
    best, t0 = -np.inf, time.perf_counter()
    with (run_dir / "log.jsonl").open("a", encoding="utf-8") as log:
        for step, clean in enumerate(loader, start=1):
            clean = clean.to(device, non_blocking=True)
            images = clean
            if cfg.noise:
                noise = torch.randn(clean.shape, generator=noise_gen, device=device)
                images = (clean + cfg.noise * noise).clamp(0, 1)
            targets = images if cfg.noise_target else clean
            loss = loss_terms(model, images, targets, chirp, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            record = {}
            if step % cfg.log_every == 0 or step == cfg.steps:
                record |= {"loss": loss.item(), "lr": sched.get_last_lr()[0]}
            if step % cfg.val_every == 0 or step == cfg.steps:
                val = validate(model, val_sets, device, cfg.image_power)
                vol = validate(model, vol_sets, device, cfg.image_power)
                record |= {f"val/{k}": v for k, v in val.items()}
                record |= {"val/vcc": vol["cc"], "val/vssim": vol["ssim"]}
                save = (model, cfg.model, kwargs)
                power = {"image_power": cfg.image_power}
                if record["val/vcc"] > best:
                    best = record["val/vcc"]
                    best_meta = meta | {"step": step, "val_vcc": best, "val_cc": record["val/cc"]}
                    net.save_weights(*save, run_dir / "best.pt", best_meta, **power)
                net.save_weights(*save, run_dir / "last.pt", meta | {"step": step}, **power)
            if record:
                record = {"step": step, "seconds": round(time.perf_counter() - t0, 1)} | record
                log.write(json.dumps(record) + "\n")
                log.flush()
                shown = {
                    k: v
                    for k, v in record.items()
                    if "/" not in k or k in ("val/cc", "val/vcc", "val/vssim")
                }
                print(" ".join(f"{k}={v:.4g}" for k, v in shown.items()), flush=True)
                if writer is not None:
                    for k, v in record.items():
                        if k not in ("step", "seconds"):
                            writer.add_scalar(k, v, step)
            if step >= cfg.steps:
                break
    if writer is not None:
        writer.close()
    return run_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", type=int, help="переопределить число шагов (для проверки)")
    parser.add_argument("--out", help="переопределить каталог прогонов")
    parser.add_argument("--workers", type=int, help="переопределить число воркеров")
    args = parser.parse_args()
    cfg = load(args.config)
    if args.steps:
        cfg.steps = args.steps
        cfg.val_every = min(cfg.val_every, args.steps)
        cfg.log_every = min(cfg.log_every, args.steps)
    if args.out:
        cfg.out = args.out
    if args.workers is not None:
        cfg.workers = args.workers
    print(train(cfg))


if __name__ == "__main__":
    main()
