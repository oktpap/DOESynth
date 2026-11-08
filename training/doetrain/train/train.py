"""Обучение сети без учителя через прямую модель: loss = Σ по плоскостям (1 − Пирсон(|K|^p, T)).

p = image_power из конфига (по умолчанию optics.IMAGE_POWER: 2 — |K|², 1 — |K|); он же
записывается в веса. Тот же loss, что у Adam-бейзлайна (doetrain.core.methods.loss_fn),
поэтому сеть и Adam оптимизируют одно и то же. Опция crosstalk = μ > 0 добавляет к каждой
плоскости μ · штраф перекрёстных помех сверх законного сходства целей (crosstalk_excess);
по умолчанию 0 — loss как у Adam. Качество на validation считаем сдаваемым путём
(квантованный ДОЭ → 8 бит, doetrain.core.pipeline.evaluate) — по нему выбираем best.pt;
в лог пишется КК в обеих конвенциях.

Запуск: python train.py configs/<конфиг>.yaml из папки training/
(DOETRAIN_REQUIRE_GPU=1 — остановиться, если GPU нет).
Результат: runs/<name>/{config.yaml, log.jsonl, last.pt, best.pt}, TensorBoard — в том же каталоге.
"""

import inspect
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from doetrain.core import net
from doetrain.core.data import VALIDATION, validation_set
from doetrain.core.device import get_device
from doetrain.core.methods import plane_losses
from doetrain.core.optics import IMAGE_POWER, POWER_KEY, check_power, chirps, propagate
from doetrain.core.pipeline import evaluate
from doetrain.train.data import SampleGenerator, TrainStream


@dataclass
class Config:
    name: str = "unet"
    model: str = "unet"
    model_kwargs: dict = field(default_factory=lambda: {"base": 32, "depth": 4})
    data: dict = field(default_factory=dict)  # аргументы SampleGenerator: weights, photos_dir
    image_power: int = IMAGE_POWER  # изображение = |K|^image_power: 2 — |K|², 1 — |K|
    crosstalk: float = 0.0  # μ — вес штрафа перекрёстных помех (crosstalk_excess); 0 — без него
    crosstalk_end: float | None = None  # μ к концу обучения (линейно); None — μ постоянен
    grad_clip: float | None = None  # порог глобальной нормы градиента (unrolled); None — без него
    steps: int = 20_000
    batch: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    workers: int = 4
    seed: int = 0
    log_every: int = 50
    val_every: int = 500
    out: str = "runs"


def load_config(path: str | Path, **overrides) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**raw)


def model_kwargs(cfg: Config) -> dict:
    """model_kwargs конфига; моделям с физикой внутри (superpos) добавляется конвенция.

    Конвенция изображения в модели обязана совпадать с конвенцией loss и валидации,
    поэтому источник один — cfg.image_power; противоречие в model_kwargs — ошибка конфига."""
    kwargs = dict(cfg.model_kwargs)
    if (
        "image_power" in inspect.signature(net.MODELS[cfg.model]).parameters
        and kwargs.setdefault("image_power", cfg.image_power) != cfg.image_power
    ):
        raise ValueError(
            f"model_kwargs.image_power = {kwargs['image_power']} противоречит "
            f"image_power = {cfg.image_power}: конвенцию задаёт одно поле image_power"
        )
    return kwargs


def batch_plane_losses(
    model, images: torch.Tensor, chirp: torch.Tensor, power: int = IMAGE_POWER
) -> torch.Tensor:
    """images [B, 4, N, N] → loss каждой плоскости [B, 4]: 1 − Пирсон(|K|^power, T)."""
    phase = model(images)
    amp = propagate(torch.polar(torch.ones_like(phase), phase), chirp).abs()
    return plane_losses(amp, images, "pearson", check_power(power))


def _standardized(x: torch.Tensor) -> torch.Tensor:
    """[..., P, N, N] → [..., P, N·N]: нулевое среднее и единичная норма по полю."""
    x = x.flatten(-2)
    x = x - x.mean(-1, keepdim=True)
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def crosstalk_excess(img: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Перекрёстные помехи сверх законного сходства целей → [..., P].

    img — изображение восстановления (|K|^power), targets — цели T, обе [..., P, N, N].
    Для плоскости i и чужой цели j: x_ij = ρ(I_i, T_j) − ρ(I_i, T_i)·ρ(T_i, T_j), ρ — Пирсон
    по полю. Это корреляция с T_j той части восстановления, которую своя цель не объясняет:
    x_ij = ⟨z(I_i) − ρ_ii·z(T_i), z(T_j)⟩, z — стандартизация по полю. У идеального
    восстановления (I_i = T_i) x_ij = 0 при любом сходстве целей (похожие буквы не
    штрафуются), у смеси целей x_ij > 0. Градиент — через обе части, без detach: тогда спуск
    по штрафу не уменьшает собственный КК (в первом порядке ⟨−∇x_ij, ∇ρ_ii⟩ ∝ ρ_ii·x_ij ≥ 0);
    с detach штраф отталкивал бы плоскость и от общей со своей целью части (проверка
    optics-reviewer). Штраф плоскости — среднее max(0, x_ij) по j ≠ i.
    """
    zi, zt = _standardized(img), _standardized(targets)
    r = zi @ zt.transpose(-1, -2)  # r[..., i, j] = ρ(I_i, T_j)
    c = zt @ zt.transpose(-1, -2)  # c[..., i, j] = ρ(T_i, T_j)
    own = r.diagonal(dim1=-2, dim2=-1)
    excess = (r - own[..., None] * c).clamp(min=0)
    p = r.shape[-1]
    off = 1 - torch.eye(p, dtype=excess.dtype, device=excess.device)  # без булевой маски:
    return (excess * off).sum(-1) / (p - 1)  # на GPU обходится без синхронизации с CPU


def batch_terms(
    model, images: torch.Tensor, chirp: torch.Tensor, power: int = IMAGE_POWER
) -> tuple[torch.Tensor, torch.Tensor]:
    """Слагаемые loss по плоскостям [B, 4]: собственное 1 − Пирсон и помехи crosstalk_excess."""
    phase = model(images)
    amp = propagate(torch.polar(torch.ones_like(phase), phase), chirp).abs()
    img = amp if check_power(power) == 1 else amp.square()
    return plane_losses(amp, images, "pearson", power), crosstalk_excess(img, images)


def batch_loss(
    model,
    images: torch.Tensor,
    chirp: torch.Tensor,
    power: int = IMAGE_POWER,
    crosstalk: float = 0.0,
) -> torch.Tensor:
    """Средний по батчу loss: сумма по 4 плоскостям (1 − Пирсон), как у Adam (methods.loss_fn);
    при crosstalk = μ > 0 к каждой плоскости добавляется μ · crosstalk_excess."""
    if crosstalk < 0:
        raise ValueError(f"crosstalk = {crosstalk} < 0 поощрял бы помехи")
    if not crosstalk:
        return batch_plane_losses(model, images, chirp, power).sum(-1).mean()
    own, xtalk = batch_terms(model, images, chirp, power)
    return (own + crosstalk * xtalk).sum(-1).mean()


def crosstalk_weight(cfg: "Config", step: int) -> float:
    """μ на шаге step (с 1): постоянный cfg.crosstalk или линейно до cfg.crosstalk_end к концу
    обучения — штраф как подсказка в начале, а финал оптимизирует чистый Пирсон."""
    if cfg.crosstalk_end is None:
        return cfg.crosstalk
    t = (step - 1) / max(cfg.steps - 1, 1)
    return cfg.crosstalk + (cfg.crosstalk_end - cfg.crosstalk) * min(max(t, 0.0), 1.0)


@torch.no_grad()
def validate(
    model: torch.nn.Module, sets: dict[str, np.ndarray], device, power: int = IMAGE_POWER
) -> dict:
    """Средний КК и ИСС по сдаваемому пути на каждом наборе validation (в конвенции power)
    и средний КК в каждой конвенции (cc_K, cc_K2) — для сравнения, не для выбора best."""
    model.eval()
    out, by_power = {}, {key: [] for key in POWER_KEY.values()}
    for key, targets_u8 in sets.items():
        images = torch.from_numpy(targets_u8.astype(np.float32) / 255).to(device)
        phase = model(images[None])[0].double().cpu().numpy()
        _, _, score = evaluate(phase, targets_u8, power=power)
        out[f"cc/{key}"] = score["cc_mean"]
        out[f"ssim/{key}"] = score["ssim_mean"]
        for pk, s in score["by_power"].items():
            by_power[pk].append(s["cc_mean"])
    out["cc"] = float(np.mean([out[f"cc/{k}"] for k in sets]))
    out["ssim"] = float(np.mean([out[f"ssim/{k}"] for k in sets]))
    out |= {f"cc_{pk}": float(np.mean(v)) for pk, v in by_power.items()}
    model.train()
    return out


def train(cfg: Config, device: torch.device | None = None) -> Path:
    """Обучает модель и возвращает путь к best.pt (формат doetrain.core.net)."""
    if cfg.crosstalk < 0 or (cfg.crosstalk_end is not None and cfg.crosstalk_end < 0):
        raise ValueError("crosstalk и crosstalk_end — веса штрафа, они не бывают отрицательными")
    device = device or get_device()
    torch.manual_seed(cfg.seed)
    run_dir = Path(cfg.out) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), allow_unicode=True), "utf-8")

    kwargs = model_kwargs(cfg)
    model = net.build_model(cfg.model, **kwargs).to(device)
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
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(run_dir)
    except ImportError:
        writer = None

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"{cfg.name}: {cfg.model} {cfg.model_kwargs}, {n_params / 1e6:.2f} M параметров, {device}"
    )
    meta = {"config": asdict(cfg), "params": n_params}
    best_cc, t0 = -np.inf, time.perf_counter()
    with (run_dir / "log.jsonl").open("a", encoding="utf-8") as log:
        for step, images in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            mu = crosstalk_weight(cfg, step)
            terms = None
            if mu:
                terms = batch_terms(model, images, chirp, cfg.image_power)
                loss = (terms[0] + mu * terms[1]).sum(-1).mean()
            else:
                loss = batch_loss(model, images, chirp, cfg.image_power)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            record = {}
            if step % cfg.log_every == 0 or step == cfg.steps:
                record |= {"loss": loss.item(), "lr": sched.get_last_lr()[0]}
                if terms is not None:  # loss с штрафом с R02 несравним — пишем слагаемые отдельно
                    own, xtalk = terms[0].detach(), terms[1].detach()
                    record |= {
                        "loss_own": own.sum(-1).mean().item(),
                        "crosstalk": xtalk.mean().item(),
                        "mu": mu,
                    }
            if step % cfg.val_every == 0 or step == cfg.steps:
                val = validate(model, val_sets, device, cfg.image_power)
                record |= {f"val/{k}": v for k, v in val.items()}
                save = (model, cfg.model, kwargs)
                power = {"image_power": cfg.image_power}
                if record["val/cc"] > best_cc:
                    best_cc = record["val/cc"]
                    best_meta = meta | {"step": step, "val_cc": best_cc}
                    net.save_weights(*save, run_dir / "best.pt", best_meta, **power)
                net.save_weights(*save, run_dir / "last.pt", meta | {"step": step}, **power)
            if record:
                record = {"step": step, "seconds": round(time.perf_counter() - t0, 1)} | record
                log.write(json.dumps(record) + "\n")
                log.flush()
                shown = {k: v for k, v in record.items() if "/" not in k or k.startswith("val/cc")}
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
