"""Нейросети «4 цели → фаза ДОЭ» и загрузка весов для инференса.

Сеть получает 4 целевых изображения T ∈ [0, 1] как 4 канала и выдаёт фазу ДОЭ [N, N].
Фаза везде кодируется двумя каналами (a, b) → atan2(b, a), чтобы выход не упирался
в разрыв фазы на ±π. Обучение — без учителя через прямую модель (doetrain.train).
Архитектуры (MODELS): «unet» — прямой U-Net «цели → фаза»; «superpos» — обученная
суперпозиция: сеть ставит целям фазы в их плоскостях, перенос на ДОЭ — точный оператор
Френеля (см. docs/decisions.md, «вторая архитектура»); «unrolled» — суперпозиция и затем
развёрнутые итерации многоплоскостного GS с обучаемой проекцией (физика переноса в каждой
итерации, сеть видит восстановления всех плоскостей и правит их).

Веса для демо лежат в пакете (`WEIGHTS`), чтобы окно и CLI работали сразу после клонирования.
Файл весов — словарь {"model": имя, "kwargs": параметры конструктора, "state_dict": …,
"image_power": конвенция изображения, под которую сеть обучена (1 — |K|, 2 — |K|²), "meta": …}.
"""

import warnings
from functools import cache
from pathlib import Path

import torch
from torch import nn

from doetrain.core.methods import unit_energy
from doetrain.core.optics import (
    IMAGE_POWER,
    POWER_LABEL,
    TASK,
    backpropagate,
    check_power,
    chirps,
    propagate,
    target_amplitude,
)

WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "net.pt"


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
    )


class UNet(nn.Module):
    """U-Net с depth понижениями разрешения; на 128×128 и depth = 4 дно 8×8 видит всё поле."""

    def __init__(self, in_ch: int = 4, base: int = 32, depth: int = 4):
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(in_ch, chans[0])])
        self.enc.extend(_block(chans[i], chans[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth))
        )
        self.dec = nn.ModuleList(_block(2 * chans[i], chans[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chans[0], 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 4, N, N] → фаза [B, N, N]."""
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else nn.functional.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec, strict=True):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        ab = self.head(x)
        return torch.atan2(ab[:, 1], ab[:, 0])


class _Trunk(nn.Module):
    """U-Net с заданным числом выходных каналов; апсемплинг билинейный + свёртка 1×1
    вместо ConvTranspose2d — у транспонированной свёртки шахматные артефакты, которые
    в фазе ДОЭ превращаются в рассеянный свет (наблюдение участника команды в proto/Holography)."""

    def __init__(self, in_ch: int, out_ch: int, base: int, depth: int):
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(in_ch, chans[0])])
        self.enc.extend(_block(chans[i], chans[i + 1]) for i in range(depth))
        self.reduce = nn.ModuleList(
            nn.Conv2d(chans[i + 1], chans[i], 1) for i in reversed(range(depth))
        )
        self.dec = nn.ModuleList(_block(2 * chans[i], chans[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chans[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else nn.functional.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for reduce, dec in zip(self.reduce, self.dec, strict=True):
            up = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
            x = dec(torch.cat([reduce(up), skips.pop()], dim=1))
        return self.head(x)


class SuperposNet(nn.Module):
    """Обученная суперпозиция (по мотивам DeepCGH и HoloNet): физика внутри модели.

    Ствол по 4 целям выдаёт на каждую плоскость p фазу ψ_p = atan2(b_p, a_p) и усиление
    m_p = 2·sigmoid(c_p); поля A_p·m_p·exp(jψ_p) (A — целевая амплитуда, √T или T по конвенции)
    переносятся на плоскость ДОЭ точным обратным оператором Френеля и складываются — глобальный
    перенос сеть не учит, он задан физикой. Малый кодировщик с нулевой головой локально правит
    комплексную сумму до чисто фазового ДОЭ; при нулевой правке фаза — ровно angle(суммы),
    то есть methods.superposition_init с выученными вместо случайных фазами плоскостей:
    обучение стартует из окрестности классического решения.

    Конвенция изображения — часть модели (от неё зависит A), поэтому задаётся конструктором
    и попадает в kwargs весов; train прокидывает её из конфига (image_power).
    """

    def __init__(
        self,
        base: int = 32,
        depth: int = 4,
        refine_base: int = 16,
        refine_depth: int = 2,
        image_power: int = IMAGE_POWER,
    ):
        super().__init__()
        self.image_power = check_power(image_power)
        planes = len(TASK.distances)
        self.trunk = _Trunk(planes, 3 * planes, base, depth)
        self.refiner = _Trunk(3, 2, refine_base, refine_depth) if refine_base > 0 else None
        if self.refiner is not None:
            nn.init.zeros_(self.refiner.head.weight)
            nn.init.zeros_(self.refiner.head.bias)
        self.register_buffer("chirp", chirps(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 4, N, N] → фаза [B, N, N]."""
        a, b, c = self.trunk(x).split(x.shape[1], dim=1)
        amp = target_amplitude(x, self.image_power) * 2 * torch.sigmoid(c)
        total = backpropagate(torch.polar(amp, torch.atan2(b, a)), self.chirp).sum(dim=-3)
        if self.refiner is None:
            return torch.atan2(total.imag, total.real)
        unit = total / total.abs().clamp_min(1e-12)
        m = total.abs() / total.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        delta = self.refiner(torch.stack([unit.real, unit.imag, m - 1], dim=1))
        return torch.atan2(unit.imag + delta[:, 1], unit.real + delta[:, 0])


class _Update(nn.Module):
    """Обучаемая проекция одной итерации: по целям T и восстановлениям всех плоскостей
    (2·P каналов) выдаёт на плоскость поправку амплитуды и комплексный множитель (3·P каналов).
    Два масштаба (N и N/2): дефокусировка соседних плоскостей — 3–9 пикселей, одной свёртки
    3×3 мало. Голова нулевая: до обучения итерация — ровно проекция GS."""

    def __init__(self, planes: int, base: int):
        super().__init__()
        self.fine = _block(2 * planes, base)
        self.coarse = _block(base, 2 * base)
        self.reduce = nn.Conv2d(2 * base, base, 1)
        self.merge = _block(2 * base, base)
        self.head = nn.Conv2d(base, 3 * planes, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.fine(x)
        h = self.coarse(nn.functional.max_pool2d(f, 2))
        up = nn.functional.interpolate(h, scale_factor=2, mode="bilinear")
        return self.head(self.merge(torch.cat([f, self.reduce(up)], dim=1)))


class UnrolledNet(nn.Module):
    """Развёрнутый многоплоскостной GS с обучаемой проекцией (unrolled optimization).

    Старт — обученная суперпозиция (SuperposNet без кодировщика): φ₀ = arg Σ_p B⁻¹[A_p m_p e^{jψ_p}].
    Затем iters итераций, у каждой свои веса: поле ДОЭ e^{jφ} переносится в плоскости (K_p);
    малая двухмасштабная свёрточная сеть по целям T и восстановлениям |K_p|^power всех плоскостей
    выдаёт на плоскость поправку амплитуды δa_p и комплексный множитель 1 + u_p + j·v_p; поле
    плоскости G_p = (A_p + δa_p) · K_p/|K_p| · (1 + u_p + j·v_p) переносится обратно точным
    оператором, поля складываются, φ = arg суммы. При нулевых головах G_p = A_p·e^{j·arg K_p} —
    ровно проекция global GS (methods.gs_global): до обучения модель = суперпозиция + iters
    итераций GS, а обучение улучшает проекцию, физику переноса сеть не учит. Итерация видит
    восстановления всех плоскостей сразу, поэтому может напрямую гасить перекрёстные помехи —
    того, что не давалось прямому U-Net (R02, R03, R05: разделение 0,08 против 0,29 у Adam).
    Поправка амплитуды аддитивна и без ограничения знака: сеть вправе добавлять свет в тёмные
    области (Пирсон к фону нечувствителен — свободный фон, как в MRAF), а relu там глушил бы
    градиент — у нулевой цели A_p = 0 (проверка optics-reviewer); отрицательная амплитуда —
    сдвиг фазы на π. A_p — амплитуда цели по конвенции (√T или T) с нормировкой на энергию
    поля ДОЭ, как в GS; конвенция — часть модели (image_power).
    """

    def __init__(
        self,
        iters: int = 6,
        base: int = 16,
        init_base: int = 16,
        init_depth: int = 4,
        image_power: int = IMAGE_POWER,
    ):
        super().__init__()
        if iters < 1:
            raise ValueError("iters ≥ 1: без итераций это SuperposNet")
        self.image_power = check_power(image_power)
        planes = len(TASK.distances)
        self.start = SuperposNet(
            base=init_base, depth=init_depth, refine_base=0, image_power=image_power
        )
        self.updates = nn.ModuleList(_Update(planes, base) for _ in range(iters))
        self.register_buffer("chirp", chirps(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 4, N, N] → фаза [B, N, N]."""
        planes = x.shape[1]
        amp = unit_energy(target_amplitude(x, self.image_power))
        phase = self.start(x)
        for update in self.updates:
            k = propagate(torch.polar(torch.ones_like(phase), phase), self.chirp)
            intensity = k.real.square() + k.imag.square()  # |K_p|²
            image = intensity if self.image_power == 2 else (intensity + 1e-12).sqrt()
            image = image / image.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            da, u, v = update(torch.cat([x, image], dim=1)).split(planes, dim=1)
            unit = k * torch.rsqrt(intensity + 1e-12)  # K_p/|K_p|; 1e-12 — от деления на нуль
            g = unit * torch.complex(1 + u, v) * (amp + da)
            total = backpropagate(g, self.chirp).sum(dim=-3)
            phase = torch.atan2(total.imag, total.real)
        return phase


MODELS = {"unet": UNet, "superpos": SuperposNet, "unrolled": UnrolledNet}


def build_model(name: str = "unet", **kwargs) -> nn.Module:
    return MODELS[name](**kwargs)


def save_weights(
    model: nn.Module,
    name: str,
    kwargs: dict,
    path: Path,
    meta: dict | None = None,
    *,
    image_power: int,
):
    """image_power обязателен: метка конвенции должна совпадать с тем, под какой loss сеть
    реально обучали, а не браться из глобального переключателя по умолчанию."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt = {"model": name, "kwargs": kwargs, "state_dict": state, "meta": meta or {}}
    torch.save(ckpt | {"image_power": check_power(image_power)}, path)


def weights_available(path: Path | None = None) -> bool:
    return Path(path or WEIGHTS).is_file()


def available_for(power: int = IMAGE_POWER, path: Path | None = None) -> bool:
    """Есть рабочие веса, обученные под эту конвенцию изображения (|K| или |K|²).

    Битый или несовместимый файл весов (указатель LFS, неизвестная модель, чужой формат)
    не роняет CLI и окно: метод «сеть» просто недоступен, остальные работают.
    """
    path = Path(path or WEIGHTS)
    if not path.is_file():
        return False
    try:
        trained = load_model(path, "cpu").image_power
    except Exception as e:  # noqa: BLE001 — любой сбой чтения весов = сеть недоступна
        warnings.warn(f"веса сети {path} не читаются ({type(e).__name__}: {e})", stacklevel=2)
        return False
    return trained == power


@cache
def load_model(path: Path, device: str = "cpu") -> nn.Module:
    """Кэшируется по (путь, устройство); после переобучения в том же процессе — cache_clear()."""
    if not path.is_file():
        raise FileNotFoundError(
            f"нет весов сети {path}: обучите её (`python train.py`) и экспортируйте веса"
        )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["model"], **ckpt["kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    trained = check_power(ckpt.get("image_power", 2))  # до переключателя обучали лишь под |K|²
    inner = ckpt["kwargs"].get("image_power")  # конвенция физики внутри модели (superpos)
    if inner is not None and inner != trained:
        raise ValueError(
            f"веса {path}: метка image_power = {trained} расходится с конвенцией модели "
            f"в kwargs ({inner}) — файл повреждён или собран вручную"
        )
    model.image_power = trained
    return model.to(device).eval()


def prepare(device: torch.device, path: Path | None = None) -> None:
    """Читает веса заранее, чтобы чтение файла не попадало в замер времени синтеза."""
    load_model(Path(path or WEIGHTS), str(device))


@torch.no_grad()
def predict(
    images: torch.Tensor, path: Path | None = None, power: int = IMAGE_POWER
) -> torch.Tensor:
    """images — T в [0, 1], [4, N, N] → фаза [N, N] на том же устройстве.

    Сеть, обученная под другую конвенцию изображения, вызывает ошибку, а не тихо ухудшает
    результат: переключение |K| ↔ |K|² требует переобучения.
    """
    model = load_model(Path(path or WEIGHTS), str(images.device))
    if model.image_power != check_power(power):
        raise ValueError(
            f"веса сети обучены под {POWER_LABEL[model.image_power]}, а выбрано "
            f"{POWER_LABEL[power]}: переобучите сеть (image_power в конфиге)"
        )
    return model(images[None].float())[0]
