"""Бенчмарк всех методов решения: выбор модели по умолчанию и параметров дооптимизации.

    python benchmark/benchmark.py --sets benchmark/sets.npz --out benchmark/results.json

Все числа — по сдаваемому пути: фаза → 8-битный ДОЭ → фаза из кода → |K|² → 8 бит
(максимум → 255) → SSIM условия и Пирсон против исходных 8-битных изображений.

Правило выбора записано до запуска:
- S — среднее по трём наборам (validation, ood, volume_test) среднего по четвёркам SSIM условия;
  P — то же для Пирсона.
- Дооптимизация: Adam по фазе со старта из фазы сети. Критерий (loss) и lr — общие для всех
  сетей (в интерфейсе одна галочка): максимум среднего по сетям S при k = 100 шагах.
- Число шагов по умолчанию k* — наименьшее k из сетки, при котором S лучшей сети не ниже её S
  при 300 шагах минус 0,01 (дальше прибавка мала, а время растёт линейно).
- Модель по умолчанию — максимум S при k*; при разнице < 0,002 — больший P, затем меньшее время.
- Итерационные бейзлайны (GS, Adam со старта суперпозиции) — те же наборы и та же оценка.

Второй проход (--v2, results_v2.json): дооптимизация — двухэтапная схема methods.finetune
(Пирсон, затем Пирсон + SSIM), её параметры подобраны на отдельной выборке настройки
(ssim_tuning.py), а не на этих наборах. Правило выбора то же: шаги N* — наименьшее N из
FT_STEPS, при котором S лучшей сети не ниже S(300) − 0,01; модель — максимум S при N*.
Adam со старта суперпозиции считается по той же схеме.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from doesynth import methods, metrics, models
from doesynth.optics import chirps, reconstruct, target_amplitude

SETS = ("validation", "ood", "volume_test")
STEPS = (10, 25, 50, 100, 200, 300)
LRS = (0.05, 0.1, 0.2, 0.4)
LOSSES = (("pearson", 1.0), ("pearson_ssim", 1.0), ("pearson_ssim", 3.0))
ADAM_STEPS = (10, 25, 50, 100, 200, 300, 500, 1000)
GS_ITERS = (25, 50, 100, 200, 500)
GS_SEQ_ITERS = (5, 10, 25, 50, 100)
FT_STEPS = (20, 50, 100, 200, 300)
FT_ADAM_STEPS = (100, 200, 300, 500, 1000)


def run_v2(images, targets, chirp, files, put, save, t_start) -> None:
    p0 = superpos_batch(images, chirp)
    for k in FT_ADAM_STEPS:
        put(f"adam|ft|{k}", score_batch(methods.finetune(images, chirp, k, p0), targets, chirp))
    print(f"adam ft готово, {time.time() - t_start:.0f} с", flush=True)
    for f in files:
        model = models.load(f.resolve(), str(images.device))
        phase = net_phase(model, images)
        put(f"{f.stem}|0", score_batch(phase, targets, chirp))
        for k in FT_STEPS:
            put(
                f"{f.stem}|ft|{k}",
                score_batch(methods.finetune(images, chirp, k, phase), targets, chirp),
            )
        print(f"{f.stem} готово, {time.time() - t_start:.0f} с", flush=True)
        save()


def score_batch(phases: torch.Tensor, targets: np.ndarray, chirp: torch.Tensor) -> np.ndarray:
    """phases [B, N, N] → [B, 3] (SSIM, Пирсон, разделение) по сдаваемому пути."""
    codes = metrics.phase_to_u8(phases.detach().cpu().double().numpy())
    q = torch.from_numpy(metrics.u8_to_phase(codes)).float().to(chirp.device)
    amp = reconstruct(q, chirp).cpu().numpy()  # [B, 4, N, N]
    out = np.empty((len(codes), 3))
    for b in range(len(codes)):
        recons = np.stack([metrics.image_to_u8(a**2) for a in amp[b]])
        p = len(recons)
        cc = np.array(
            [[metrics.pearson(targets[b, i], recons[j]) for j in range(p)] for i in range(p)]
        )
        ss = np.mean([metrics.ssim(targets[b, i], recons[i]) for i in range(p)])
        diag = np.diag(cc)
        out[b] = (ss, diag.mean(), diag.mean() - cc[~np.eye(p, dtype=bool)].mean())
    return out


def superpos_batch(images: torch.Tensor, chirp: torch.Tensor) -> torch.Tensor:
    return torch.stack([methods.superposition_init(target_amplitude(x), chirp, 0) for x in images])


def net_phase(model, images: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([model(images[i : i + chunk]) for i in range(0, len(images), chunk)])


def adam_curve(images, chirp, phase0, steps, lr, kind, w, targets) -> dict[int, np.ndarray]:
    curve = {}

    def cb(k, phase):
        if k in steps:
            curve[k] = score_batch(phase, targets, chirp)

    methods.adam(images, chirp, max(steps), phase0, kind=kind, lr=lr, ssim_weight=w, callback=cb)
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", type=Path, default=Path("benchmark/sets.npz"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/results.json"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--v2", action="store_true", help="второй проход: схема finetune")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.sets)
    targets = data["images"]
    labels = data["set"]
    images = torch.from_numpy(targets.astype(np.float32) / 255).to(device)
    chirp = chirps(device=device)
    category = data["category"] if "category" in data else labels
    res: dict = {"sets": labels.tolist(), "category": category.tolist(), "runs": {}}
    t_start = time.time()

    def put(name: str, arr: np.ndarray) -> None:
        res["runs"][name] = arr.round(5).tolist()

    def save() -> None:
        args.out.write_text(json.dumps(res), encoding="utf-8")

    files = sorted(args.weights.glob("*.pt"))
    if args.models:
        files = [f for f in files if f.stem in args.models]
    if args.v2:
        res["finetune"] = methods.FINETUNE
        run_v2(images, targets, chirp, files, put, save, t_start)
        save()
        print("готово", args.out)
        return

    # Контроль без голограммы: в каждой плоскости среднее четырёх целей
    avg = np.rint(targets.astype(np.float64).mean(1)).astype(np.uint8)
    ctrl = np.array(
        [
            [
                np.mean([metrics.ssim(t[i], a) for i in range(4)]),
                np.mean([metrics.pearson(t[i], a) for i in range(4)]),
                0.0,
            ]
            for t, a in zip(targets, avg, strict=True)
        ]
    )
    put("control", ctrl)

    # Итерационные методы
    p0 = superpos_batch(images, chirp)
    put("superposition", score_batch(p0, targets, chirp))
    for kind, w in LOSSES:
        for lr in LRS:
            curve = adam_curve(images, chirp, p0, ADAM_STEPS, lr, kind, w, targets)
            for k, arr in curve.items():
                put(f"adam|{kind}|{w}|{lr}|{k}", arr)
        print(f"adam {kind} {w} готово, {time.time() - t_start:.0f} с", flush=True)
    for name, fn, grid in (
        ("gs", methods.gs_global, GS_ITERS),
        ("gs_seq", methods.gs_sequential, GS_SEQ_ITERS),
    ):
        phases = {k: [] for k in grid}
        for b in range(len(images)):
            amp = target_amplitude(images[b])
            ph, done = p0[b], 0
            for k in grid:
                ph = fn(amp, chirp, k - done, ph)
                done = k
                phases[k].append(ph)
        for k in grid:
            put(f"{name}|{k}", score_batch(torch.stack(phases[k]), targets, chirp))
        print(f"{name} готово, {time.time() - t_start:.0f} с", flush=True)

    # Сети и дооптимизация
    for f in files:
        model = models.load(f.resolve(), str(device))
        phase = net_phase(model, images)
        put(f"{f.stem}|0", score_batch(phase, targets, chirp))
        for kind, w in LOSSES:
            for lr in LRS:
                curve = adam_curve(images, chirp, phase, STEPS, lr, kind, w, targets)
                for k, arr in curve.items():
                    put(f"{f.stem}|{kind}|{w}|{lr}|{k}", arr)
        print(f"{f.stem} готово, {time.time() - t_start:.0f} с", flush=True)
        args.out.write_text(json.dumps(res), encoding="utf-8")
    args.out.write_text(json.dumps(res), encoding="utf-8")
    print("готово", args.out)


if __name__ == "__main__":
    main()
