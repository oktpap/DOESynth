"""Время синтеза на CPU: сеть, шаг дооптимизации и итерационные методы (после прогрева).

    python benchmark/timing.py --out benchmark/timing.json

Замер — как в приложении (pipeline.run): чтение весов и прогрев вне замера, медиана повторов.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from doesynth import catalog, methods, models
from doesynth.optics import chirps, target_amplitude


def median_time(fn, repeats: int) -> float:
    fn()  # прогрев
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", type=Path, default=Path("benchmark/sets.npz"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/timing.json"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    torch.set_grad_enabled(True)
    quad = np.load(args.sets)["images"][0]
    images = torch.from_numpy(quad.astype(np.float32) / 255)
    chirp = chirps()
    amp = target_amplitude(images)
    p0 = methods.superposition_init(amp, chirp, 0)
    out = {
        "device": f"CPU, {torch.get_num_threads()} потоков",
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "models": {},
        "iterative": {},
    }

    # время одного шага дооптимизации (двухэтапная схема methods.finetune): по 200 шагам
    out["adam_step_s"] = (
        median_time(lambda: methods.finetune(images, chirp, 200, p0), args.repeats) / 200
    )
    out["gs_iter_s"] = median_time(lambda: methods.gs_global(amp, chirp, 50, p0), args.repeats) / 50
    out["gs_seq_iter_s"] = (
        median_time(lambda: methods.gs_sequential(amp, chirp, 20, p0), args.repeats) / 20
    )
    out["superposition_s"] = median_time(
        lambda: methods.superposition_init(amp, chirp, 0), args.repeats
    )
    for m in catalog.NET_MODELS:
        if not m.path.is_file():
            continue
        net = models.load(m.path, "cpu")

        def forward(net=net):
            with torch.no_grad():
                return net(images[None])[0]

        out["models"][m.id] = {
            "net_s": median_time(forward, args.repeats),
            "params": models.count_parameters(net),
        }
        print(m.id, round(out["models"][m.id]["net_s"] * 1000, 1), "мс", flush=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
