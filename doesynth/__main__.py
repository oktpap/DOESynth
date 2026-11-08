"""Командная строка: python -m doesynth {serve, synth, batch, info}.

serve  — веб-интерфейс (по умолчанию http://0.0.0.0:8000)
synth  — один комплект: 4 изображения → ДОЭ, восстановления и метрики в папку
batch  — папка с комплектами (подпапки по 4 изображения) → результаты и summary.csv
info   — список моделей и методов
selftest — самопроверка: метрики, формат ДОЭ, физика и сквозной синтез
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from doesynth import catalog
from doesynth.catalog import DEFAULTS
from doesynth.imageio import is_image_name, load_image, save_png


def _specs(args) -> list:
    from doesynth import pipeline

    specs = []
    if args.mode in ("net", "compare"):
        steps = 0 if args.no_finetune else args.steps
        specs.append(pipeline.NetSpec(model=args.model, finetune=steps > 0, steps=steps))
    if args.mode in ("iter", "compare"):
        specs.append(pipeline.IterSpec(method=args.method, iters=args.iters))
    return specs


def _write(out: Path, targets: np.ndarray, results: list) -> list[dict]:
    rows = []
    for i, t in enumerate(targets):
        save_png(out / f"input_{i + 1}.png", t)
    for r in results:
        d = out if len(results) == 1 else out / r.kind
        save_png(d / "doe.png", r.doe_u8)
        for i, rec in enumerate(r.recons_u8):
            save_png(d / f"recon_{i + 1}.png", rec)
        (d / "metrics.json").write_text(
            json.dumps(r.summary(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        rows.append(r)
    return rows


def _print(kit: str, r) -> None:
    m = r.metrics
    print(
        f"{kit:24s} {r.kind:4s} SSIM {m['ssim_mean']:.4f}  Пирсон {m['pearson_mean']:.4f}  "
        f"синтез {r.seconds:.3f} с  ({r.label})",
        flush=True,
    )


def _kits(root: Path) -> dict[str, list[Path]]:
    direct = sorted(p for p in root.iterdir() if p.is_file() and is_image_name(p.name))
    if direct:
        return {root.name: direct}
    return {
        d.name: sorted(p for p in d.iterdir() if p.is_file() and is_image_name(p.name))
        for d in sorted(root.iterdir())
        if d.is_dir()
    }


def cmd_synth(args) -> None:
    from doesynth import pipeline

    if len(args.images) != 4:
        sys.exit("нужно ровно 4 изображения — по одному на плоскость (z = 24,1 … 27,7 мм)")
    loaded = [load_image(p) for p in args.images]
    for _, warns in loaded:
        for w in warns:
            print("предупреждение:", w)
    targets = np.stack([img for img, _ in loaded])
    results = [pipeline.run(targets, s) for s in _specs(args)]
    _write(Path(args.out), targets, results)
    for r in results:
        _print("комплект", r)
    print(f"файлы: {Path(args.out).resolve()}")


def cmd_batch(args) -> None:
    from doesynth import pipeline

    root, out = Path(args.input), Path(args.out)
    kits = _kits(root)
    specs = _specs(args)
    table = []
    for kit, files in kits.items():
        if len(files) != 4:
            print(f"{kit}: пропущен — {len(files)} изображений вместо 4", flush=True)
            continue
        targets = np.stack([load_image(p)[0] for p in files])
        results = [pipeline.run(targets, s) for s in specs]
        _write(out / kit, targets, results)
        for r in results:
            _print(kit, r)
            m = r.metrics
            table.append(
                {
                    "kit": kit,
                    "kind": r.kind,
                    "method": r.label,
                    "ssim_mean": round(m["ssim_mean"], 4),
                    "pearson_mean": round(m["pearson_mean"], 4),
                    "seconds": round(r.seconds, 4),
                    "total_seconds": round(r.total_seconds, 4),
                    **{f"ssim_{i + 1}": round(v, 4) for i, v in enumerate(m["ssim"])},
                    **{f"pearson_{i + 1}": round(v, 4) for i, v in enumerate(m["pearson"])},
                    "ssim_mean_absK": round(m["by_abs_K"]["ssim_mean"], 4),
                    "pearson_mean_absK": round(m["by_abs_K"]["pearson_mean"], 4),
                }
            )
    if not table:
        sys.exit("комплектов из 4 изображений не найдено")
    for kind in sorted({r["kind"] for r in table}):
        sel = [r for r in table if r["kind"] == kind]
        mean = {"kit": "СРЕДНЕЕ", "kind": kind, "method": sel[0]["method"]}
        for key in sel[0]:
            if key not in mean:
                mean[key] = round(float(np.mean([r[key] for r in sel])), 4)
        table.append(mean)
        print(
            f"{'СРЕДНЕЕ':24s} {kind:4s} SSIM {mean['ssim_mean']:.4f}  "
            f"Пирсон {mean['pearson_mean']:.4f}  синтез {mean['seconds']:.3f} с  "
            f"({len(sel)} комплектов)"
        )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"файлы: {out.resolve()}")


def cmd_serve(args) -> None:
    import uvicorn

    from doesynth import server

    server.start_warmup()
    print(f"DOESynth: откройте http://localhost:{args.port}", flush=True)
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="warning")


def cmd_selftest(_args) -> None:
    from doesynth import selftest

    sys.exit(0 if selftest.run() else 1)


def cmd_info(_args) -> None:
    from doesynth import pipeline

    print(f"устройство: {pipeline.device_label()}")
    print("сети (--model):")
    for m in catalog.NET_MODELS:
        mark = "*" if m.id == DEFAULTS.model else " "
        have = "" if m.path.is_file() else "  [нет весов]"
        print(f" {mark} {m.id:26s} {m.title}{have}")
    print("итерационные методы (--method):")
    for m in catalog.ITER_METHODS:
        print(f"   {m.id:26s} {m.title}, по умолчанию {m.default_iters} {m.unit}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m doesynth", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd")

    def method_args(p):
        p.add_argument("--mode", choices=("net", "iter", "compare"), default="net")
        p.add_argument("--model", default=DEFAULTS.model, help="сеть (см. info)")
        p.add_argument("--steps", type=int, default=DEFAULTS.finetune_steps, help="шаги Adam")
        p.add_argument("--no-finetune", action="store_true", help="без дооптимизации")
        p.add_argument("--method", default=DEFAULTS.iter_method, help="gs | gs_seq | adam")
        p.add_argument("--iters", type=int, default=None, help="итерации метода")

    p = sub.add_parser("serve", help="веб-интерфейс")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("synth", help="один комплект")
    p.add_argument("images", nargs="+", help="4 изображения по порядку плоскостей")
    p.add_argument("--out", default="result")
    method_args(p)
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("batch", help="папка с комплектами")
    p.add_argument("input", help="папка: подпапки по 4 изображения (или 4 изображения)")
    p.add_argument("out", help="папка для результатов")
    method_args(p)
    p.set_defaults(fn=cmd_batch)
    p = sub.add_parser("info", help="модели и методы")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("selftest", help="самопроверка")
    p.set_defaults(fn=cmd_selftest)

    args = parser.parse_args(argv)
    if args.cmd is None:
        args = parser.parse_args(["serve"])
    args.fn(args)


if __name__ == "__main__":
    main()
