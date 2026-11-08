"""Набор сечений 3D-объектов (doetrain.train.volume) на диске: экспорт, проверка, контактные листы.

    python -m doetrain.train.volume_export export --out data/volume \
        --train 2000 --validation 200 --test 200 --edge 100 --workers 8
    python -m doetrain.train.volume_export verify data/volume
    python -m doetrain.train.volume_export sheet data/volume --split test

Структура: <out>/manifest.json — параметры генератора, оптики и разбиений;
<out>/<split>/<index>/plane_1.png … plane_4.png (uint8 128×128, плоскости z₁…z₄ — те же
четыре файла, что принимает `python -m doesynth synth`) и meta.json (сид, семейство, параметры объекта,
координаты сечений, статус каждого сечения, sha256 массива [4, 128, 128]);
<out>/<split>/index.jsonl — строка на пример; с --npy ещё <out>/<split>.npy — uint8 [M, 4, N, N].
Пример index — объект index // variants и его вариант index % variants, как в
volume.dataset_item; verify строит каждый пример заново из сида и из описания объекта
и сравнивает с PNG. Канон — PNG: на той же платформе совпадение побайтное; на другой ОС или
версии numpy отдельные пиксели на границах форм могут отличаться (разные exp, cos, arctan2
в libm), поэтому manifest хранит окружение, а verify принимает допуск --max-levels/--max-pixels.
"""

import hashlib
import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image, ImageDraw, ImageFont

from doetrain.core.io import save_png
from doetrain.core.optics import TASK
from doetrain.train import volume
from doetrain.train.volume import (
    SEED,
    SPLITS,
    SUPERSAMPLE,
    VolumeGenerator,
    dataset_item,
    item_meta,
    object_key,
    planes,
    side_view,
    to_u8,
)

NOTE = (
    "Геометрические сечения одного 3D-объекта на глубинах плоскостей задачи; это не распределения "
    "света, достижимые одним фазовым ДОЭ: дефокус соседних слоёв не добавлен."
)


def environment() -> dict:
    """Где выгружен набор: от версий numpy, scipy и ОС зависят последние биты рендера."""
    import scipy

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True,
                timeout=10, check=True,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }  # fmt: skip


def _sha(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


def _export_object(task: tuple) -> list[tuple[dict, np.ndarray]]:
    """Объект obj разбиения split и его варианты: пишет PNG и meta.json, возвращает строки
    index.jsonl и массивы. Функция верхнего уровня — для пула процессов."""
    root, split, obj, variants, count, seed, families, supersample = task
    gen = VolumeGenerator(families, TASK, supersample, edge=split == "edge")
    rng = np.random.default_rng(object_key(split, obj, 0, seed))
    base_img, base_desc = gen.sample(rng, gen.split_family(split, obj))
    out = []
    for var in range(variants):
        index = obj * variants + var
        if index >= count:
            break
        img, desc = base_img, base_desc
        if var:
            img, desc = gen.vary(
                base_desc, np.random.default_rng(object_key(split, obj, var, seed))
            )
        meta = item_meta(gen, split, index, variants, seed, img, desc)
        item = Path(root) / split / f"{index:06d}"
        for k, plane in enumerate(img):
            save_png(item / f"plane_{k + 1}.png", plane)
        (item / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
        line = {
            "id": meta["id"], "index": index, "object_index": obj, "variant": var,
            "family": meta["family"], "layout": desc.get("layout"),
            "status": [p["status"] for p in meta["planes"]],
            "coverage": [p["coverage"] for p in meta["planes"]], "sha256": meta["sha256"],
        }  # fmt: skip
        out.append((line, img))
    return out


def export(
    out: str | Path,
    counts: dict[str, int],
    variants: int = 1,
    seed: int = SEED,
    families: dict[str, float] | None = None,
    supersample: int = SUPERSAMPLE,
    npy: bool = False,
    workers: int = 1,
    on_split=None,
) -> Path:
    """Пишет набор в out (каталог должен быть новым или пустым) и возвращает путь к manifest.json."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} не пуст: укажите новый каталог")
    unknown = set(counts) - set(SPLITS)
    if unknown or variants < 1:
        raise ValueError(f"разбиения — из {tuple(SPLITS)}, variants ≥ 1; получено {counts}")
    weights = VolumeGenerator(families).weights
    manifest = {
        "generator": volume.__name__, "version": volume.VERSION, "note": NOTE, "seed": seed,
        "variants": variants, "supersample": supersample, "families": weights,
        "optics": {"wavelength_m": TASK.wavelength, "pixel_m": TASK.pixel, "n": TASK.n,
                   "distances_m": list(TASK.distances)},
        "planes": [{"plane": k + 1, "z_mm": round(p.z, 6), "pitch_um": round(p.pitch * 1e3, 6),
                    "fov_mm": round(p.fov, 6)} for k, p in enumerate(planes(TASK))],
        "grid": "строка i, столбец j плоскости k ↔ x = (j − N/2)·p_k, y = (i − N/2)·p_k, z = z_k, мм",
        "environment": environment(), "splits": {},
    }  # fmt: skip
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        for split, count in counts.items():
            if count <= 0:
                continue
            objects = -(-count // variants)
            tasks = [(str(out), split, o, variants, count, seed, weights, supersample)
                     for o in range(objects)]  # fmt: skip
            results = (
                pool.map(_export_object, tasks, chunksize=4) if pool else map(_export_object, tasks)
            )
            lines, stack = [], []
            for items in results:
                for line, img in items:
                    lines.append(line)
                    if npy:
                        stack.append(img)
            (out / split).mkdir(parents=True, exist_ok=True)
            text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
            (out / split / "index.jsonl").write_text(text, "utf-8")
            if npy:
                np.save(out / f"{split}.npy", np.stack(stack))
            families_count = {}
            for line in lines:
                families_count[line["family"]] = families_count.get(line["family"], 0) + 1
            manifest["splits"][split] = {
                "count": count, "objects": objects, "code": SPLITS[split],
                "edge": split == "edge", "families": dict(sorted(families_count.items())),
            }  # fmt: skip
            if on_split:
                on_split(split, count)
    finally:
        if pool:
            pool.shutdown()
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
    return path


def read_item(root: str | Path, split: str, index: int) -> tuple[np.ndarray, dict]:
    item = Path(root) / split / f"{index:06d}"
    meta = json.loads((item / "meta.json").read_text("utf-8"))
    img = []
    for p in meta["planes"]:
        with Image.open(item / p["file"]) as png:
            img.append(np.asarray(png))
    return np.stack(img), meta


def verify(
    root: str | Path, from_seed: bool = True, max_levels: int = 0, max_pixels: int = 0
) -> dict:
    """Каждый пример: sha256 PNG = meta, рендер из описания объекта = PNG и (from_seed) пример,
    построенный заново из сида, = PNG. С допуском (max_levels, max_pixels) проверка считается
    пройденной, если отличаются не больше max_pixels пикселей и не больше чем на max_levels;
    точные совпадения считаются отдельно. → сводка и до 10 непрошедших примеров."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    report = {"items": 0, "exact": 0, "sha_mismatch": 0, "desc_mismatch": 0,
              "seed_mismatch": 0, "max_abs_diff": 0, "max_diff_pixels": 0, "failed": []}  # fmt: skip
    ss, weights = manifest["supersample"], manifest["families"]
    for split, info in manifest["splits"].items():
        gen = VolumeGenerator(weights, TASK, ss, edge=split == "edge")
        for index in range(info["count"]):
            img, meta = read_item(root, split, index)
            again = {"desc": gen.render(meta["object"])}
            if from_seed:
                again["seed"] = dataset_item(
                    split, index, manifest["variants"], manifest["seed"], weights, TASK, ss
                )[0]
            checks, exact = {"sha": _sha(img) == meta["sha256"]}, _sha(img) == meta["sha256"]
            for name, other in again.items():
                diff = np.abs(other.astype(np.int64) - img)
                levels, pixels = int(diff.max()), int((diff > 0).sum())
                report["max_abs_diff"] = max(report["max_abs_diff"], levels)
                report["max_diff_pixels"] = max(report["max_diff_pixels"], pixels)
                checks[name] = levels <= max_levels and pixels <= max_pixels
                exact &= pixels == 0
            report["items"] += 1
            report["exact"] += exact
            for name, ok in checks.items():
                if not ok:
                    report[f"{name}_mismatch"] += 1
            if not all(checks.values()) and len(report["failed"]) < 10:
                report["failed"].append(meta["id"])
    report["ok"] = not report["failed"] and report["items"] > 0
    here, there = environment(), manifest.get("environment", {})
    report["same_environment"] = all(
        here.get(k) == there.get(k) for k in ("python", "numpy", "scipy", "platform")
    )
    return report


# ---------- Контактные листы ----------

BG, FG, MUTED, FRAME, MARK = (
    (22, 22, 22),
    (230, 230, 230),
    (150, 150, 150),
    (70, 70, 70),
    (255, 140, 0),
)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # старый Pillow без размера у встроенного шрифта
        return ImageFont.load_default()


def contact_sheet(
    rows: list[tuple[list[str], np.ndarray, dict | None]], title: str = "", setup=TASK
) -> Image.Image:
    """Строка — объект: подпись, четыре сечения (uint8 [4, n, n]) по столбцам и, если задано,
    вид сбоку (описание объекта → volume.side_view) с отметками плоскостей и их полей зрения."""
    pls = planes(setup)
    n, gap, label_w, head = setup.n, 6, 190, 58 + (22 if title else 0)
    sides = [side_view(desc, setup) if desc is not None else None for _, _, desc in rows]
    side_w = max((s[0].shape[1] for s in sides if s is not None), default=0)
    heights = [n if s is None else max(n, s[0].shape[0]) for s in sides]
    width = label_w + 4 * (n + gap) + (side_w + 3 * gap if side_w else 0) + gap
    height = head + sum(heights) + gap * (len(rows) + 1)
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    font, small = _font(13), _font(11)
    y0 = gap
    if title:
        draw.text((gap, y0), title, fill=FG, font=_font(15))
        y0 += 22
    for k, p in enumerate(pls):
        x = label_w + k * (n + gap)
        draw.text((x, y0), f"plane {k + 1}: z = {p.z:.1f} mm", fill=FG, font=font)
        draw.text((x, y0 + 16), f"pitch {p.pitch * 1e3:.2f} um", fill=MUTED, font=small)
        draw.text((x, y0 + 30), f"FOV {p.fov:.3f} mm", fill=MUTED, font=small)
    if side_w:
        x = label_w + 4 * (n + gap) + 2 * gap
        draw.text(
            (x, y0), "side view: max over y (x down, z right, same scale)", fill=FG, font=font
        )
        draw.text((x, y0 + 16), "orange: plane z_k and its field of view", fill=MUTED, font=small)
    y = head
    for (label, img, _), side, row_h in zip(rows, sides, heights, strict=True):
        for i, line in enumerate(label):
            draw.text((gap, y + 2 + 15 * i), line, fill=FG if i == 0 else MUTED, font=small)
        for k in range(4):
            x = label_w + k * (n + gap)
            sheet.paste(Image.fromarray(img[k]).convert("RGB"), (x, y))
            draw.rectangle([x - 1, y - 1, x + n, y + n], outline=FRAME)
        if side is not None:
            view, xs, zs = side
            x = label_w + 4 * (n + gap) + 2 * gap
            sheet.paste(Image.fromarray(to_u8(view)).convert("RGB"), (x, y))
            draw.rectangle([x - 1, y - 1, x + view.shape[1], y + view.shape[0]], outline=FRAME)
            for p in pls:
                col = x + int(np.argmin(np.abs(zs - p.z)))
                inside = np.flatnonzero(np.abs(xs) <= p.fov / 2)
                draw.line([(col, y + inside[0]), (col, y + inside[-1])], fill=MARK)
        y += row_h + gap
    return sheet


def row_label(meta: dict) -> list[str]:
    obj = meta["object"]
    kind = obj["family"] + (f" / {obj['layout']}" if obj.get("layout") else "")
    status = " ".join(p["status"] for p in meta["planes"])
    cover = " ".join(f"{p['coverage']:.2f}" for p in meta["planes"])
    lines = [meta["id"], kind, f"status: {status}", f"coverage: {cover}"]
    if meta["variant"]:
        lines.append(f"variant {meta['variant']} of object {meta['object_index']}")
    return lines


# ---------- Командная строка ----------

app = typer.Typer(help="Набор сечений 3D-объектов: экспорт, проверка, контактные листы.")


@app.command("export")
def export_cmd(
    out: Annotated[Path, typer.Option(help="Новый каталог набора")] = Path("data/volume"),
    train: int = 2000,
    validation: int = 200,
    test: int = 200,
    edge: Annotated[int, typer.Option(help="Крайние случаи: пустые и сплошные сечения")] = 100,
    variants: Annotated[int, typer.Option(help="Примеров на объект (близкие варианты)", min=1)] = 1,
    seed: int = SEED,
    supersample: Annotated[
        int, typer.Option(help="Подотсчётов на пиксель по оси", min=1)
    ] = SUPERSAMPLE,
    npy: Annotated[bool, typer.Option(help="Ещё <split>.npy: uint8 [M, 4, N, N]")] = False,
    workers: Annotated[int, typer.Option(help="Процессов", min=1)] = 1,
) -> None:
    """Пишет train/validation/test/edge: 4 PNG и meta.json на пример, index.jsonl, manifest.json."""
    counts = {"train": train, "validation": validation, "test": test, "edge": edge}
    path = export(
        out, counts, variants, seed, None, supersample, npy, workers,
        on_split=lambda s, c: typer.echo(f"{s}: {c} примеров"),
    )  # fmt: skip
    typer.echo(f"готово: {path}")


@app.command("verify")
def verify_cmd(
    root: Path,
    from_seed: Annotated[bool, typer.Option(help="Ещё и строить заново из сида")] = True,
    max_levels: Annotated[int, typer.Option(help="Допуск: уровней из 255 (другая ОС — 16)")] = 0,
    max_pixels: Annotated[
        int, typer.Option(help="Допуск: пикселей на пример (другая ОС — 10)")
    ] = 0,
) -> None:
    """Сверяет PNG с meta.json, с рендером из описания объекта и с генерацией из сида."""
    report = verify(root, from_seed, max_levels, max_pixels)
    typer.echo(json.dumps(report, ensure_ascii=False))
    raise typer.Exit(0 if report["ok"] else 1)


@app.command("sheet")
def sheet_cmd(
    root: Path,
    split: str = "test",
    rows: int = 12,
    start: int = 0,
    side: Annotated[bool, typer.Option(help="Добавить вид сбоку для проверки геометрии")] = False,
    out: Annotated[
        Path | None, typer.Option(help="PNG; по умолчанию <root>/sheet_<split>.png")
    ] = None,
) -> None:
    """Контактный лист: строка — объект, четыре столбца — его сечения."""
    count = json.loads((root / "manifest.json").read_text("utf-8"))["splits"][split]["count"]
    items = [read_item(root, split, i) for i in range(start, min(start + rows, count))]
    sheet = contact_sheet(
        [(row_label(m), img, m["object"] if side else None) for img, m in items],
        title=f"{root.name} / {split}",
    )
    out = out or root / f"sheet_{split}.png"
    sheet.save(out)
    typer.echo(f"лист: {out}")


if __name__ == "__main__":
    app()
