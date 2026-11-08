"""Веб-сервер приложения: интерфейс (doesynth/static) и API над pipeline.

GET  /                      — интерфейс
GET  /api/config            — каталог методов, умолчания, параметры схемы, устройство
GET  /api/examples/{name}   — пример комплекта (4 изображения)
POST /api/images            — приведение загруженных файлов к условию (8 бит, серый, 128×128)
POST /api/run               — синтез одним или двумя методами (сравнение)
GET  /api/slice/{rid}/{i}   — восстановление результата i на произвольном z (мм, ?z=)
GET  /api/download/{rid}    — ZIP с ДОЭ, восстановлениями, входами и метриками
POST /api/batch             — пакетная обработка комплектов (папки или ZIP)
"""

import base64
import csv
import hashlib
import io
import json
import threading
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from doesynth import __version__, catalog, pipeline
from doesynth.imageio import is_image_name, load_image, png_bytes
from doesynth.optics import TASK

STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = catalog.ROOT / "examples"


class _Static(StaticFiles):
    """Скрипты и стили — с проверкой свежести при каждой загрузке: после обновления образа окно
    приложения сразу получает новые файлы, а не версию из кэша браузера."""

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="DOESynth", version=__version__, docs_url=None, redoc_url=None)
app.mount("/static", _Static(directory=STATIC), name="static")

_RESULTS: "OrderedDict[str, dict]" = OrderedDict()  # последние расчёты для скачивания и срезов
_KEEP = 16


def _remember(entry: dict) -> str:
    rid = uuid.uuid4().hex[:12]
    _RESULTS[rid] = entry
    while len(_RESULTS) > _KEEP:
        _RESULTS.popitem(last=False)
    return rid


def _b64(image_u8: np.ndarray) -> str:
    return base64.b64encode(png_bytes(image_u8)).decode("ascii")


def _decode(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.split(",", 1)[-1])
    img, _ = load_image(raw)
    return img


def _asset_version() -> str:
    """Короткий хэш содержимого static/: меняется с каждым обновлением интерфейса."""
    h = hashlib.sha256()
    for p in sorted(STATIC.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(STATIC).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


# Ссылки на скрипт и стили с версией: браузер, у которого в кэше лежат файлы прошлой версии
# (в том числе отданные без Cache-Control), всё равно загрузит новые.
_ASSET_V = _asset_version()
_INDEX = (
    (STATIC / "index.html")
    .read_text(encoding="utf-8")
    .replace("/static/app.js", f"/static/app.js?v={_ASSET_V}")
    .replace("/static/style.css", f"/static/style.css?v={_ASSET_V}")
)


@app.get("/")
def index() -> Response:
    return Response(
        _INDEX, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"}
    )


@app.get("/api/config")
def config() -> dict:
    examples = sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir()) if EXAMPLES.is_dir() else []
    return {
        "version": __version__,
        "device": pipeline.device_label(),
        "physics": {
            "wavelength_nm": TASK.wavelength * 1e9,
            "pixel_um": TASK.pixel * 1e6,
            "n": TASK.n,
            "distances_mm": [round(z * 1e3, 1) for z in TASK.distances],
        },
        "examples": examples,
        **catalog.describe(),
    }


def _kit_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and is_image_name(p.name))


@app.get("/api/examples/{name}")
def example(name: str) -> dict:
    folder = (EXAMPLES / name).resolve()
    if not folder.is_dir() or folder.parent != EXAMPLES.resolve():
        raise HTTPException(404, "нет такого примера")
    files = _kit_files(folder)[:4]
    images = [load_image(p)[0] for p in files]
    return {"names": [p.name for p in files], "images": [_b64(x) for x in images]}


@app.post("/api/images")
async def images(files: list[UploadFile] = File(...)) -> dict:
    out, warnings = [], []
    for f in sorted(files, key=lambda f: f.filename or ""):
        try:
            img, w = load_image(await f.read(), f.filename)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        out.append({"name": f.filename, "image": _b64(img)})
        warnings += w
    return {"images": out, "warnings": warnings}


class NetParams(BaseModel):
    model: str = catalog.DEFAULTS.model
    finetune: bool = catalog.DEFAULTS.finetune
    steps: int = catalog.DEFAULTS.finetune_steps


class IterParams(BaseModel):
    method: str = catalog.DEFAULTS.iter_method
    iters: int | None = None


class RunRequest(BaseModel):
    images: list[str]
    mode: str = "net"  # net | iter | compare
    net: NetParams = NetParams()
    iter: IterParams = IterParams()


def _net_spec(p: NetParams) -> pipeline.NetSpec:
    try:
        catalog.net_model(p.model)
    except KeyError as e:
        raise HTTPException(400, str(e)) from e
    steps = max(0, min(int(p.steps), catalog.DEFAULTS.max_finetune_steps))
    return pipeline.NetSpec(model=p.model, finetune=p.finetune and steps > 0, steps=steps)


def _iter_spec(p: IterParams) -> pipeline.IterSpec:
    try:
        method = catalog.iter_method(p.method)
    except KeyError as e:
        raise HTTPException(400, str(e)) from e
    iters = method.default_iters if p.iters is None else max(1, min(int(p.iters), method.max_iters))
    return pipeline.IterSpec(method=p.method, iters=iters)


def _result_json(r: pipeline.Result) -> dict:
    return {
        "kind": r.kind,
        "label": r.label,
        "params": r.params,
        "doe": _b64(r.doe_u8),
        "recons": [_b64(x) for x in r.recons_u8],
        "metrics": r.metrics,
        "seconds": r.seconds,
        "total_seconds": r.total_seconds,
    }


@app.post("/api/run")
def run(req: RunRequest) -> dict:
    if len(req.images) != 4:
        raise HTTPException(400, "нужно ровно 4 изображения — по одному на плоскость")
    try:
        targets = np.stack([_decode(b) for b in req.images])
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    specs: list = []
    if req.mode in ("net", "compare"):
        specs.append(_net_spec(req.net))
    if req.mode in ("iter", "compare"):
        specs.append(_iter_spec(req.iter))
    if not specs:
        raise HTTPException(400, f"неизвестный режим {req.mode!r}")
    results = [pipeline.run(targets, s) for s in specs]
    rid = _remember({"targets": targets, "results": results})
    return {
        "id": rid,
        "results": [_result_json(r) for r in results],
        "control": pipeline.control(targets),
    }


@app.get("/api/slice/{rid}/{index}")
def slice_png(rid: str, index: int, z: float) -> Response:
    entry = _RESULTS.get(rid)
    if entry is None or not 0 <= index < len(entry["results"]):
        raise HTTPException(404, "результат устарел — запустите синтез ещё раз")
    # ниже NΔx²/λ = 24,06 мм чирп формулы с одним БПФ дискретизован с наложением
    z = float(np.clip(z, 24.06, 200.0))
    img = pipeline.slice_at(entry["results"][index].phase, z)
    return Response(png_bytes(img), media_type="image/png", headers={"Cache-Control": "no-store"})


def _slug(label: str) -> str:
    keep = [c if c.isalnum() else "_" for c in label.lower()]
    return "_".join(filter(None, "".join(keep).split("_")))[:60]


def _write_result(zf: zipfile.ZipFile, prefix: str, r: pipeline.Result) -> None:
    zf.writestr(f"{prefix}doe.png", png_bytes(r.doe_u8))
    for i, rec in enumerate(r.recons_u8):
        zf.writestr(f"{prefix}recon_{i + 1}.png", png_bytes(rec))
    zf.writestr(f"{prefix}metrics.json", json.dumps(r.summary(), ensure_ascii=False, indent=1))


def _summary_csv(rows: list[dict]) -> bytes:
    """CSV в UTF-8 с BOM (чтобы Excel показал кириллицу); средние по комплектам — в конце."""
    cols = ["kit", "method", "ssim_mean", "pearson_mean", "seconds", "total_seconds"]
    cols += [f"ssim_{i + 1}" for i in range(4)] + [f"pearson_{i + 1}" for i in range(4)]
    cols += ["ssim_mean_absK", "pearson_mean_absK"]
    table = list(rows)
    for method in dict.fromkeys(r["method"] for r in rows):
        sel = [r for r in rows if r["method"] == method]
        if len(sel) > 1:
            mean = {c: round(float(np.mean([r[c] for r in sel])), 4) for c in cols[2:]}
            table.append({"kit": "СРЕДНЕЕ", "method": method, **mean})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
    return buf.getvalue().encode("utf-8-sig")


def _row(kit: str, r: pipeline.Result) -> dict:
    m = r.metrics
    row = {
        "kit": kit,
        "method": r.label,
        "ssim_mean": round(m["ssim_mean"], 4),
        "pearson_mean": round(m["pearson_mean"], 4),
        "seconds": round(r.seconds, 4),
        "total_seconds": round(r.total_seconds, 4),
    }
    for i in range(4):
        row[f"ssim_{i + 1}"] = round(m["ssim"][i], 4)
        row[f"pearson_{i + 1}"] = round(m["pearson"][i], 4)
    k = m.get("by_abs_K")
    if k:  # справочно: восстановление = |K| вместо |K|²
        row["ssim_mean_absK"] = round(k["ssim_mean"], 4)
        row["pearson_mean_absK"] = round(k["pearson_mean"], 4)
    return row


@app.get("/api/download/{rid}")
def download(rid: str) -> Response:
    entry = _RESULTS.get(rid)
    if entry is None:
        raise HTTPException(404, "результат устарел — запустите синтез ещё раз")
    buf = io.BytesIO()
    rows = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if "kits" in entry:  # пакет
            for kit, targets, results in entry["kits"]:
                for i, t in enumerate(targets):
                    zf.writestr(f"{kit}/input_{i + 1}.png", png_bytes(t))
                for r in results:
                    sub = "" if len(results) == 1 else f"{_slug(r.kind)}/"
                    _write_result(zf, f"{kit}/{sub}", r)
                    rows.append(_row(kit, r))
        else:
            for i, t in enumerate(entry["targets"]):
                zf.writestr(f"input_{i + 1}.png", png_bytes(t))
            for r in entry["results"]:
                sub = "" if len(entry["results"]) == 1 else f"{r.kind}/"
                _write_result(zf, sub, r)
                rows.append(_row("kit", r))
        zf.writestr("summary.csv", _summary_csv(rows))
    name = "doe_batch.zip" if "kits" in entry else "doe_result.zip"
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


def _group_kits(items: list[tuple[str, bytes]]) -> dict[str, list[tuple[str, bytes]]]:
    """Файлы с относительными путями → комплекты по папкам. Если все файлы лежат в одной
    папке и их ровно 4 — это один комплект."""
    kits: dict[str, list[tuple[str, bytes]]] = {}
    for path, data in items:
        # без «..», «.» и корня: имена комплектов попадают в пути внутри ZIP с результатами
        parts = [
            x for x in PurePosixPath(path.replace("\\", "/")).parts if x not in ("..", ".", "/")
        ]
        if not parts or not is_image_name(parts[-1]) or "__MACOSX" in parts:
            continue
        kit = "/".join(parts[:-1]) or "kit"
        kits.setdefault(kit, []).append((parts[-1], data))
    # общий корень (выбранная папка) из имён комплектов убираем
    names = list(kits)
    if len(names) > 1:
        roots = {n.split("/", 1)[0] for n in names}
        if len(roots) == 1 and all("/" in n for n in names):
            kits = {n.split("/", 1)[1]: v for n, v in kits.items()}
    elif names and "/" in names[0]:
        kits = {names[0].rsplit("/", 1)[-1]: kits[names[0]]}
    return {k: sorted(v) for k, v in sorted(kits.items())}


@app.post("/api/batch")
async def batch(
    files: list[UploadFile] = File(...),
    paths: str = Form("[]"),
    settings: str = Form("{}"),
) -> dict:
    rel = json.loads(paths)
    items: list[tuple[str, bytes]] = []
    for i, f in enumerate(files):
        data = await f.read()
        name = rel[i] if i < len(rel) and rel[i] else (f.filename or f"file{i}")
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                items += [(n, zf.read(n)) for n in zf.namelist() if not n.endswith("/")]
        else:
            items.append((name, data))
    # расчёт — в пуле потоков, чтобы долгий пакет не блокировал остальные запросы
    return await run_in_threadpool(_batch, items, json.loads(settings))


def _batch(items: list[tuple[str, bytes]], opts: dict) -> dict:
    kits = _group_kits(items)
    if not kits:
        raise HTTPException(400, "не найдено изображений: выберите папку с комплектами или ZIP")
    mode = opts.get("mode", "net")
    specs = []
    if mode in ("net", "compare"):
        specs.append(_net_spec(NetParams(**opts.get("net", {}))))
    if mode in ("iter", "compare"):
        specs.append(_iter_spec(IterParams(**opts.get("iter", {}))))
    out, stored, rows = [], [], []
    for kit, kit_files in kits.items():
        if len(kit_files) != 4:
            out.append({"kit": kit, "error": f"в комплекте {len(kit_files)} изображений, нужно 4"})
            continue
        try:
            targets = np.stack([load_image(data, name)[0] for name, data in kit_files])
        except ValueError as e:
            out.append({"kit": kit, "error": str(e)})
            continue
        results = [pipeline.run(targets, s) for s in specs]
        stored.append((kit, targets, results))
        out.append(
            {
                "kit": kit,
                "files": [name for name, _ in kit_files],
                "doe": _b64(results[0].doe_u8),
                "results": [_row(kit, r) | {"kind": r.kind} for r in results],
            }
        )
        rows += [_row(kit, r) | {"kind": r.kind} for r in results]
    rid = _remember({"kits": stored}) if stored else None
    means = {}
    for kind in ("net", "iter"):
        sel = [r for r in rows if r["kind"] == kind]
        if sel:
            means[kind] = {
                k: round(float(np.mean([r[k] for r in sel])), 4)
                for k in ("ssim_mean", "pearson_mean", "seconds", "total_seconds")
            } | {"kits": len(sel), "method": sel[0]["method"]}
    return {"id": rid, "kits": out, "means": means}


@app.exception_handler(Exception)
async def _errors(_, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


def start_warmup() -> None:
    threading.Thread(target=pipeline.warm_default, daemon=True).start()
