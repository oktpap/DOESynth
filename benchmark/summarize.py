"""Сводка бенчмарка и выбор умолчаний по правилу из docstring benchmark.py.

    python benchmark/summarize.py   → benchmark/summary.json (для интерфейса), benchmark/summary.md

Первый проход (results.json): дооптимизация одним Пирсоном, сетка критериев и lr. Второй
(results_v2.json): двухэтапная схема methods.finetune, по ней выбираются умолчания.
"""

import argparse
import json
from pathlib import Path

import numpy as np

SETS = ("validation", "ood", "volume_test")
SET_TITLE = {
    "validation": "validation (4)",
    "ood": "незнакомые категории (24)",
    "volume_test": "сечения 3D (200)",
}
STEPS_V1 = (10, 25, 50, 100, 200, 300)
STEPS_V2 = (20, 50, 100, 200, 300)
TIE = 0.002
TOL = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("benchmark/results.json"))
    parser.add_argument("--results-v2", type=Path, default=Path("benchmark/results_v2.json"))
    parser.add_argument("--timing", type=Path, default=Path("benchmark/timing.json"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/summary.json"))
    parser.add_argument("--md", type=Path, default=Path("benchmark/summary.md"))
    parser.add_argument(
        "--select",
        type=Path,
        default=Path("benchmark/results_tune_v2.json"),
        help="второй проход на выборке настройки: по нему выбираются шаги и модель",
    )
    args = parser.parse_args()

    res = json.loads(args.results.read_text(encoding="utf-8"))
    timing = json.loads(args.timing.read_text(encoding="utf-8")) if args.timing.is_file() else {}
    labels = np.array(res["sets"])
    runs = {k: np.array(v) for k, v in res["runs"].items()}
    v2 = args.results_v2.is_file()
    if v2:
        res2 = json.loads(args.results_v2.read_text(encoding="utf-8"))
        assert res2["sets"] == res["sets"], "наборы проходов расходятся"
        runs |= {k: np.array(v) for k, v in res2["runs"].items()}

    def per_set(name: str, col: int) -> dict[str, float]:
        a = runs[name][:, col]
        return {s: float(a[labels == s].mean()) for s in SETS}

    def S(name):  # обозначение из правила
        return float(np.mean(list(per_set(name, 0).values())))

    def P(name):
        return float(np.mean(list(per_set(name, 1).values())))

    nets = sorted({k.split("|")[0] for k in runs if k.endswith("|0") and k.count("|") == 1})
    nets = [m for m in nets if m not in ("gs", "gs_seq")]

    # Первый проход: критерий и lr — максимум среднего по сетям S при k = 100
    configs = sorted(
        {tuple(k.split("|")[1:4]) for k in runs if k.split("|")[0] in nets and k.count("|") == 4}
    )
    cfg_score = {c: float(np.mean([S(f"{m}|{'|'.join(c)}|100") for m in nets])) for c in configs}
    best_cfg = max(cfg_score, key=cfg_score.get)

    tag, steps_grid = ("ft", STEPS_V2) if v2 else ("|".join(best_cfg), STEPS_V1)

    def key(m: str, k: int) -> str:
        return f"{m}|{tag}|{k}"

    def net_time(m: str, k: int) -> float:
        t = timing.get("models", {}).get(m, {}).get("net_s")
        return (t or 0) + k * timing.get("adam_step_s", 0)

    # Выбор шагов и модели — на выборке настройки (если есть), отчёт — на наборах отчёта
    sel_sets, sel_runs, sel_labels = SETS, runs, labels
    selected_on = "наборы отчёта"
    if v2 and args.select.is_file():
        rs = json.loads(args.select.read_text(encoding="utf-8"))
        sel_labels = np.array(rs["sets"])
        sel_sets = tuple(dict.fromkeys(rs["sets"]))
        sel_runs = {k: np.array(v) for k, v in rs["runs"].items()}
        selected_on = "выборка настройки (" + ", ".join(sel_sets) + ")"

    def sel(name: str, col: int) -> float:
        a = sel_runs[name][:, col]
        return float(np.mean([a[sel_labels == s].mean() for s in sel_sets]))

    s_best = {k: max(sel(key(m, k), 0) for m in nets) for k in steps_grid}
    k_star = min(k for k in steps_grid if s_best[k] >= s_best[300] - TOL)
    sel_ranked = sorted(nets, key=lambda m: -sel(key(m, k_star), 0))
    top = sel(key(sel_ranked[0], k_star), 0)
    tied = [m for m in sel_ranked if top - sel(key(m, k_star), 0) < TIE]
    default = sorted(tied, key=lambda m: (-sel(key(m, k_star), 1), net_time(m, k_star)))[0]
    best_pure = max(nets, key=lambda m: (sel(f"{m}|0", 0), sel(f"{m}|0", 1)))
    ranked = sorted(nets, key=lambda m: -S(key(m, k_star)))

    adam_v1 = [c for c in {tuple(k.split("|")[1:4]) for k in runs if k.startswith("adam|")}]
    adam_v1 = [c for c in adam_v1 if c[0] == "pearson"]
    adam_lr = max(adam_v1, key=lambda c: S(f"adam|{'|'.join(c)}|300"))
    adam_key = "adam|ft|300" if v2 else f"adam|{'|'.join(adam_lr)}|300"
    iterative = {
        "gs": ("gs|200", 200, timing.get("gs_iter_s")),
        "gs_seq": ("gs_seq|50", 50, timing.get("gs_seq_iter_s")),
        "adam": (adam_key, 300, timing.get("adam_step_s")),
    }

    summary = {
        "meta": {
            "label": "Бенчмарк, 228 комплектов",
            "sets": list(SETS),
            "finetune": res2.get("finetune") if v2 else {"loss": best_cfg},
            "default_steps": k_star,
            "selected_on": selected_on,
            "default_model": default,
            "best_pure_model": best_pure,
            "device": timing.get("device"),
        },
        "models": {},
        "iterative": {},
        "control": {"ssim": S("control"), "pearson": P("control")},
    }
    for m in nets:
        t = timing.get("models", {}).get(m, {})
        summary["models"][m] = {
            "net": {"ssim": S(f"{m}|0"), "pearson": P(f"{m}|0")},
            "finetune": {"ssim": S(key(m, k_star)), "pearson": P(key(m, k_star)), "steps": k_star},
            "net_s": t.get("net_s"),
            "finetune_s": net_time(m, k_star) if t else None,
            "params": t.get("params"),
        }
    for mid, (k, iters, unit) in iterative.items():
        seconds = unit * iters + timing.get("superposition_s", 0) if unit else None
        summary["iterative"][mid] = {
            "ssim": S(k),
            "pearson": P(k),
            "iters": iters,
            "seconds": seconds,
        }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- Markdown ---
    def f3(x: float) -> str:
        return f"{x:.3f}"

    def table_row(label: str, k: str, times: tuple = ()) -> str:
        ps = per_set(k, 0)
        cells = [f3(ps[s]) for s in SETS] + [f3(S(k)), f3(P(k))] + list(times)
        return f"| {label} | " + " | ".join(cells) + " |"

    def head(extra: tuple = ()) -> list[str]:
        cols = [f"SSIM: {SET_TITLE[s]}" for s in SETS] + ["SSIM", "Пирсон", *extra]
        return ["| метод | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]

    def ms(m: str) -> str:
        t = timing.get("models", {}).get(m, {}).get("net_s")
        return f"{t * 1000:.0f}" if t else "—"

    lines = [
        "# Сводка бенчмарка",
        "",
        "SSIM и Пирсон — средние по трём наборам (каждый набор — среднее по четвёркам), по "
        "сдаваемому пути: 8-битный ДОЭ → Френель → |K|² → 8 бит (максимум → 255).",
        f"Время — CPU в контейнере ({timing.get('device', '—')}), после прогрева.",
        "",
        f"Выбор сделан на: {selected_on}; таблицы ниже — по наборам отчёта.",
        f"Шагов дооптимизации по умолчанию: **{k_star}** (наименьшее N, где SSIM лучшей сети "
        f"≥ SSIM при 300 шагах − {TOL}). Модель по умолчанию: **{default}** (максимум SSIM, "
        f"при разнице < {TIE} — больший Пирсон). Лучшая сеть без дооптимизации: "
        f"**{best_pure}**.",
        "",
        "SSIM лучшей сети на выборке выбора в зависимости от числа шагов дооптимизации:",
        "",
        "| шаги | " + " | ".join(str(k) for k in steps_grid) + " |",
        "|---|" + "---|" * len(steps_grid),
        "| SSIM | " + " | ".join(f3(s_best[k]) for k in steps_grid) + " |",
        "",
        "Пять лучших сетей на выборке выбора (SSIM / Пирсон при выбранном N): "
        + "; ".join(
            f"{m} {sel(key(m, k_star), 0):.4f} / {sel(key(m, k_star), 1):.4f}"
            for m in sel_ranked[:5]
        )
        + ".",
        "",
        f"## Наборы отчёта: сети + дооптимизация ({k_star} шагов)",
        "",
        *head(("сеть, мс", "всего, с")),
    ]
    for m in ranked:
        total = f"{net_time(m, k_star):.2f}" if timing else "—"
        lines.append(table_row(m, key(m, k_star), (ms(m), total)))
    lines += ["", "## Сети без дооптимизации", "", *head(("сеть, мс",))]
    for m in sorted(nets, key=lambda m: -S(f"{m}|0")):
        lines.append(table_row(m, f"{m}|0", (ms(m),)))
    lines += ["", "## Итерационные методы и контроль", "", *head()]
    keys = [k for k in runs if k.startswith(("gs|", "gs_seq|"))]
    if v2:
        keys += [k for k in runs if k.startswith("adam|ft|")]
    keys += [f"adam|{'|'.join(adam_lr)}|{k}" for k in (100, 300, 1000)]
    keys += ["superposition", "control"]
    for k in keys:
        lines.append(table_row(k, k))
    lines += [
        "",
        "## Первый проход: дооптимизация одним критерием, 100 шагов",
        "",
        "| критерий | вес SSIM | lr | SSIM, среднее по сетям |",
        "|---|---|---|---|",
    ]
    for c, v in sorted(cfg_score.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {c[0]} | {c[1]} | {c[2]} | {v:.4f} |")
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
