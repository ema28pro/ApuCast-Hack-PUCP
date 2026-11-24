"""Diagnóstico descriptivo de los checkpoints del v2 sobre el corte histórico."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from validar_apucast import error_by_day, load_starter


def ratios(prediction, observed, climatology):
    pred_rmse = np.sqrt(np.square(prediction - observed).mean(axis=(0, 1), dtype=np.float64))
    clim_rmse = np.sqrt(np.square(climatology - observed).mean(axis=(0, 1), dtype=np.float64))
    if not np.all(clim_rmse > 0):
        raise ValueError("La climatología tiene RMSE cero en un grupo")
    return pred_rmse, clim_rmse, pred_rmse / clim_rmse


def bootstrap(v2, starter, truth, climatology, days):
    # A draw keeps all hours, cells and channels of a UTC day paired.
    v2_sum, _ = error_by_day(v2, truth, days)
    starter_sum, _ = error_by_day(starter, truth, days)
    clim_sum, _ = error_by_day(climatology, truth, days)
    indices = np.random.default_rng(42).integers(len(v2_sum), size=(5000, len(v2_sum)))
    denominator = clim_sum[indices].sum(axis=1)
    delta = (
        np.sqrt(starter_sum[indices].sum(axis=1) / denominator)
        - np.sqrt(v2_sum[indices].sum(axis=1) / denominator)
    )
    samples = np.column_stack([delta.mean(axis=1), delta])
    return np.percentile(samples, [2.5, 97.5], axis=0).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ns = load_starter(args.repo)
    val = ns["val"]
    times = ns["ttr"][val]
    local_hour = (ns["hour"](times) - 5) % 24
    truth = ns["Ytr"][val]
    climate = ns["clim_all"][ns["hour"](times)]
    models, predictions = [], []
    started = time.perf_counter()
    for seed in (0, 1, 2):
        model = ns["Net"]().to(device=ns["dev"], memory_format=torch.channels_last)
        state = torch.load(args.checkpoints / f"starter_{seed}.pt",
                           map_location=ns["dev"], weights_only=True)
        model.load_state_dict(state["weights"])
        models.append(model)
        predictions.append(ns["predict"](model, ns["Wtr"][val], times))
    inference = time.perf_counter() - started
    v2 = np.mean(predictions, axis=0, dtype=np.float32)
    starter = predictions[0]
    assert v2.shape == truth.shape and np.isfinite(v2).all()
    total = ns["local_score"](v2, truth)[0]
    assert abs(float(total) - 0.347986102104187) < 0.000001

    all_hours = np.ones(len(times), dtype=bool)
    dem = ns["dem"]
    all_cells = np.ones_like(dem, dtype=bool)
    bands = {"<2000": dem < 2000, "2000-4000": (dem >= 2000) & (dem <= 4000), ">4000": dem > 4000}
    hour_groups = {f"{hour:02d}:00": local_hour == hour for hour in range(24)}
    groups = [("total", "todo", all_hours, all_cells)]
    groups += [("hora_Lima", name, hours, all_cells) for name, hours in hour_groups.items()]
    groups += [("altitud_m", name, all_hours, cells) for name, cells in bands.items()]
    groups += [("hora_Lima_x_altitud_m", f"{hour}/{band}", hours, cells)
               for hour, hours in hour_groups.items() for band, cells in bands.items()]
    rows = []
    for kind, group, hours, cells in groups:
        observed = truth[hours][:, cells]
        reference = climate[hours][:, cells]
        rmse, clim_rmse, v2_ratio = ratios(v2[hours][:, cells], observed, reference)
        _, _, starter_ratio = ratios(starter[hours][:, cells], observed, reference)
        for variable, model_error, clim_error, base, current in zip(
            ["TOTAL", *ns["VARS"]], [None, *rmse], [None, *clim_rmse],
            [starter_ratio.mean(), *starter_ratio], [v2_ratio.mean(), *v2_ratio]
        ):
            rows.append({
                "dimension": kind, "grupo": group, "variable": variable,
                "horas": int(hours.sum()), "celdas": int(cells.sum()),
                "rmse_v2": "" if model_error is None else f"{model_error:.6f}",
                "rmse_climatologia": "" if clim_error is None else f"{clim_error:.6f}",
                "ratio_v2": f"{current:.6f}", "score_v2": f"{1-current:.6f}",
                "score_starter": f"{1-base:.6f}", "delta_vs_starter": f"{base-current:.6f}",
            })
    with (args.output / "diagnostico_v2.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    intervals = bootstrap(v2, starter, truth, climate, np.array([stamp[:8] for stamp in times]))
    params = sum(parameter.numel() for parameter in models[0].parameters())
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        forecasts = [ns["predict"](model, ns["Wtr"][val][:24], times[:24]) for model in models]
        np.mean(forecasts, axis=0, dtype=np.float32)
        timings.append(time.perf_counter() - started)
    summary = {
        "score_v2_reduccion_notebook_float32": f"{total:.6f}",
        "score_v2_reduccion_diagnostico_float64": rows[0]["score_v2"],
        "score_starter_reduccion_notebook_float32": f"{ns['local_score'](starter, truth)[0]:.6f}",
        "parametros_por_modelo": params,
        "parametros_ensamble": len(models) * params,
        "procesador_torch": ns["dev"],
        "segundos_inferencia_corte_incluida_carga": f"{inference:.6f}",
        "segundos_mediana_24h_sin_descarga": f"{np.median(timings):.6f}",
        "dias_utc": len(np.unique([stamp[:8] for stamp in times])),
        "ic95_vs_starter": {key: [f"{bound:.6f}" for bound in interval]
                            for key, interval in zip(["TOTAL", *ns["VARS"]], intervals)},
    }
    (args.output / "diagnostico_v2_resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
