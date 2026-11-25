"""Temporal validation of a local ridge correction and the CNN ensemble."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def load_starter(repo):
    notebook = json.loads((repo / "train/starter_notebook.ipynb").read_text(encoding="utf-8"))
    ns = {"repo": repo}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "fig, ax = " in source:
            continue
        if "tr = np.load" in source:
            source = source.replace("tr = np.load", "def find(name): return repo / 'data' / name\n\ntr = np.load")
        if "\nmodel = Net()" in source:
            exec(source.split("\nmodel = Net()", 1)[0], ns)
            return ns
        exec(source, ns)
    raise RuntimeError("No se encontró la definición de la CNN")


def make_features(ns):
    # Shared synoptic pattern plus local forecast, causal change and hour harmonics.
    windows = ns["Wtr"]
    raw = np.concatenate([windows[:, -1], windows[:, -1] - windows[:, 0]], axis=-1)
    grid = ns["GRID"]
    interpolated = []
    for batch in np.array_split(raw, 24):
        x = ns["to_t"](batch).permute(0, 3, 1, 2)
        fine = F.grid_sample(x, grid.expand(len(batch), -1, -1, -1),
                             align_corners=True, padding_mode="border")
        interpolated.append(fine.flatten(2).cpu().numpy())
    local = np.concatenate(interpolated).transpose(2, 0, 1)
    hour = ns["hour"](ns["ttr"]) * (2 * np.pi / 24)
    regional = raw.mean(axis=(1, 2))
    shared = np.column_stack([regional, np.sin(hour), np.cos(hour)]).astype(np.float32)
    return local, shared


def ridge_prediction(features, y, hours, train, test, clim):
    local, shared = features
    fit = np.flatnonzero(train)
    evaluate = np.flatnonzero(test)
    targets = (y[fit] - clim[hours[fit]]).reshape(len(fit), -1, 4)
    result = np.empty((len(evaluate), local.shape[0], 4), dtype=np.float32)
    for start in range(0, local.shape[0], 96):
        tile = slice(start, start + 96)
        current = local[tile]
        fit_local = current[:, fit]
        eval_local = current[:, evaluate]
        x = np.concatenate([fit_local, np.broadcast_to(shared[fit], (*fit_local.shape[:2], shared.shape[-1]))], axis=-1)
        z = np.concatenate([eval_local, np.broadcast_to(shared[evaluate], (*eval_local.shape[:2], shared.shape[-1]))], axis=-1)
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True).clip(min=1e-3)
        x = (x - mean) / std
        z = (z - mean) / std
        xt = x.swapaxes(1, 2)
        coefficients = np.linalg.solve(
            xt @ x + np.eye(x.shape[-1], dtype=np.float32) * (0.05 * len(fit)),
            xt @ targets[:, tile].transpose(1, 0, 2),
        )
        result[:, tile] = (z @ coefficients).transpose(1, 0, 2)
    return result.reshape(len(evaluate), *y.shape[1:]) + clim[hours[evaluate]]


def score(errors, reference):
    return float(1 - np.sqrt(errors / reference).mean())


def error_by_day(predicted, truth, days):
    error = np.square(predicted - truth).mean(axis=(1, 2), dtype=np.float64)
    dates, inverse = np.unique(days, return_inverse=True)
    sums = np.zeros((len(dates), 4))
    np.add.at(sums, inverse, error)
    return sums, np.bincount(inverse)


def paired_interval(model, baseline, reference):
    # Resample adjacent two-day blocks, keeping all cells and variables paired.
    rng = np.random.default_rng(42)
    n = len(model)
    starts = rng.integers(0, n - 1, size=(2000, (n + 1) // 2))
    draws = np.stack([starts, starts + 1], axis=-1).reshape(2000, -1)[:, :n]
    def skill(error):
        return 1 - np.sqrt(error[draws].sum(axis=1) / reference[draws].sum(axis=1)).mean(axis=1)
    return np.percentile(skill(model) - skill(baseline), [2.5, 97.5]).tolist()


def morning_metrics(pred, truth, times, dem, alert_threshold_c=0.0):
    local = pd.to_datetime(times, format="%Y%m%d%H") - pd.Timedelta(hours=5)
    mask = (dem >= 2000) & (dem <= 4000)
    count = int(mask.sum() * 0.1)
    overlaps, errors, observed_cold, hits, predicted_cold = [], [], [], [], []
    for date in local.normalize().unique():
        morning = (local.normalize() == date) & (local.hour <= 6)
        if morning.sum() != 7:
            continue
        observed = truth[morning, ..., 1][:, mask].min(axis=0)
        forecast = pred[morning, ..., 1][:, mask].min(axis=0)
        a = np.argsort(observed, kind="stable")[:count]
        b = np.argsort(forecast, kind="stable")[:count]
        overlaps.append(np.intersect1d(a, b).size / count)
        errors.append(np.abs(observed - forecast).mean())
        actual = observed < 273.15
        forecast_cold = forecast < (273.15 + alert_threshold_c)
        observed_cold.append(int(actual.sum()))
        hits.append(int((actual & forecast_cold).sum()))
        predicted_cold.append(int(forecast_cold.sum()))

    n_mornings = len(errors)
    if n_mornings == 0:
        return {"mornings": 0, "top_10_overlap": None, "top_10_overlap_ic95": None,
                "minimum_mae_c": None,
                "subzero_observed": 0, "subzero_predicted": 0, "subzero_hits": 0,
                "subzero_recall": None, "subzero_precision": None, "subzero_f1": None,
                "subzero_recall_ic95": None}
    recall = sum(hits) / sum(observed_cold) if sum(observed_cold) else None
    precision = sum(hits) / sum(predicted_cold) if sum(predicted_cold) else None
    f1 = (2 * recall * precision / (recall + precision)
          if recall is not None and precision is not None and recall + precision else 0.0)
    rng = np.random.default_rng(42)
    draws = rng.integers(n_mornings, size=(5000, n_mornings))
    overlap_interval = np.percentile(np.asarray(overlaps)[draws].mean(axis=1), [2.5, 97.5]).tolist()
    hit_draws = np.asarray(hits)[draws].sum(axis=1)
    cold_draws = np.asarray(observed_cold)[draws].sum(axis=1)
    valid = cold_draws > 0
    recall_interval = (np.percentile(hit_draws[valid] / cold_draws[valid], [2.5, 97.5]).tolist()
                       if valid.any() else None)
    return {"mornings": n_mornings, "top_10_overlap": float(np.mean(overlaps)),
            "top_10_overlap_ic95": overlap_interval,
            "minimum_mae_c": float(np.mean(errors)), "subzero_observed": sum(observed_cold),
            "subzero_predicted": sum(predicted_cold), "subzero_hits": sum(hits),
            "subzero_recall": recall, "subzero_precision": precision, "subzero_f1": f1,
            "alert_threshold_c": float(alert_threshold_c),
            "subzero_recall_ic95": recall_interval}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ns = load_starter(args.repo)
    truth, stamps = ns["Ytr"], ns["ttr"]
    hours = ns["hour"](stamps)
    features = make_features(ns)
    reports = {}
    for start, end in [("2024051600", "2024060100"), ("2024060100", "2024061600"),
                       ("2024061600", "2024070100")]:
        train = stamps < start
        test = (stamps >= start) & (stamps < end)
        clim = ns["climatology"](truth[train], stamps[train])
        reference = clim[hours[test]]
        ridge = ridge_prediction(features, truth, hours, train, test, clim)
        candidates = {"ridge": ridge}
        if start == "2024061600":
            predictions = []
            for seed in range(3):
                model = ns["Net"]().to(device=ns["dev"], memory_format=torch.channels_last)
                record = torch.load(args.checkpoints / f"starter_{seed}.pt",
                                    map_location=ns["dev"], weights_only=True)
                model.load_state_dict(record["weights"])
                predictions.append(ns["predict"](model, ns["Wtr"][test], stamps[test]))
            ensemble = np.mean(predictions, axis=0, dtype=np.float32)
            blend = 0.75 * ensemble + 0.25 * ridge
            humidity_only = ensemble.copy()
            humidity_only[..., 0] = blend[..., 0]
            bounded = humidity_only.copy()
            bounded[..., 0] = bounded[..., 0].clip(0, 100)
            candidates.update(ensemble=ensemble, blend=blend, humidity_only=humidity_only,
                              humidity_bounded=bounded)
        days = np.array([stamp[:8] for stamp in stamps[test]])
        baseline_error, counts = error_by_day(reference, truth[test], days)
        split_report = {}
        for label, prediction in candidates.items():
            assert prediction.shape == truth[test].shape and np.isfinite(prediction).all()
            error, _ = error_by_day(prediction, truth[test], days)
            split_report[label] = {
                "score": score(error.sum(axis=0), baseline_error.sum(axis=0)),
                "rmse": np.sqrt(error.sum(axis=0) / counts.sum()).tolist(),
                **morning_metrics(prediction, truth[test], stamps[test], ns["dem"]),
            }
            if label not in {"ensemble", "ridge"}:
                ensemble_error, _ = error_by_day(ensemble, truth[test], days)
                split_report[label]["delta_interval_95"] = paired_interval(error, ensemble_error, baseline_error)
        reports[f"{start}-{end}"] = split_report
        print(json.dumps(split_report, indent=2), flush=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
