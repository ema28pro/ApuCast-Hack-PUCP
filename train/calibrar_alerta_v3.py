"""Calibrate frost-alert thresholds on a temporal selection window, then report later."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from entrenar_v3 import FOLDS, VARIANTS, Fold, interpolate_gfs, make_model
from validar_apucast import load_starter, morning_metrics


def predict(ns, fold, variant, checkpoint, indices):
    architecture = variant.removesuffix("_frost")
    base = ns["gfs_fine"] if architecture == "residuo_gfs" else fold.reference
    scale = ns["to_t"](fold.scale)
    model = make_model(ns, variant)
    record = torch.load(checkpoint, map_location=ns["dev"], weights_only=True)
    model.load_state_dict(record["weights"])
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), 64):
            batch = indices[start:start + 64]
            x = ns["to_t"]((ns["Wtr"][batch] - fold.mu) / fold.sd)
            hour = ns["to_t"](fold.hours[batch])
            correction = model(x, hour)
            output.append((correction * scale + ns["to_t"](base[batch])).cpu().numpy())
    return np.concatenate(output)


def f2(metrics):
    precision, recall = metrics["subzero_precision"], metrics["subzero_recall"]
    if precision is None or recall is None or (4 * precision + recall) == 0:
        return 0.0
    return 5 * precision * recall / (4 * precision + recall)


def calibrate(prediction, truth, times, dem):
    thresholds = np.arange(-1.0, 5.01, 0.5)
    candidates = []
    for threshold in thresholds:
        metrics = morning_metrics(prediction, truth, times, dem, alert_threshold_c=threshold)
        candidates.append((f2(metrics), threshold, metrics))
    return max(candidates, key=lambda item: (item[0], item[2]["subzero_recall"] or 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=2, choices=range(len(FOLDS)))
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    ns = load_starter(Path(__file__).resolve().parents[1])
    ns["gfs_fine"] = interpolate_gfs(ns, ns["Xtr"])
    bounds = FOLDS[args.fold]
    fold = Fold(ns, bounds)
    selection = fold.select
    report = fold.report
    rows = {}

    candidates = {"gfs_interpolado": (ns["gfs_fine"][selection], ns["gfs_fine"][report])}
    for variant in VARIANTS:
        select_runs, report_runs = [], []
        for seed in range(args.seeds):
            checkpoint = args.checkpoints / f"{bounds[1]}_{variant}_{seed}.pt"
            select_runs.append(predict(ns, fold, variant, checkpoint, selection))
            report_runs.append(predict(ns, fold, variant, checkpoint, report))
        candidates[variant] = (np.mean(select_runs, axis=0), np.mean(report_runs, axis=0))

    for name, (selection_prediction, report_prediction) in candidates.items():
        threshold_score, threshold, selected = calibrate(
            selection_prediction, ns["Ytr"][selection], ns["ttr"][selection], ns["dem"])
        evaluated = morning_metrics(report_prediction, ns["Ytr"][report], ns["ttr"][report],
                                   ns["dem"], alert_threshold_c=threshold)
        rows[name] = {"umbral_alerta_c": threshold, "f2_seleccion": threshold_score,
                      "seleccion": selected, "reporte_temporal": evaluated}
        print(f"{name}: umbral={threshold:+.1f} C, F2 selección={threshold_score:.3f}, "
              f"recobrado reporte={evaluated['subzero_hits']}/{evaluated['subzero_observed']} "
              f"({evaluated['subzero_recall']})", flush=True)

    result = {"corte": bounds, "semillas": args.seeds, "umbrales_candidatos_c": [-1, 5, 0.5],
              "variantes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
