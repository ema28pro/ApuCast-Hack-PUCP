"""Compare fixed and calibrated ApuCast blends on chronological holdouts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generar_mezcla_v3 import check, read_verified
from generar_submission_v3 import SEEDS, VARIABLES, predict, train_seed
from runtime_apucast import load_starter


CUTS = (
    ("2024050900", "2024051600", "2024060100"),
    ("2024052500", "2024060100", "2024061600"),
    ("2024060900", "2024061600", "2024070100"),
)
NORMALIZERS = np.array([25.36, 2.0652, 2.6776, 2.0876])
EPOCHS = 25


def partitions(ns, cut):
    calibration, start, end = cut
    times = ns["ttr"]
    hours = ns["hour"](times)
    sampled = hours % 3 == 0
    train = times < calibration
    calibrate = (times >= calibration) & (times < start) & sampled
    report = (times >= start) & (times < end) & sampled
    assert train.any() and calibrate.any() and report.any()
    return train, calibrate, report, hours


def run_worker(args, ns):
    jobs = [(variant, seed) for variant in ("v2", "resnet") for seed in SEEDS]
    for cut in CUTS:
        train, calibrate, report, hours = partitions(ns, cut)
        evaluation = calibrate | report
        coarse = ns["Xtr"][train]
        climate = ns["climatology"](ns["Ytr"][train], ns["ttr"][train])
        mu, sd = coarse.mean(axis=(0, 1, 2)), coarse.std(axis=(0, 1, 2))
        anomalies = ns["Ytr"][train] - climate[hours[train]]
        scale = anomalies.std(axis=(0, 1, 2))
        tensor = ns["to_t"]
        x_train = tensor((ns["Wtr"][train] - mu) / sd)
        y_train = tensor(anomalies / scale)
        h_train = tensor(hours[train])
        x_eval = tensor((ns["Wtr"][evaluation] - mu) / sd)
        h_eval = tensor(hours[evaluation])
        climate_eval = tensor(climate[hours[evaluation]])
        y_scale = tensor(scale)
        folder = args.output / cut[1]
        folder.mkdir(exist_ok=True, parents=True)
        for variant, seed in jobs[args.worker::2]:
            prefix = folder / f"{variant}_{seed}"
            path = prefix.with_suffix(".npy")
            if path.is_file():
                continue
            print(f"{cut[1]} | {variant} | semilla {seed}", flush=True)
            model = train_seed(
                ns, seed, EPOCHS, x_train, h_train, y_train,
                prefix.with_suffix(".pt"), variant=variant,
            )
            result = predict(model, x_eval, h_eval, climate_eval, y_scale)
            if not np.isfinite(result).all():
                raise ValueError(f"Predicciones no finitas en {prefix}")
            with prefix.with_suffix(".tmp").open("wb") as stream:
                np.save(stream, result, allow_pickle=False)
            prefix.with_suffix(".tmp").replace(path)
            del model


def calibration_sums(cnn, resnet, truth):
    difference = resnet.astype(np.float64) - cnn
    residual = truth.astype(np.float64) - cnn
    return (difference * residual).sum(axis=(0, 1, 2)), np.square(difference).sum(axis=(0, 1, 2))


def resnet_weights(numerator, denominator):
    # Ridge penalty equivalent to shrinking the least-squares weight 20% toward 0.5.
    estimate = np.divide(numerator, denominator, out=np.full(4, 0.5), where=denominator > 0)
    return np.clip(0.8 * estimate + 0.1, 0, 1)


def metrics(prediction, truth):
    rmse = np.sqrt(np.square(prediction - truth, dtype=np.float64).mean(axis=(0, 1, 2)))
    return {"score_proxy": float(1 - (rmse / NORMALIZERS).mean()),
            "rmse": dict(zip(VARIABLES, rmse.tolist()))}


def evaluate(cnn, resnet, calibration, truth):
    cnn, resnet = cnn.copy(), resnet.copy()
    cnn[..., 0] = cnn[..., 0].clip(0, 100)
    resnet[..., 0] = resnet[..., 0].clip(0, 100)
    count = len(calibration)
    numerator, denominator = calibration_sums(cnn[:count], resnet[:count], calibration)
    weights = resnet_weights(numerator, denominator)
    cnn, resnet = cnn[count:], resnet[count:]
    fixed = metrics(0.5 * (cnn + resnet), truth)
    adaptive = metrics((1 - weights) * cnn + weights * resnet, truth)
    return {
        "cnn": metrics(cnn, truth),
        "resnet": metrics(resnet, truth),
        "mezcla_50_50": fixed,
        "mezcla_calibrada": adaptive,
        "delta_calibrada": adaptive["score_proxy"] - fixed["score_proxy"],
        "pesos_resnet": dict(zip(VARIABLES, weights.tolist())),
    }, numerator, denominator


def summarize(args, ns):
    results = []
    rows = []
    cal_numerator, cal_denominator = np.zeros(4), np.zeros(4)
    methods = ("cnn", "resnet", "mezcla_50_50", "mezcla_calibrada")
    for cut in CUTS:
        train, calibrate, report, _ = partitions(ns, cut)
        calibration, truth = ns["Ytr"][calibrate], ns["Ytr"][report]
        folder = args.output / cut[1]
        cnn_total = resnet_total = 0
        scores = []
        for seed in SEEDS:
            cnn = np.load(folder / f"v2_{seed}.npy", allow_pickle=False)
            resnet = np.load(folder / f"resnet_{seed}.npy", allow_pickle=False)
            cnn_total = cnn_total + cnn
            resnet_total = resnet_total + resnet
            score, _, _ = evaluate(cnn, resnet, calibration, truth)
            scores.append({"semilla": seed, **score})
        ensemble, numerator, denominator = evaluate(
            cnn_total / len(SEEDS), resnet_total / len(SEEDS), calibration, truth,
        )
        cal_numerator += numerator
        cal_denominator += denominator
        scores.append({"semilla": "ensamble", **ensemble})
        for score in scores:
            rows.append({"inicio_utc": cut[1], "fin_exclusivo_utc": cut[2],
                         "semilla": score["semilla"],
                         **{key: score[key]["score_proxy"] for key in methods},
                         "delta_calibrada": score["delta_calibrada"]})
        results.append({
            "inicio_calibracion_utc": cut[0], "inicio_evaluacion_utc": cut[1],
            "fin_exclusivo_utc": cut[2], "horas_entrenamiento": int(train.sum()),
            "horas_calibracion": len(calibration), "horas_evaluacion": len(truth),
            "scores": scores,
        })
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "scores_temporadas_semillas.csv", index=False)
    ensemble_mean = table.loc[table.semilla == "ensamble", [*methods, "delta_calibrada"]].mean()
    single_seed_mean = table.loc[table.semilla != "ensamble", [*methods, "delta_calibrada"]].mean()
    weights = resnet_weights(cal_numerator, cal_denominator)
    summary = {
        "metrica": "1 - promedio(RMSE / denominador), horas UTC múltiplos de 3",
        "denominadores": dict(zip(VARIABLES, NORMALIZERS.tolist())),
        "epocas": EPOCHS, "semillas": list(SEEDS),
        "media_por_corte_del_ensamble": ensemble_mean.to_dict(),
        "media_de_las_nueve_semillas_por_corte": single_seed_mean.to_dict(),
        "pesos_resnet_de_calibraciones_combinadas": dict(zip(VARIABLES, weights.tolist())),
        "cortes": results,
    }
    (args.output / "backtest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print("Promedio por corte del ensamble:", ensemble_mean.to_dict(), flush=True)

    if args.predictions is not None:
        export_calibrated(args.data, args.predictions, weights, args.output / "submission_calibrada.csv")


def export_calibrated(data, predictions, weights, submission):
    v2_path = predictions / "v2_top.csv"
    resnet_path = predictions / "resnet.csv"
    v2, v2_sha, _ = read_verified(v2_path, v2_path.with_suffix(".json"))
    resnet, resnet_sha, resnet_manifest = read_verified(resnet_path, resnet_path.with_suffix(".json"))
    if resnet_manifest.get("semillas") != list(SEEDS) or resnet_manifest.get("prueba_rapida"):
        raise ValueError("Se requieren los tres modelos ResNet del entrenamiento completo.")
    template = pd.read_csv(data / "sample_submission.csv", usecols=["id"])
    check(v2, template)
    check(resnet, template)
    result = ((1 - weights) * v2[VARIABLES] + weights * resnet[VARIABLES]).round(3)
    result.insert(0, "id", template.id)
    temporary = submission.with_suffix(".tmp")
    result.to_csv(temporary, index=False)
    check(pd.read_csv(temporary), template)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(submission)
    manifest = {"archivo": submission.name, "sha256": digest,
                "filas": len(template), "score_publico": None,
                "fuentes_sha256": {"v2": v2_sha, "resnet": resnet_sha},
                "pesos_resnet": dict(zip(VARIABLES, weights.tolist()))}
    submission.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{submission.name}: {len(template)} filas, SHA-256 {digest}", flush=True)
    return digest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=[0, 1])
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    ns = load_starter(Path(__file__).resolve().parents[1], args.data)
    if args.worker is None:
        summarize(args, ns)
    else:
        run_worker(args, ns)


if __name__ == "__main__":
    main()
