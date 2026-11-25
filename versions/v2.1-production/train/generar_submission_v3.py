"""Train the ResNet candidate on all official train labels and export a submission."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from arquitectura_resnet import make_model
from runtime_apucast import load_starter


SEEDS = (0, 1, 2)
VARIABLES = ["RH", "T", "U", "V"]


def train_seed(ns, seed, epochs, windows, hours, targets, destination):
    torch.manual_seed(seed)
    model = make_model(ns, "resnet")
    if destination.exists():
        checkpoint = torch.load(destination, map_location=ns["dev"], weights_only=True)
        assert checkpoint["epochs"] == epochs and checkpoint["seed"] == seed
        model.load_state_dict(checkpoint["weights"])
        return model
    rng = np.random.RandomState(seed)
    movement = np.arange(len(windows))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, total_steps=epochs * ((len(movement) + 31) // 32)
    )
    started = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        rng.shuffle(movement)
        for offset in range(0, len(movement), 32):
            batch = movement[offset:offset + 32]
            prediction = model(windows[batch], hours[batch])
            loss = F.mse_loss(prediction, targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        if not torch.isfinite(loss):
            raise ValueError(f"Pérdida no finita. Semilla {seed}, época {epoch}")
        print(f"Semilla {seed}, época {epoch:02d}/{epochs}. Último MSE {loss.item():.6f}", flush=True)
    checkpoint = {"weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "seed": seed, "epochs": epochs, "seconds": time.perf_counter() - started}
    temporary = destination.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)
    print(f"Semilla {seed} guardada. Tiempo {checkpoint['seconds']:.6f} s.", flush=True)
    return model


@torch.inference_mode()
def predict(model, windows, hours, reference, scale):
    model.eval()
    result = []
    for start in range(0, len(windows), 64):
        batch = slice(start, start + 64)
        correction = model(windows[batch], hours[batch])
        result.append((correction * scale + reference[batch]).cpu().numpy())
    return np.concatenate(result)


def save_submission(template, forecast, stamps, destination):
    positions = {stamp: index for index, stamp in enumerate(stamps)}
    parts = template.id.str.split("_", expand=True)
    time_indices = parts[0].map(positions)
    row, col = parts[1].astype(int), parts[2].astype(int)
    assert time_indices.notna().all()
    assert row.between(0, forecast.shape[1] - 1).all() and col.between(0, forecast.shape[2] - 1).all()
    values = forecast[time_indices, row, col]
    frame = pd.DataFrame(values.round(3), columns=VARIABLES)
    frame.insert(0, "id", template.id)
    temporary = destination.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    checked = pd.read_csv(temporary)
    assert checked.shape == (646_800, 5)
    assert checked.columns.tolist() == template.columns.tolist() == ["id", *VARIABLES]
    assert checked.id.is_unique and checked.id.equals(template.id)
    assert np.isfinite(checked[VARIABLES].to_numpy()).all() and checked.RH.between(0, 100).all()
    temporary.replace(destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds, epochs = ((0,), 1) if args.smoke else (SEEDS, args.epochs)
    if epochs < 1 or (args.smoke and args.output.name == "submission_v3_resnet.csv"):
        parser.error("Usa un nombre distinto para el smoke y un número positivo de épocas.")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Elige un nombre nuevo para el CSV y su registro")
    repo = Path(__file__).resolve().parents[1]
    args.work.mkdir(parents=True, exist_ok=True)
    protocol = {"modelo": "resnet", "semillas": seeds, "epocas_fijas": epochs,
                "seleccion_con_test": False, "prueba_rapida": args.smoke}
    path = args.work / "protocolo.json"
    serialized = json.dumps(protocol, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        parser.error("El directorio de trabajo contiene otro protocolo")
    path.write_text(serialized, encoding="utf-8")
    data = args.data or repo / "data"
    ns = load_starter(repo, data)
    climatology = ns["climatology"](ns["Ytr"], ns["ttr"])
    mu, sd = ns["Xtr"].mean(axis=(0, 1, 2)), ns["Xtr"].std(axis=(0, 1, 2))
    anomalies = ns["Ytr"] - climatology[ns["hour"](ns["ttr"])]
    scale = anomalies.std(axis=(0, 1, 2))
    preprocessor = args.work / "preprocesamiento.npz"
    np.savez_compressed(preprocessor, mu=mu, sd=sd, scale=scale, climatology=climatology)
    tensor = ns["to_t"]
    train = tensor((ns["Wtr"] - mu) / sd)
    train_hours = tensor(ns["hour"](ns["ttr"]))
    targets = tensor(anomalies / scale)
    test = tensor((ns["Wte"] - mu) / sd)
    test_hours = ns["hour"](ns["tte"])
    reference = tensor(climatology[test_hours])
    model_paths = []
    predictions = []
    for seed in seeds:
        model_path = args.work / f"resnet_{seed}.pt"
        model = train_seed(ns, seed, epochs, train, train_hours, targets, model_path)
        predictions.append(predict(model, test, tensor(test_hours), reference, tensor(scale)))
        model_paths.append(model_path)
    forecast = np.mean(predictions, axis=0, dtype=np.float32)
    forecast[..., 0] = forecast[..., 0].clip(0, 100)
    assert np.isfinite(forecast).all()
    template = pd.read_csv(data / "sample_submission.csv", usecols=["id", *VARIABLES])
    fingerprint = save_submission(template, forecast, ns["tte"], args.output)
    audit = {**protocol, "archivo": args.output.name, "sha256": fingerprint,
             "filas": len(template), "inicio_train": str(ns["ttr"][0]), "fin_train": str(ns["ttr"][-1]),
             "score_publico": None, "descripcion": "ApuCast v3 ResNet | 3 semillas | train oficial completo | RH 0–100"}
    audit_path = args.output.with_suffix(".json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.smoke:
        bundle = args.output.with_name("apucast_v3_resnet_modelo.zip")
        with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
            for artifact in [*model_paths, preprocessor, audit_path]:
                archive.write(artifact, artifact.name)
        with ZipFile(bundle) as archive:
            assert archive.testzip() is None
    print(f"CSV verificado: {args.output.resolve()}. SHA-256 {fingerprint}", flush=True)


if __name__ == "__main__":
    main()
