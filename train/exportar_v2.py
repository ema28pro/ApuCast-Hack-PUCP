"""Export the frozen three-seed v2 with the official submission template."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from runtime_apucast import load_starter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("La ruta de salida ya existe")
    root = Path(__file__).resolve().parents[1]
    data = args.data or root / "data"
    ns = load_starter(root, data)
    model_dir = root / "models/v2-top"
    with np.load(model_dir / "preprocesamiento.npz") as preprocessor:
        for key, notebook_key in [("mu_x", "mu_x"), ("sd_x", "sd_x"),
                                  ("climatology", "clim_all"), ("anom_sd", "anom_sd")]:
            np.testing.assert_array_equal(preprocessor[key], ns[notebook_key])
    predictions = []
    for seed in (0, 1, 2):
        model = ns["Net"]().to(device=ns["dev"], memory_format=torch.channels_last)
        checkpoint = torch.load(model_dir / f"starter_{seed}.pt", map_location=ns["dev"], weights_only=True)
        model.load_state_dict(checkpoint["weights"])
        predictions.append(ns["predict"](model, ns["Wte"], ns["tte"]))
    forecast = np.mean(predictions, axis=0, dtype=np.float32)
    forecast[..., 0] = forecast[..., 0].clip(0, 100)
    sample = pd.read_csv(data / "sample_submission.csv")
    positions = {stamp: index for index, stamp in enumerate(ns["tte"])}
    ids = sample.id.str.split("_", expand=True)
    hours = ids[0].map(positions)
    assert hours.notna().all()
    values = forecast[hours, ids[1].astype(int), ids[2].astype(int)].round(3)
    result = pd.DataFrame(values, columns=["RH", "T", "U", "V"])
    result.insert(0, "id", sample.id)
    temporary = args.output.with_suffix(".tmp")
    result.to_csv(temporary, index=False)
    checked = pd.read_csv(temporary)
    assert checked.shape == (646_800, 5) and checked.columns.equals(sample.columns)
    assert checked.id.equals(sample.id) and checked.id.is_unique
    assert np.isfinite(checked.iloc[:, 1:].to_numpy()).all() and checked.RH.between(0, 100).all()
    temporary.replace(args.output)
    print(f"CSV validado ({len(checked)} filas). SHA-256 {hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
