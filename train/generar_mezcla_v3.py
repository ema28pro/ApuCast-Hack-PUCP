"""Export a fixed equal blend of the uploaded v2 and the trained ResNet v3."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ["id", "RH", "T", "U", "V"]


def read_verified(path, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest["sha256"] != digest or manifest["archivo"] != path.name:
        raise ValueError(f"El CSV y su registro no coinciden: {path}")
    return pd.read_csv(path), digest, manifest


def check(frame, template):
    if (frame.shape != (646_800, 5) or frame.columns.tolist() != COLUMNS
            or not frame.id.equals(template.id) or not frame.id.is_unique):
        raise ValueError("Formato o IDs diferentes de la plantilla oficial")
    if not np.isfinite(frame[COLUMNS[1:]].to_numpy()).all() or not frame.RH.between(0, 100).all():
        raise ValueError("Predicciones no finitas o humedad fuera de 0–100 %")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--resnet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".json")
    if args.output.exists() or manifest_path.exists():
        parser.error("El archivo de destino ya existe")
    data = args.data or Path(__file__).resolve().parents[1] / "data"
    official = data / "sample_submission.csv"
    template = pd.read_csv(official, usecols=["id"])
    v2, hash_v2, _ = read_verified(args.v2, args.v2_manifest)
    resnet, hash_resnet, candidate_settings = read_verified(args.resnet, args.resnet.with_suffix(".json"))
    if candidate_settings.get("prueba_rapida") or candidate_settings.get("semillas") != [0, 1, 2]:
        raise ValueError("Se requieren las tres semillas completas de ResNet")
    check(v2, template)
    check(resnet, template)
    predictions = 0.5 * v2[COLUMNS[1:]] + 0.5 * resnet[COLUMNS[1:]]
    predictions["RH"] = predictions.RH.clip(0, 100)
    result = predictions.round(3)
    result.insert(0, "id", template.id)
    temporary = args.output.with_suffix(".tmp")
    result.to_csv(temporary, index=False)
    check(pd.read_csv(temporary), template)
    manifest = {
        "archivo": args.output.name, "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
        "fuentes": [
            {"archivo": args.v2.name, "sha256": hash_v2, "peso": 0.5},
            {"archivo": args.resnet.name, "sha256": hash_resnet, "peso": 0.5},
        ],
        "score_publico": None,
        "descripcion": "ApuCast v3 | Mezcla fija 50/50 v2 y ResNet | RH 0–100",
    }
    temporary.replace(args.output)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CSV verificado: {args.output.resolve()} | SHA-256 {manifest['sha256']}")


if __name__ == "__main__":
    main()
