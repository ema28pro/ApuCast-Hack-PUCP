"""Ablaciones v3 sobre los archivos oficiales de ApuCast."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from validar_apucast import error_by_day, load_starter, morning_metrics, paired_interval


# Cada corte reserva siete días previos para seleccionar checkpoints.
FOLDS = (
    ("2024050900", "2024051600", "2024060100"),
    ("2024052500", "2024060100", "2024061600"),
    ("2024060900", "2024061600", "2024070100"),
)
VARIANTS = ("v2", "resnet", "residuo_gfs")


class ResidualBlock(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.skip = nn.Conv2d(inputs, width, 1) if inputs != width else nn.Identity()

    def forward(self, x):
        return F.gelu(self.conv2(F.gelu(self.conv1(x))) + self.skip(x))


def make_model(ns, variant):
    model = ns["Net"]()
    if variant == "resnet":
        # Igual ancho, dos convoluciones espaciales por etapa y la misma cabeza.
        model.enc = ResidualBlock(14, 64)
        model.dec = nn.Sequential(ResidualBlock(73, 64), nn.Conv2d(64, 4, 1))
    return model.to(device=ns["dev"], memory_format=torch.channels_last)


def interpolate_gfs(ns, inputs):
    result = []
    for start in range(0, len(inputs), 128):
        x = ns["to_t"](inputs[start:start + 128]).permute(0, 3, 1, 2)
        fine = F.grid_sample(x, ns["GRID"].expand(len(x), -1, -1, -1),
                             align_corners=True, padding_mode="border")
        result.append(fine.permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(result)


class Fold:
    def __init__(self, ns, bounds):
        select, report, end = bounds
        times, targets = ns["ttr"], ns["Ytr"]
        self.train = np.flatnonzero(times < select)
        self.select = np.flatnonzero((times >= select) & (times < report))
        self.report = np.flatnonzero((times >= report) & (times < end))
        assert max(times[self.train]) < min(times[self.select])
        assert max(times[self.select]) < min(times[self.report])
        self.hours = ns["hour"](times)
        self.clim = ns["climatology"](targets[self.train], times[self.train])
        self.reference = self.clim[self.hours]
        coarse = ns["Xtr"][self.train]
        self.mu = coarse.mean(axis=(0, 1, 2))
        self.sd = coarse.std(axis=(0, 1, 2))
        self.scale = (targets[self.train] - self.reference[self.train]).std(axis=(0, 1, 2))
        assert np.all(self.sd > 0) and np.all(self.scale > 0)


def skill(prediction, truth, reference):
    axes = (0, 1, 2)
    model_mse = np.square(prediction - truth, dtype=np.float64).mean(axis=axes)
    clim_mse = np.square(reference - truth, dtype=np.float64).mean(axis=axes)
    ratio = np.sqrt(model_mse / clim_mse)
    return {"total": float(1 - ratio.mean()), "ratio": ratio.tolist(),
            "por_variable": (1 - ratio).tolist(), "rmse": np.sqrt(model_mse).tolist()}


def run_seed(ns, fold, variant, seed, epochs, save_path):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = make_model(ns, variant)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    base = ns["gfs_fine"] if variant == "residuo_gfs" else fold.reference
    # Escala común para aislar el efecto de la referencia de reconstrucción.
    scale = ns["to_t"](fold.scale)
    targets = ns["Ytr"]
    total_steps = epochs * ((len(fold.train) + 31) // 32)
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total_steps)

    def predict(indices, training=False):
        x = ns["to_t"]((ns["Wtr"][indices] - fold.mu) / fold.sd)
        h = ns["to_t"](fold.hours[indices])
        correction = model(x, h)
        if training:
            return correction
        return (correction * scale + ns["to_t"](base[indices])).cpu().numpy()

    @torch.inference_mode()
    def evaluate(indices):
        model.eval()
        predictions = np.concatenate([predict(indices[i:i + 64]) for i in range(0, len(indices), 64)])
        if not np.isfinite(predictions).all():
            raise ValueError("Predicción no finita")
        return predictions

    best = None
    started = time.perf_counter()
    order = fold.train.copy()
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            target = ns["to_t"](targets[batch] - base[batch]) / scale
            loss = F.mse_loss(predict(batch, training=True), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            schedule.step()
        indices = fold.select
        score = skill(evaluate(indices), targets[indices], fold.reference[indices])["total"]
        if best is None or score > best["score"]:
            best = {"weights": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch, "score": score, "seed": seed, "variant": variant}
        if epoch - best["epoch"] >= 5:
            break
    model.load_state_dict(best["weights"])
    best["seconds"] = time.perf_counter() - started
    torch.save(best, save_path)
    print(f"{save_path.stem} selección={best['score']:.6f} época={best['epoch']} "
          f"segundos={best['seconds']:.6f}", flush=True)
    return evaluate(fold.report), sum(p.numel() for p in model.parameters()), best["seconds"]


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--output", type=Path, required=True)
    args.add_argument("--smoke", action="store_true")
    config = args.parse_args()
    config.output.mkdir(parents=True, exist_ok=False)
    ns = load_starter(Path(__file__).resolve().parents[1])
    ns["gfs_fine"] = interpolate_gfs(ns, ns["Xtr"])
    seeds, epochs = ((0,), 1) if config.smoke else ((0, 1, 2), 25)
    reports = []
    for bounds in (FOLDS[-1:] if config.smoke else FOLDS):
        fold = Fold(ns, bounds)
        indices = fold.report
        truth, stamps = ns["Ytr"][indices], ns["ttr"][indices]
        reference = fold.reference[indices]
        days = np.array([stamp[:8] for stamp in stamps])
        reference_error, _ = error_by_day(reference, truth, days)
        np.savez_compressed(config.output / f"preprocesamiento_{bounds[1]}.npz",
                            mu=fold.mu, sd=fold.sd, scale=fold.scale, climatology=fold.clim)
        report = {"corte": bounds, "horas": [len(fold.train), len(fold.select), len(indices)],
                  "semillas": seeds, "variantes": {}}
        for variant in VARIANTS:
            runs = [run_seed(ns, fold, variant, seed, epochs,
                             config.output / f"{bounds[1]}_{variant}_{seed}.pt") for seed in seeds]
            prediction = np.mean([run[0] for run in runs], axis=0)
            error, _ = error_by_day(prediction, truth, days)
            values = skill(prediction, truth, reference)
            if variant == "v2":
                baseline_error, baseline_score = error, values["total"]
            bands = {"<2000": ns["dem"] < 2000, "2000-4000": (ns["dem"] >= 2000) & (ns["dem"] <= 4000),
                     ">4000": ns["dem"] > 4000}
            values.update(
                params=sum(run[1] for run in runs), segundos=sum(run[2] for run in runs),
                delta_vs_v2=values["total"] - baseline_score,
                ic95_vs_v2=paired_interval(error, baseline_error, reference_error),
                producto=morning_metrics(prediction, truth, stamps, ns["dem"]),
                por_altitud={name: skill(prediction[:, mask, None], truth[:, mask, None],
                                         reference[:, mask, None]) for name, mask in bands.items()},
                por_hora_lima={f"{hour:02d}": skill(prediction[mask], truth[mask], reference[mask])
                               for hour in range(24)
                               if (mask := (fold.hours[indices] - 5) % 24 == hour).any()},
            )
            report["variantes"][variant] = values
            print(f"{variant} reporte={values['total']:.6f} delta={values['delta_vs_v2']:+.6f}", flush=True)
        reports.append(report)
        (config.output / "resultados.json").write_text(
            json.dumps({"prueba_rapida": config.smoke, "cortes": reports}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
