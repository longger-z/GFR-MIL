from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import SurvivalBagDataset, write_csv_rows
from .engine import make_loader
from .utils import ensure_dir, to_device


class NLLSurvLoss(nn.Module):

    def __init__(self, alpha: float = 0.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        hazards: torch.Tensor,
        survival: torch.Tensor,
        label: torch.Tensor,
        censorship: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(label)
        y = label.view(batch_size, 1)
        c = censorship.view(batch_size, 1).float()
        survival_padded = torch.cat([torch.ones_like(c), survival], dim=1)

        uncensored_loss = -(1 - c) * (
            torch.log(torch.gather(survival_padded, 1, y).clamp(min=self.eps))
            + torch.log(torch.gather(hazards, 1, y).clamp(min=self.eps))
        )
        censored_loss = -c * torch.log(torch.gather(survival_padded, 1, y + 1).clamp(min=self.eps))
        loss = censored_loss + uncensored_loss
        loss = (1 - self.alpha) * loss + self.alpha * uncensored_loss
        return loss.mean()


def concordance_index(event_times: list[float], risks: list[float], censorships: list[float]) -> float:
    times = np.asarray(event_times, dtype=float)
    risk = np.asarray(risks, dtype=float)
    observed = np.asarray(censorships, dtype=int) == 0
    concordant = 0.0
    permissible = 0.0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] == times[j]:
                continue
            if times[i] < times[j] and observed[i]:
                permissible += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
            elif times[j] < times[i] and observed[j]:
                permissible += 1.0
                if risk[j] > risk[i]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / permissible) if permissible > 0 else float("nan")


def _forward(model: nn.Module, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch = to_device(batch, device)
    return model(
        features=batch["features"],
        coords=batch.get("coords"),
        high_features=batch.get("high_features"),
        high_coords=batch.get("high_coords"),
        region_labels=batch.get("region_labels"),
    )


def train_survival_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: NLLSurvLoss,
    device: torch.device,
    grad_accum: int = 1,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    risks: list[float] = []
    times: list[float] = []
    censorships: list[float] = []

    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch, device)
        output = model(
            features=batch["features"],
            coords=batch.get("coords"),
            high_features=batch.get("high_features"),
            high_coords=batch.get("high_coords"),
            region_labels=batch.get("region_labels"),
            label=batch.get("label"),
        )
        aux_loss = output.get("aux_loss", torch.zeros((), device=device))
        loss = loss_fn(output["hazards"], output["survival"], batch["label"], batch["censorship"]) + aux_loss
        (loss / grad_accum).backward()
        if step % grad_accum == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        risks.append(float(output["risk"].detach().cpu().item()))
        times.append(float(batch["event_time"].cpu().item()))
        censorships.append(float(batch["censorship"].cpu().item()))

    return {
        "loss": total_loss / max(len(loader), 1),
        "c_index": concordance_index(times, risks, censorships),
    }


@torch.no_grad()
def evaluate_survival(
    model: nn.Module,
    loader,
    loss_fn: NLLSurvLoss,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    risks: list[float] = []
    times: list[float] = []
    censorships: list[float] = []
    rows: list[dict[str, object]] = []

    for batch in loader:
        batch = to_device(batch, device)
        output = model(
            features=batch["features"],
            coords=batch.get("coords"),
            high_features=batch.get("high_features"),
            high_coords=batch.get("high_coords"),
            region_labels=batch.get("region_labels"),
            label=batch.get("label"),
        )
        loss = loss_fn(output["hazards"], output["survival"], batch["label"], batch["censorship"])
        risk = float(output["risk"].cpu().item())
        time = float(batch["event_time"].cpu().item())
        censorship = float(batch["censorship"].cpu().item())

        total_loss += loss.item()
        risks.append(risk)
        times.append(time)
        censorships.append(censorship)
        row = {
            "slide_id": batch["slide_id"][0],
            "disc_label": int(batch["label"].cpu().item()),
            "event_time": time,
            "censorship": int(censorship),
            "risk": risk,
        }
        hazards = output["hazards"].cpu().numpy()[0]
        row.update({f"hazard_{idx}": float(value) for idx, value in enumerate(hazards)})
        rows.append(row)

    return {
        "loss": total_loss / max(len(loader), 1),
        "c_index": concordance_index(times, risks, censorships),
    }, rows


def fit_survival(
    model: nn.Module,
    train_dataset: SurvivalBagDataset,
    val_dataset: SurvivalBagDataset,
    test_dataset: SurvivalBagDataset | None,
    output_dir: str | Path,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_accum: int = 1,
    patience: int = 10,
    num_workers: int = 4,
    alpha: float = 0.0,
) -> dict[str, float]:
    output_dir = ensure_dir(output_dir)
    model.to(device)
    loss_fn = NLLSurvLoss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_loader(train_dataset, True, False, num_workers)
    val_loader = make_loader(val_dataset, False, False, num_workers)
    test_loader = make_loader(test_dataset, False, False, num_workers) if test_dataset else None

    best_score = -np.inf
    best_epoch = -1
    best_state = deepcopy(model.state_dict())
    no_improve = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_survival_epoch(model, train_loader, optimizer, loss_fn, device, grad_accum)
        val_metrics, _ = evaluate_survival(model, val_loader, loss_fn, device)
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)

        score = val_metrics["c_index"]
        if np.isnan(score):
            score = -val_metrics["loss"]
        print(
            f"epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} "
            f"train c-index {train_metrics['c_index']:.4f} "
            f"val loss {val_metrics['loss']:.4f} val c-index {val_metrics['c_index']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, output_dir / "best.pt")
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    write_csv_rows(output_dir / "history.csv", [dict(row) for row in history])
    model.load_state_dict(best_state)

    val_metrics, val_predictions = evaluate_survival(model, val_loader, loss_fn, device)
    write_csv_rows(output_dir / "val_predictions.csv", val_predictions)
    summary = {f"val_{key}": value for key, value in val_metrics.items()}
    summary["best_epoch"] = float(best_epoch)

    if test_loader is not None:
        test_metrics, test_predictions = evaluate_survival(model, test_loader, loss_fn, device)
        write_csv_rows(output_dir / "test_predictions.csv", test_predictions)
        summary.update({f"test_{key}": value for key, value in test_metrics.items()})

    summary["n_bins"] = float(train_dataset.num_classes)
    write_csv_rows(output_dir / "summary.csv", [summary])
    write_csv_rows(
        output_dir / "survival_bins.csv",
        [{"bin": idx, "left": train_dataset.bin_edges[idx], "right": train_dataset.bin_edges[idx + 1]}
         for idx in range(train_dataset.num_classes)],
    )
    return summary
