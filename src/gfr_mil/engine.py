from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

from .data import SlideBagDataset, collate_bags, write_csv_rows
from .metrics import classification_metrics
from .utils import ensure_dir, to_device


def make_loader(
    dataset: SlideBagDataset,
    train: bool,
    weighted_sample: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    if train and weighted_sample:
        weights = []
        class_counts = [max(len(ids), 1) for ids in dataset.slide_cls_ids]
        for idx in range(len(dataset)):
            label = int(dataset.targets[idx])
            weights.append(1.0 / class_counts[label])
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
    elif train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bags,
    )


def _forward(model: nn.Module, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch = to_device(batch, device)
    return model(
        features=batch["features"],
        coords=batch.get("coords"),
        high_features=batch.get("high_features"),
        high_coords=batch.get("high_coords"),
        region_labels=batch.get("region_labels"),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_accum: int = 1,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    labels: list[int] = []
    preds: list[int] = []
    probs: list[list[float]] = []
    optimizer.zero_grad()

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
        loss = (loss_fn(output["logits"], batch["label"]) + aux_loss) / grad_accum
        loss.backward()
        if step % grad_accum == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        labels.append(int(batch["label"].item()))
        preds.append(int(output["preds"].item()))
        probs.append(output["probs"].detach().cpu().numpy()[0].tolist())

    metrics = classification_metrics(labels, probs, preds, loader.dataset.num_classes)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    preds: list[int] = []
    probs: list[list[float]] = []
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
        loss = loss_fn(output["logits"], batch["label"])
        prob = output["probs"].detach().cpu().numpy()[0]
        label = int(batch["label"].item())
        pred = int(output["preds"].item())

        total_loss += loss.item()
        labels.append(label)
        preds.append(pred)
        probs.append(prob.tolist())
        row = {"slide_id": batch["slide_id"][0], "label": label, "pred": pred}
        row.update({f"prob_{i}": float(prob[i]) for i in range(len(prob))})
        rows.append(row)

    metrics = classification_metrics(labels, probs, preds, loader.dataset.num_classes)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics, rows


def fit(
    model: nn.Module,
    train_dataset: SlideBagDataset,
    val_dataset: SlideBagDataset,
    test_dataset: SlideBagDataset | None,
    output_dir: str | Path,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_accum: int = 1,
    patience: int = 10,
    weighted_sample: bool = False,
    num_workers: int = 4,
) -> dict[str, float]:
    output_dir = ensure_dir(output_dir)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_loader(train_dataset, True, weighted_sample, num_workers)
    val_loader = make_loader(val_dataset, False, False, num_workers)
    test_loader = make_loader(test_dataset, False, False, num_workers) if test_dataset else None

    best_auc = -np.inf
    best_epoch = -1
    best_state = deepcopy(model.state_dict())
    no_improve = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, grad_accum)
        val_metrics, _ = evaluate(model, val_loader, loss_fn, device)
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        score = val_metrics["auc"]
        if np.isnan(score):
            score = val_metrics["f1_macro"]
        print(
            f"epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} "
            f"val loss {val_metrics['loss']:.4f} val auc {val_metrics['auc']:.4f} "
            f"val f1 {val_metrics['f1_macro']:.4f}"
        )

        if score > best_auc:
            best_auc = score
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

    val_metrics, val_predictions = evaluate(model, val_loader, loss_fn, device)
    write_csv_rows(output_dir / "val_predictions.csv", val_predictions)
    summary = {f"val_{k}": v for k, v in val_metrics.items()}
    summary["best_epoch"] = float(best_epoch)

    if test_loader is not None:
        test_metrics, test_predictions = evaluate(model, test_loader, loss_fn, device)
        write_csv_rows(output_dir / "test_predictions.csv", test_predictions)
        summary.update({f"test_{k}": v for k, v in test_metrics.items()})

    write_csv_rows(output_dir / "summary.csv", [summary])
    return summary
