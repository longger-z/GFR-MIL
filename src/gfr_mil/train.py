from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import FeatureConfig, SlideBagDataset, parse_label_map, read_split, write_csv_rows
from .engine import fit
from .models import build_model
from .utils import default_device, ensure_dir, seed_everything


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GFR-MIL for slide classification.")
    parser.add_argument("--label_csv", required=True, help="CSV with slide_id and label columns.")
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--split_csv", default=None, help="CSV with train/val/test columns.")
    split_group.add_argument("--split_dir", default=None, help="Directory containing splits_{fold}.csv files.")
    parser.add_argument("--feature_dir", required=True, help="Low-scale h5 feature directory.")
    parser.add_argument("--high_feature_dir", required=True, help="High-scale h5 feature directory.")
    parser.add_argument("--mapping_dir", required=True, help="High-to-low patch mapping directory.")
    parser.add_argument("--output_dir", default="results/run", help="Where checkpoints and CSVs are saved.")
    parser.add_argument("--label_map", required=True, help='Example: "ADH:0,FEA:0,N:1,PB:1,UDH:1,DCIS:2,IC:2"')
    parser.add_argument("--slide_col", default="slide_id")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--input_dim", type=int, required=True)
    parser.add_argument("--high_input_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Fold ids used with --split_dir.")
    parser.add_argument("--weighted_sample", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu.")
    return parser


def _split_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.split_csv:
        return [("fold_0", Path(args.split_csv))]

    split_dir = Path(args.split_dir)
    if args.folds is not None:
        return [(f"fold_{fold}", split_dir / f"split_{fold}.csv") for fold in args.folds]

    split_paths = sorted(split_dir.glob("splits_*.csv"), key=lambda path: path.name)
    if not split_paths:
        raise FileNotFoundError(f"no splits_*.csv files found in {split_dir}")
    return [(f"fold_{path.stem.split('_')[-1]}", path) for path in split_paths]


def _summarize_folds(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_keys = [
        key for key in rows[0]
        if key != "fold" and all(isinstance(row.get(key), (int, float)) for row in rows)
    ]
    summary: dict[str, object] = {"fold": "mean"}
    for key in numeric_keys:
        values = torch.tensor([float(row[key]) for row in rows], dtype=torch.float64)
        summary[key] = float(values.mean().item())
        summary[f"{key}_std"] = float(values.std(unbiased=False).item())
    return [summary]


def train_fold(
    args: argparse.Namespace,
    split_csv: Path,
    fold_name: str,
    feature_config: FeatureConfig,
    label_map: dict[str, int],
    device: torch.device,
) -> dict[str, object]:
    train_ids = read_split(split_csv, "train")
    val_ids = read_split(split_csv, "val")
    test_ids = read_split(split_csv, "test")

    dataset_kwargs = {
        "label_csv": args.label_csv,
        "feature_config": feature_config,
        "label_map": label_map,
        "slide_col": args.slide_col,
        "label_col": args.label_col,
    }
    train_dataset = SlideBagDataset(slide_ids=train_ids, **dataset_kwargs)
    val_dataset = SlideBagDataset(slide_ids=val_ids, **dataset_kwargs)
    test_dataset = SlideBagDataset(slide_ids=test_ids, **dataset_kwargs) if test_ids else None

    model = build_model(
        input_dim=args.input_dim,
        high_input_dim=args.high_input_dim,
        num_classes=train_dataset.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    fold_output_dir = ensure_dir(Path(args.output_dir) / fold_name if args.split_dir else args.output_dir)
    print(f"\n=== {fold_name} | split: {split_csv} ===")
    print(f"device: {device}")
    print(f"train/val/test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset) if test_dataset else 0}")
    summary = fit(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        output_dir=fold_output_dir,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        patience=args.patience,
        weighted_sample=args.weighted_sample,
        num_workers=args.num_workers,
    )
    summary["fold"] = fold_name
    print("summary:", summary)
    return summary


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device) if args.device else default_device()

    label_map = parse_label_map(args.label_map)
    feature_config = FeatureConfig(
        feature_dir=Path(args.feature_dir),
        high_feature_dir=Path(args.high_feature_dir),
        mapping_dir=Path(args.mapping_dir),
    )

    summaries = [
        train_fold(args, split_csv, fold_name, feature_config, label_map, device)
        for fold_name, split_csv in _split_paths(args)
    ]
    if len(summaries) > 1:
        write_csv_rows(output_dir / "summary_folds.csv", summaries)
        write_csv_rows(output_dir / "summary_mean.csv", _summarize_folds(summaries))


if __name__ == "__main__":
    main()
