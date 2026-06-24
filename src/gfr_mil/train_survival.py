from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .data import FeatureConfig, SurvivalBagDataset, make_survival_bins_from_frame, read_split, write_csv_rows
from .models import build_survival_model
from .survival import fit_survival
from .utils import default_device, ensure_dir, seed_everything


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GFR-MIL for slide prognosis.")
    parser.add_argument("--label_csv", required=True, help="CSV with slide_id, time and censorship columns.")
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--split_csv", default=None, help="CSV with train/val/test columns.")
    split_group.add_argument("--split_dir", default=None, help="Directory containing splits_{fold}.csv files.")
    parser.add_argument("--feature_dir", required=True, help="Low-scale h5 feature directory.")
    parser.add_argument("--high_feature_dir", required=True, help="High-scale h5 feature directory.")
    parser.add_argument("--mapping_dir", required=True, help="High-to-low patch mapping directory.")
    parser.add_argument("--output_dir", default="results/survival_run")
    parser.add_argument("--slide_col", default="slide_id")
    parser.add_argument("--time_col", required=True, help="Survival/event time column.")
    parser.add_argument("--censor_col", required=True, help="0=event observed, 1=censored.")
    parser.add_argument("--n_bins", type=int, default=4)
    parser.add_argument(
        "--bin_strategy",
        choices=["panther", "rrt"],
        default="panther",
        help="panther: per-fold train uncensored bins; rrt: all-CSV uncensored bins before splitting.",
    )
    parser.add_argument("--input_dim", type=int, required=True)
    parser.add_argument("--high_input_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--alpha_surv", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Fold ids used with --split_dir.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu.")
    return parser


def _split_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.split_csv:
        return [("fold_0", Path(args.split_csv))]

    split_dir = Path(args.split_dir)
    if args.folds is not None:
        return [(f"fold_{fold}", split_dir / f"splits_{fold}.csv") for fold in args.folds]

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


def _bin_rows(
    args: argparse.Namespace,
    all_data: pd.DataFrame,
    train_ids: list[str],
) -> pd.DataFrame:
    if args.bin_strategy == "rrt":
        return all_data
    return all_data[all_data[args.slide_col].astype(str).isin(set(train_ids))].copy()


def train_fold(
    args: argparse.Namespace,
    split_csv: Path,
    fold_name: str,
    feature_config: FeatureConfig,
    all_data: pd.DataFrame,
    rrt_bin_edges,
    device: torch.device,
) -> dict[str, object]:
    train_ids = read_split(split_csv, "train")
    val_ids = read_split(split_csv, "val")
    test_ids = read_split(split_csv, "test")

    bin_edges = (
        rrt_bin_edges
        if args.bin_strategy == "rrt"
        else make_survival_bins_from_frame(
            _bin_rows(args, all_data, train_ids),
            args.time_col,
            args.censor_col,
            args.n_bins,
            range_frame=all_data,
        )
    )

    dataset_kwargs = {
        "label_csv": args.label_csv,
        "feature_config": feature_config,
        "slide_col": args.slide_col,
        "time_col": args.time_col,
        "censor_col": args.censor_col,
        "n_bins": args.n_bins,
        "bin_edges": bin_edges,
    }
    train_dataset = SurvivalBagDataset(slide_ids=train_ids, **dataset_kwargs)
    val_dataset = SurvivalBagDataset(slide_ids=val_ids, **dataset_kwargs)
    test_dataset = SurvivalBagDataset(slide_ids=test_ids, **dataset_kwargs) if test_ids else None

    model = build_survival_model(
        input_dim=args.input_dim,
        high_input_dim=args.high_input_dim,
        n_bins=train_dataset.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    fold_output_dir = ensure_dir(Path(args.output_dir) / fold_name if args.split_dir else args.output_dir)
    print(f"\n=== {fold_name} | split: {split_csv} ===")
    print(f"device: {device}")
    print(f"train/val/test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset) if test_dataset else 0}")
    print(f"bin strategy: {args.bin_strategy}")
    print(f"survival bins: {train_dataset.bin_edges.tolist()}")
    summary = fit_survival(
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
        num_workers=args.num_workers,
        alpha=args.alpha_surv,
    )
    summary["fold"] = fold_name
    summary["bin_strategy"] = args.bin_strategy
    print("summary:", summary)
    return summary


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device) if args.device else default_device()

    feature_config = FeatureConfig(
        feature_dir=Path(args.feature_dir),
        high_feature_dir=Path(args.high_feature_dir),
        mapping_dir=Path(args.mapping_dir),
    )
    all_data = pd.read_csv(args.label_csv)
    rrt_bin_edges = (
        make_survival_bins_from_frame(all_data, args.time_col, args.censor_col, args.n_bins)
        if args.bin_strategy == "rrt"
        else None
    )

    summaries = [
        train_fold(args, split_csv, fold_name, feature_config, all_data, rrt_bin_edges, device)
        for fold_name, split_csv in _split_paths(args)
    ]
    if len(summaries) > 1:
        write_csv_rows(output_dir / "summary_folds.csv", summaries)
        write_csv_rows(output_dir / "summary_mean.csv", _summarize_folds(summaries))


if __name__ == "__main__":
    main()
