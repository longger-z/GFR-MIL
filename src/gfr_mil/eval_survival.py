from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import (
    FeatureConfig,
    SurvivalBagDataset,
    read_csv_rows,
    read_split,
    write_csv_rows,
    make_survival_bins_from_frame,
)
from .engine import make_loader
from .models import build_survival_model
from .survival import NLLSurvLoss, evaluate_survival
from .utils import default_device, ensure_dir, seed_everything

SUBSET_CHOICES = ["train", "val", "test", "all"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained GFR-MIL prognosis model.")
    parser.add_argument("--label_csv", required=True, help="CSV with slide_id, time and censorship columns.")
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--split_csv", default=None, help="CSV with train/val/test columns.")
    split_group.add_argument("--split_dir", default=None, help="Directory containing splits_{fold}.csv files.")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", default=None, help="Path to one best.pt checkpoint.")
    checkpoint_group.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Directory containing best.pt, or fold_*/best.pt when --split_dir is used.",
    )
    parser.add_argument("--feature_dir", required=True, help="Low-scale h5 feature directory.")
    parser.add_argument("--high_feature_dir", required=True, help="High-scale h5 feature directory.")
    parser.add_argument("--mapping_dir", required=True, help="High-to-low patch mapping directory.")
    parser.add_argument("--output_dir", default="results/survival_eval")
    parser.add_argument("--subset", choices=SUBSET_CHOICES, default="test")
    parser.add_argument("--slide_col", default="slide_id")
    parser.add_argument("--time_col", required=True, help="Survival/event time column.")
    parser.add_argument("--censor_col", required=True, help="0=event observed, 1=censored.")
    parser.add_argument("--n_bins", type=int, default=4)
    parser.add_argument(
        "--bin_strategy",
        choices=["panther", "rrt"],
        default="panther",
        help="Used only when survival_bins.csv is not found beside the checkpoint.",
    )
    parser.add_argument("--input_dim", type=int, required=True)
    parser.add_argument("--high_input_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
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


def _subsets(name: str) -> list[str]:
    return ["train", "val", "test"] if name == "all" else [name]


def _checkpoint_path(args: argparse.Namespace, fold_name: str) -> Path:
    if args.checkpoint:
        if args.split_dir:
            raise ValueError("--checkpoint is only valid with --split_csv; use --checkpoint_dir for multi-fold eval")
        return Path(args.checkpoint)

    checkpoint_dir = Path(args.checkpoint_dir)
    return checkpoint_dir / fold_name / "best.pt" if args.split_dir else checkpoint_dir / "best.pt"


def _checkpoint_run_dir(args: argparse.Namespace, fold_name: str, checkpoint_path: Path) -> Path:
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        return checkpoint_dir / fold_name if args.split_dir else checkpoint_dir
    return checkpoint_path.parent


def _load_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


def _saved_bin_edges(path: Path) -> np.ndarray | None:
    bin_csv = path / "survival_bins.csv"
    if not bin_csv.exists():
        return None
    rows = sorted(read_csv_rows(bin_csv), key=lambda row: int(row["bin"]))
    if not rows:
        return None
    edges = [float(rows[0]["left"])]
    edges.extend(float(row["right"]) for row in rows)
    return np.asarray(edges, dtype=float)


def _bin_rows(
    args: argparse.Namespace,
    all_data: pd.DataFrame,
    train_ids: list[str],
) -> pd.DataFrame:
    if args.bin_strategy == "rrt":
        return all_data
    return all_data[all_data[args.slide_col].astype(str).isin(set(train_ids))].copy()


def _bin_edges(
    args: argparse.Namespace,
    all_data: pd.DataFrame,
    train_ids: list[str],
    run_dir: Path,
) -> np.ndarray:
    saved = _saved_bin_edges(run_dir)
    if saved is not None:
        return saved
    return make_survival_bins_from_frame(
        _bin_rows(args, all_data, train_ids),
        args.time_col,
        args.censor_col,
        args.n_bins,
        range_frame=all_data,
    )


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    groups = sorted({str(row.get("subset", "")) for row in rows})
    if len(groups) <= 1:
        return [_summarize_group(rows, subset=groups[0] if groups else None)]
    return [_summarize_group([row for row in rows if row.get("subset") == subset], subset=subset) for subset in groups]


def _summarize_group(rows: list[dict[str, object]], subset: str | None = None) -> dict[str, object]:
    summary: dict[str, object] = {"fold": "mean"}
    if subset:
        summary["subset"] = subset
    numeric_keys = [
        key for key in rows[0]
        if key not in {"fold", "subset"} and all(isinstance(row.get(key), (int, float)) for row in rows)
    ]
    for key in numeric_keys:
        values = torch.tensor([float(row[key]) for row in rows], dtype=torch.float64)
        summary[key] = float(values.mean().item())
        summary[f"{key}_std"] = float(values.std(unbiased=False).item())
    return summary


def evaluate_fold(
    args: argparse.Namespace,
    split_csv: Path,
    fold_name: str,
    feature_config: FeatureConfig,
    all_data: pd.DataFrame,
    device: torch.device,
) -> list[dict[str, object]]:
    checkpoint_path = _checkpoint_path(args, fold_name)
    run_dir = _checkpoint_run_dir(args, fold_name, checkpoint_path)
    split_ids = {subset: read_split(split_csv, subset) for subset in ["train", "val", "test"]}
    bin_edges = _bin_edges(args, all_data, split_ids["train"], run_dir)

    dataset_kwargs = {
        "label_csv": args.label_csv,
        "feature_config": feature_config,
        "slide_col": args.slide_col,
        "time_col": args.time_col,
        "censor_col": args.censor_col,
        "n_bins": len(bin_edges) - 1,
        "bin_edges": bin_edges,
    }
    model = build_survival_model(
        input_dim=args.input_dim,
        high_input_dim=args.high_input_dim,
        n_bins=len(bin_edges) - 1,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    _load_state(model, checkpoint_path, device)
    model.to(device)

    fold_output_dir = ensure_dir(Path(args.output_dir) / fold_name if args.split_dir else args.output_dir)
    write_csv_rows(
        fold_output_dir / "survival_bins.csv",
        [{"bin": idx, "left": bin_edges[idx], "right": bin_edges[idx + 1]} for idx in range(len(bin_edges) - 1)],
    )
    print(f"\n=== {fold_name} | checkpoint: {checkpoint_path} ===")
    print(f"device: {device}")
    print(f"survival bins: {bin_edges.tolist()}")

    summaries: list[dict[str, object]] = []
    loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    for subset in _subsets(args.subset):
        ids = split_ids[subset]
        if not ids:
            if args.subset == "all":
                continue
            raise ValueError(f"split '{subset}' is empty in {split_csv}")

        dataset = SurvivalBagDataset(slide_ids=ids, **dataset_kwargs)
        loader = make_loader(dataset, train=False, weighted_sample=False, num_workers=args.num_workers)
        metrics, predictions = evaluate_survival(model, loader, loss_fn, device)
        write_csv_rows(fold_output_dir / f"{subset}_predictions.csv", predictions)

        summary: dict[str, object] = {"fold": fold_name, "subset": subset}
        summary.update(metrics)
        summaries.append(summary)
        print(f"{subset}: {summary}")

    write_csv_rows(fold_output_dir / "summary.csv", summaries)
    return summaries


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

    summaries = [
        row
        for fold_name, split_csv in _split_paths(args)
        for row in evaluate_fold(args, split_csv, fold_name, feature_config, all_data, device)
    ]
    if len(summaries) > 1:
        write_csv_rows(output_dir / "summary_folds.csv", summaries)
        write_csv_rows(output_dir / "summary_mean.csv", _summarize(summaries))


if __name__ == "__main__":
    main()
