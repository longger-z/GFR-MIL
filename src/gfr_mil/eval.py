from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import FeatureConfig, SlideBagDataset, parse_label_map, read_split, write_csv_rows
from .engine import evaluate, make_loader
from .models import build_model
from .utils import default_device, ensure_dir, seed_everything

SUBSET_CHOICES = ["train", "val", "test", "all"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained GFR-MIL classifier.")
    parser.add_argument("--label_csv", required=True, help="CSV with slide_id and label columns.")
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
    parser.add_argument("--output_dir", default="results/eval", help="Where evaluation CSVs are saved.")
    parser.add_argument("--subset", choices=SUBSET_CHOICES, default="test")
    parser.add_argument("--label_map", required=True, help='Example: "ADH:0,FEA:0,N:1,PB:1,UDH:1,DCIS:2,IC:2"')
    parser.add_argument("--slide_col", default="slide_id")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--input_dim", type=int, required=True)
    parser.add_argument("--high_input_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
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


def _load_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


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
    label_map: dict[str, int],
    device: torch.device,
) -> list[dict[str, object]]:
    checkpoint_path = _checkpoint_path(args, fold_name)
    split_ids = {subset: read_split(split_csv, subset) for subset in ["train", "val", "test"]}

    dataset_kwargs = {
        "label_csv": args.label_csv,
        "feature_config": feature_config,
        "label_map": label_map,
        "slide_col": args.slide_col,
        "label_col": args.label_col,
    }
    num_classes = len(set(label_map.values()))
    model = build_model(
        input_dim=args.input_dim,
        high_input_dim=args.high_input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    _load_state(model, checkpoint_path, device)
    model.to(device)

    fold_output_dir = ensure_dir(Path(args.output_dir) / fold_name if args.split_dir else args.output_dir)
    print(f"\n=== {fold_name} | checkpoint: {checkpoint_path} ===")
    print(f"device: {device}")

    summaries: list[dict[str, object]] = []
    for subset in _subsets(args.subset):
        ids = split_ids[subset]
        if not ids:
            if args.subset == "all":
                continue
            raise ValueError(f"split '{subset}' is empty in {split_csv}")

        dataset = SlideBagDataset(slide_ids=ids, **dataset_kwargs)
        loader = make_loader(dataset, train=False, weighted_sample=False, num_workers=args.num_workers)
        metrics, predictions = evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)
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

    label_map = parse_label_map(args.label_map)
    feature_config = FeatureConfig(
        feature_dir=Path(args.feature_dir),
        high_feature_dir=Path(args.high_feature_dir),
        mapping_dir=Path(args.mapping_dir),
    )

    summaries = [
        row
        for fold_name, split_csv in _split_paths(args)
        for row in evaluate_fold(args, split_csv, fold_name, feature_config, label_map, device)
    ]
    if len(summaries) > 1:
        write_csv_rows(output_dir / "summary_folds.csv", summaries)
        write_csv_rows(output_dir / "summary_mean.csv", _summarize(summaries))


if __name__ == "__main__":
    main()
