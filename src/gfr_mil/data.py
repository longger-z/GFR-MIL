from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FeatureConfig:
    feature_dir: Path
    high_feature_dir: Path | None = None
    mapping_dir: Path | None = None
    feature_key: str = "features"
    coord_key: str = "coords"


def parse_label_map(text: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split(":", maxsplit=1)
        labels[name.strip()] = int(value)
    if not labels:
        raise ValueError("label map is empty")
    return labels


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_split(split_csv: str | Path, split_name: str) -> list[str]:
    rows = read_csv_rows(split_csv)
    if rows and split_name not in rows[0]:
        raise ValueError(f"split csv must contain column '{split_name}'")
    return [row[split_name] for row in rows if row.get(split_name)]


def _h5_path(root: Path, slide_id: str) -> Path:
    direct = root / f"{slide_id}.h5"
    nested = root / "h5_files" / f"{slide_id}.h5"
    if direct.exists():
        return direct
    return nested


def _load_h5(path: Path, feature_key: str, coord_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"feature file not found: {path}")
    with h5py.File(path, "r") as h5_file:
        features = torch.from_numpy(h5_file[feature_key][:]).float()
        if coord_key in h5_file:
            coords = torch.from_numpy(h5_file[coord_key][:]).float()
        else:
            coords = torch.zeros((features.shape[0], 2), dtype=torch.float32)
    return features, coords


class SlideBagDataset(Dataset):
    def __init__(
        self,
        label_csv: str | Path,
        feature_config: FeatureConfig,
        label_map: dict[str, int],
        slide_ids: list[str] | None = None,
        slide_col: str = "slide_id",
        label_col: str = "label",
        case_col: str = "case_id",
    ) -> None:
        self.feature_config = feature_config
        self.slide_col = slide_col
        self.label_col = label_col
        self.case_col = case_col
        self.label_map = label_map
        self.num_classes = len(set(label_map.values()))

        rows = read_csv_rows(label_csv)
        if rows and (slide_col not in rows[0] or label_col not in rows[0]):
            raise ValueError(f"label csv must contain '{slide_col}' and '{label_col}'")

        slide_filter = set(slide_ids) if slide_ids is not None else None
        self.slide_data: list[dict[str, str | int]] = []
        for row in rows:
            if row.get(label_col) not in label_map:
                continue
            slide_id = row[slide_col]
            if slide_filter is not None and slide_id not in slide_filter:
                continue
            item: dict[str, str | int] = dict(row)
            item["target"] = label_map[row[label_col]]
            self.slide_data.append(item)

        if not self.slide_data:
            raise ValueError("dataset is empty after filtering labels/splits")

        self.targets = [int(row["target"]) for row in self.slide_data]
        self.slide_cls_ids = [
            np.where(np.asarray(self.targets) == class_id)[0]
            for class_id in range(self.num_classes)
        ]

    def __len__(self) -> int:
        return len(self.slide_data)

    def _feature_item(self, slide_id: str) -> dict[str, torch.Tensor | str]:
        cfg = self.feature_config
        if cfg.high_feature_dir is None or cfg.mapping_dir is None:
            raise ValueError("GFR-MIL requires both high_feature_dir and mapping_dir")

        low_features, low_coords = _load_h5(
            _h5_path(cfg.feature_dir, slide_id), cfg.feature_key, cfg.coord_key
        )
        item: dict[str, torch.Tensor | str] = {
            "features": low_features,
            "coords": low_coords,
            "slide_id": slide_id,
        }

        high_features, high_coords = _load_h5(
            _h5_path(cfg.high_feature_dir, slide_id), cfg.feature_key, cfg.coord_key
        )
        item["high_features"] = high_features
        item["high_coords"] = high_coords
        item["region_labels"] = self._load_region_labels(slide_id, high_coords, low_coords)

        return item

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.slide_data[index]
        slide_id = str(row[self.slide_col])
        label = int(row["target"])
        item = self._feature_item(slide_id)
        item["label"] = torch.tensor(label, dtype=torch.long)
        return item

    def _load_region_labels(
        self, slide_id: str, high_coords: torch.Tensor, low_coords: torch.Tensor
    ) -> torch.Tensor:
        mapping_path = self.feature_config.mapping_dir / slide_id / "8_448.csv"
        if not mapping_path.exists():
            raise FileNotFoundError(f"mapping file not found: {mapping_path}")

        rows = read_csv_rows(mapping_path)
        required = {"x", "y", "grid_x", "grid_y"}
        if not rows:
            raise ValueError(f"mapping csv is empty: {mapping_path}")
        if not required.issubset(rows[0]):
            raise ValueError(f"mapping csv missing columns {required}: {mapping_path}")

        group_min: dict[tuple[str, str], tuple[float, float]] = {}
        for row in rows:
            key = (row["grid_x"], row["grid_y"])
            coord = (float(row["x"]), float(row["y"]))
            if key not in group_min:
                group_min[key] = coord
            else:
                group_min[key] = (min(group_min[key][0], coord[0]), min(group_min[key][1], coord[1]))

        region_by_high_coord = {
            (float(row["x"]), float(row["y"])): group_min[(row["grid_x"], row["grid_y"])]
            for row in rows
        }
        low_lookup = {tuple(map(float, coord)): idx for idx, coord in enumerate(low_coords.cpu().numpy())}
        labels = [
            low_lookup.get(region_by_high_coord.get(tuple(map(float, coord)), (np.nan, np.nan)), -1)
            for coord in high_coords.cpu().numpy()
        ]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        if not torch.any(labels_tensor >= 0):
            raise ValueError(f"mapping did not match any high coordinates to low coordinates: {mapping_path}")
        return labels_tensor


def make_survival_bins(
    rows: list[dict[str, str]],
    time_col: str,
    censor_col: str,
    n_bins: int,
    eps: float = 1e-6,
) -> np.ndarray:
    frame = _survival_frame(rows, time_col, censor_col)
    return make_survival_bins_from_frame(frame, time_col, censor_col, n_bins, eps=eps)


def make_survival_bins_from_frame(
    frame: pd.DataFrame,
    time_col: str,
    censor_col: str,
    n_bins: int,
    eps: float = 1e-6,
    range_frame: pd.DataFrame | None = None,
) -> np.ndarray:
    frame = _valid_survival_frame(frame, time_col, censor_col)
    if frame.empty:
        raise ValueError("cannot make survival bins from an empty dataframe")
    range_frame = frame if range_frame is None else _valid_survival_frame(range_frame, time_col, censor_col)
    if range_frame.empty:
        raise ValueError("cannot set survival bin range from an empty dataframe")

    uncensored = frame[frame[censor_col] == 0]
    qcut_source = uncensored if len(uncensored) >= 2 else frame
    try:
        _, bins = pd.qcut(qcut_source[time_col], q=n_bins, retbins=True, labels=False)
    except ValueError:
        _, bins = pd.qcut(
            qcut_source[time_col],
            q=min(n_bins, qcut_source[time_col].nunique()),
            retbins=True,
            labels=False,
            duplicates="drop",
        )

    bins = np.asarray(bins, dtype=float)
    if len(bins) < 2:
        value = float(frame[time_col].iloc[0])
        bins = np.asarray([value - 1.0, value + 1.0], dtype=float)

    bins[0] = float(range_frame[time_col].min()) - eps
    bins[-1] = float(range_frame[time_col].max()) + eps
    return bins


def assign_survival_disc_labels(
    frame: pd.DataFrame,
    time_col: str,
    censor_col: str,
    bin_edges: list[float] | np.ndarray,
) -> pd.DataFrame:
    frame = _valid_survival_frame(frame, time_col, censor_col).copy()
    edges = np.asarray(bin_edges, dtype=float)
    labels = pd.cut(
        frame[time_col],
        bins=edges,
        labels=False,
        right=False,
        include_lowest=True,
    )
    if labels.isna().any():
        missing = frame.loc[labels.isna(), time_col].tolist()
        raise ValueError(f"survival times outside bin edges: {missing}")
    frame["disc_label"] = labels.astype(int)
    return frame


def _survival_frame(
    rows: list[dict[str, str]],
    time_col: str,
    censor_col: str,
) -> pd.DataFrame:
    return _valid_survival_frame(pd.DataFrame(rows), time_col, censor_col)


def _valid_survival_frame(
    frame: pd.DataFrame,
    time_col: str,
    censor_col: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    missing = [col for col in [time_col, censor_col] if col not in frame.columns]
    if missing:
        raise ValueError(f"survival csv missing columns: {missing}")

    frame = frame.copy()
    frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce")
    frame[censor_col] = pd.to_numeric(frame[censor_col], errors="coerce")
    frame = frame.dropna(subset=[time_col, censor_col])
    frame[censor_col] = frame[censor_col].astype(int)
    frame = frame[frame[censor_col].isin([0, 1])]
    return frame.reset_index(drop=True)


class SurvivalBagDataset(SlideBagDataset):
    def __init__(
        self,
        label_csv: str | Path,
        feature_config: FeatureConfig,
        slide_ids: list[str] | None = None,
        slide_col: str = "slide_id",
        time_col: str = "survival_days",
        censor_col: str = "censorship",
        n_bins: int = 4,
        bin_edges: list[float] | np.ndarray | None = None,
    ) -> None:
        self.feature_config = feature_config
        self.slide_col = slide_col
        self.time_col = time_col
        self.censor_col = censor_col
        self.num_classes = n_bins

        all_rows = read_csv_rows(label_csv)
        if all_rows and any(col not in all_rows[0] for col in [slide_col, time_col, censor_col]):
            raise ValueError(
                f"survival csv must contain '{slide_col}', '{time_col}' and '{censor_col}'"
            )

        valid_frame = _survival_frame(all_rows, time_col, censor_col)
        if bin_edges is None:
            bin_edges = make_survival_bins_from_frame(valid_frame, time_col, censor_col, n_bins)
        self.bin_edges = np.asarray(bin_edges, dtype=float)
        self.num_classes = len(self.bin_edges) - 1

        slide_filter = set(slide_ids) if slide_ids is not None else None
        labeled_frame = assign_survival_disc_labels(valid_frame, time_col, censor_col, self.bin_edges)
        self.slide_data: list[dict[str, str | int | float]] = []
        for row in labeled_frame.to_dict("records"):
            slide_id = row[slide_col]
            if slide_filter is not None and slide_id not in slide_filter:
                continue
            event_time = float(row[time_col])
            item: dict[str, str | int | float] = dict(row)
            item["event_time"] = event_time
            item["censorship"] = int(float(row[censor_col]))
            item["disc_label"] = int(row["disc_label"])
            self.slide_data.append(item)

        if not self.slide_data:
            raise ValueError("survival dataset is empty after filtering labels/splits")

        self.targets = [int(row["disc_label"]) for row in self.slide_data]
        self.slide_cls_ids = [
            np.where(np.asarray(self.targets) == class_id)[0]
            for class_id in range(self.num_classes)
        ]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.slide_data[index]
        slide_id = str(row[self.slide_col])
        item = self._feature_item(slide_id)
        item["label"] = torch.tensor(int(row["disc_label"]), dtype=torch.long)
        item["event_time"] = torch.tensor(float(row["event_time"]), dtype=torch.float32)
        item["censorship"] = torch.tensor(float(row["censorship"]), dtype=torch.float32)
        return item


def collate_bags(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    if len(batch) != 1:
        raise ValueError("MIL bag training expects batch_size=1")
    item = batch[0]
    output: dict[str, torch.Tensor | list[str]] = {}
    for key, value in item.items():
        if key == "label" and torch.is_tensor(value):
            output[key] = value.view(1)
        elif key == "slide_id":
            output[key] = [str(value)]
        else:
            output[key] = value
    return output
