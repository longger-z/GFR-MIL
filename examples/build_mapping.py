from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def h5_paths(root: Path) -> list[Path]:
    direct = sorted(root.glob("*.h5"))
    nested = sorted((root / "h5_files").glob("*.h5"))
    return direct or nested


def read_coords(path: Path, coord_key: str) -> np.ndarray:
    with h5py.File(path, "r") as h5_file:
        if coord_key not in h5_file:
            raise KeyError(f"{path} does not contain '{coord_key}'")
        return np.asarray(h5_file[coord_key][:])


def write_mapping(
    slide_id: str,
    coords: np.ndarray,
    output_dir: Path,
    high_patch_size: int,
    high_magnification: float,
    low_magnification: float,
    low_patch_size: int,
) -> Path:
    if high_magnification < low_magnification:
        raise ValueError("high_magnification must be greater than or equal to low_magnification")
    scale = high_magnification / low_magnification
    region_size = low_patch_size * scale
    grid = np.floor(coords[:, :2] / region_size).astype(np.int64)
    folder = output_dir / slide_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{int(scale)}_{high_patch_size}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "grid_x", "grid_y"])
        writer.writeheader()
        for coord, grid_coord in zip(coords[:, :2], grid):
            writer.writerow(
                {
                    "x": int(coord[0]),
                    "y": int(coord[1]),
                    "grid_x": int(grid_coord[0]),
                    "grid_y": int(grid_coord[1]),
                }
            )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-to-low coordinate mapping CSVs for GFR-MIL.")
    parser.add_argument("--high_feature_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--coord_key", default="coords")
    parser.add_argument("--high_patch_size", type=int, default=448)
    parser.add_argument("--high_magnification", type=float, default=40.0)
    parser.add_argument("--low_magnification", type=float, default=5.0)
    parser.add_argument("--low_patch_size", type=int, default=448)
    args = parser.parse_args()

    root = Path(args.high_feature_dir)
    output_dir = Path(args.output_dir)
    paths = h5_paths(root)
    if not paths:
        raise FileNotFoundError(f"no h5 files found in {root}")
    for path in paths:
        coords = read_coords(path, args.coord_key)
        mapping_path = write_mapping(
            path.stem,
            coords,
            output_dir,
            args.high_patch_size,
            args.high_magnification,
            args.low_magnification,
            args.low_patch_size,
        )
        print(mapping_path)


if __name__ == "__main__":
    main()
