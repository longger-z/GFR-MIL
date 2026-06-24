# GFR-MIL

Official training and evaluation code for **GFR-MIL: Glance-Focus-Reflect Based Multiple Instance Learning for Multi-Scale Whole Slide Image Analysis**.

GFR-MIL starts from pre-extracted low- and high-magnification WSI features. The model iteratively glances over low-scale regions, focuses on mapped high-scale patches, and reflects high-resolution evidence back to the global representation. During training it uses stochastic region sampling and a recycling token for unselected regions; during evaluation it uses deterministic top-k region selection without recycling.

## Installation

```bash
git clone <your-github-url>/GFR-MIL.git
cd GFR-MIL
pip install -e .
```

`xformers` installation depends on your CUDA/PyTorch build. If the generic install fails, install the matching wheel from the official xFormers instructions, then rerun `pip install -e .`.

## WSI Preprocessing

This repository does not implement WSI segmentation, patching, image cropping, or feature extraction. For raw WSI processing, use the official CLAM preprocessing workflow:

https://github.com/mahmoodlab/CLAM

Use CLAM or your own feature extractor to create H5 files with:

```text
features: float array with shape [num_patches, feature_dim]
coords: integer or float array with shape [num_patches, 2]
```

GFR-MIL requires both low- and high-magnification feature directories:

```text
low_features/h5_files/<slide_id>.h5
high_features/h5_files/<slide_id>.h5
mapping/<slide_id>/8_448.csv
```

The mapping CSV must contain `x`, `y`, `grid_x`, and `grid_y`. Each high-scale coordinate `(x, y)` is assigned to a low-scale region defined by `(grid_x, grid_y)`.

## Build Mapping Files

If your high-scale H5 files already contain coordinates, generate mapping CSVs with:

```bash
python examples/build_mapping.py \
  --high_feature_dir /path/to/high_features/h5_files \
  --output_dir /path/to/mapping \
  --high_patch_size 448 \
  --high_magnification 40 \
  --low_magnification 5 \
  --low_patch_size 448
```

This script only processes coordinates. It does not read or crop WSI images.

## Classification

Create a label CSV:

```text
slide_id,case_id,label
slide_001,case_001,A
slide_002,case_002,B
```

Create a split CSV:

```text
train,val,test
slide_001,slide_010,slide_020
slide_002,slide_011,slide_021
```

Train:

```bash
mil-train \
  --label_csv labels.csv \
  --split_csv splits_0.csv \
  --feature_dir /path/to/low_features \
  --high_feature_dir /path/to/high_features \
  --mapping_dir /path/to/mapping \
  --label_map "A:0,B:1" \
  --input_dim 1024 \
  --high_input_dim 1024 \
  --output_dir results/classification
```

Evaluate:

```bash
mil-eval \
  --label_csv labels.csv \
  --split_csv splits_0.csv \
  --checkpoint results/classification/best.pt \
  --feature_dir /path/to/low_features \
  --high_feature_dir /path/to/high_features \
  --mapping_dir /path/to/mapping \
  --label_map "A:0,B:1" \
  --input_dim 1024 \
  --high_input_dim 1024 \
  --output_dir results/classification_eval
```

## Survival

Create a survival CSV:

```text
slide_id,case_id,survival_days,censorship
slide_001,case_001,420,0
slide_002,case_002,760,1
```

Train:

```bash
mil-survival-train \
  --label_csv survival.csv \
  --split_csv splits_0.csv \
  --feature_dir /path/to/low_features \
  --high_feature_dir /path/to/high_features \
  --mapping_dir /path/to/mapping \
  --time_col survival_days \
  --censor_col censorship \
  --input_dim 1024 \
  --high_input_dim 1024 \
  --output_dir results/survival
```

Evaluate:

```bash
mil-survival-eval \
  --label_csv survival.csv \
  --split_csv splits_0.csv \
  --checkpoint results/survival/best.pt \
  --feature_dir /path/to/low_features \
  --high_feature_dir /path/to/high_features \
  --mapping_dir /path/to/mapping \
  --time_col survival_days \
  --censor_col censorship \
  --input_dim 1024 \
  --high_input_dim 1024 \
  --output_dir results/survival_eval
```
