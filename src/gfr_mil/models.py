from __future__ import annotations

import torch.nn as nn

from .model_zoo.adapters import ExactModelAdapter
from .model_zoo.gfr_mil_exact import PathoMIL_RE


def build_model(
    input_dim: int,
    num_classes: int,
    high_input_dim: int | None = None,
    hidden_dim: int = 512,
    dropout: float = 0.25,
) -> nn.Module:
    return ExactModelAdapter(
        PathoMIL_RE(
            dim=input_dim,
            high_dim=high_input_dim or input_dim,
            hidden_dim=hidden_dim,
            n_classes=num_classes,
            task="subtype",
            dropout=dropout,
        ),
        task="classification",
    )


def build_survival_model(
    input_dim: int,
    n_bins: int,
    high_input_dim: int | None = None,
    hidden_dim: int = 512,
    dropout: float = 0.25,
) -> nn.Module:
    return ExactModelAdapter(
        PathoMIL_RE(
            dim=input_dim,
            high_dim=high_input_dim or input_dim,
            hidden_dim=hidden_dim,
            n_classes=n_bins,
            task="survival",
            dropout=dropout,
        ),
        task="survival",
    )
