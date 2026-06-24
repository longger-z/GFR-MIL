from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _classification_output(logits: torch.Tensor, **extra: torch.Tensor) -> dict[str, torch.Tensor]:
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    output = {"logits": logits, "probs": probs, "preds": preds}
    output.update(extra)
    return output


def _survival_output(hazards: torch.Tensor, survival: torch.Tensor, **extra: torch.Tensor) -> dict[str, torch.Tensor]:
    risk = -torch.sum(survival, dim=1)
    preds = torch.argmax(hazards, dim=1)
    output = {
        "logits": hazards,
        "hazards": hazards,
        "survival": survival,
        "risk": risk,
        "preds": preds,
    }
    output.update(extra)
    return output


def _adapt_tuple(output, task: str) -> dict[str, torch.Tensor]:
    if not isinstance(output, tuple):
        logits = output
        return _classification_output(logits) if task != "survival" else _survival_output(torch.sigmoid(logits), torch.cumprod(1 - torch.sigmoid(logits), dim=1))

    if task == "survival":
        first = output[0]
        extra = {}
        if len(output) >= 2 and torch.is_tensor(output[1]) and output[1].shape == first.shape:
            hazards, survival = first, output[1]
        else:
            hazards = torch.sigmoid(first)
            survival = torch.cumprod(1 - hazards, dim=1)
            if len(output) > 1 and torch.is_tensor(output[1]) and output[1].ndim == 0:
                extra["aux_loss"] = output[1]
        if len(output) > 3 and torch.is_tensor(output[3]):
            extra["aux_loss"] = output[3]
        return _survival_output(hazards, survival, **extra)

    logits = output[0]
    if len(output) >= 3 and torch.is_tensor(output[1]) and output[1].shape == logits.shape:
        result = {"logits": logits, "probs": output[1], "preds": output[2]}
    else:
        result = _classification_output(logits)
        if len(output) > 1 and torch.is_tensor(output[1]) and output[1].ndim == 0:
            result["aux_loss"] = output[1]
    if len(output) > 3 and torch.is_tensor(output[3]):
        result["aux_loss"] = output[3]
    return result


class ExactModelAdapter(nn.Module):

    def __init__(self, model: nn.Module, task: str, batched_features: bool = True) -> None:
        super().__init__()
        self.model = model
        self.task = task
        self.batched_features = batched_features

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor | None = None,
        high_features: torch.Tensor | None = None,
        high_coords: torch.Tensor | None = None,
        region_labels: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if high_features is not None:
            model_region_labels = (
                region_labels.unsqueeze(0)
                if region_labels is not None and region_labels.dim() == 1
                else region_labels
            )
            output = self.model(high_features, high_coords, features, coords, model_region_labels, **kwargs)
        else:
            model_features = features.unsqueeze(0) if self.batched_features and features.dim() == 2 else features
            model_coords = coords.unsqueeze(0) if self.batched_features and coords is not None and coords.dim() == 2 else coords
            try:
                output = self.model(model_features, coords=model_coords, label=label, instance_eval=label is not None, **kwargs)
            except TypeError:
                try:
                    output = self.model(model_features, label=label, instance_eval=label is not None, **kwargs)
                except TypeError:
                    try:
                        output = self.model(model_features, coords=model_coords, **kwargs)
                    except TypeError:
                        output = self.model(model_features, **kwargs)
        return _adapt_tuple(output, self.task)
