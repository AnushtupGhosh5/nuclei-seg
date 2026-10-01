"""Strict minimum-total-validation-loss selection for geometric experiments."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass


CHECKPOINT_SELECTION = "minimum_validation_total_loss"


def should_early_stop(epoch: int, bad_validations: int, patience: int | None, min_epoch: int) -> bool:
    """Epoch is one-based; patience counts validation events, including warmup."""
    return patience is not None and epoch >= min_epoch and bad_validations >= patience


@dataclass
class LossCheckpointTracker:
    best_val_loss: float = float("inf")
    best_epoch: int = -1  # zero-based, like stored training epoch
    bad_validations: int = 0

    def observe(self, val_total: float, epoch: int) -> bool:
        if not math.isfinite(val_total):
            raise FloatingPointError(f"Non-finite validation total loss: {val_total}")
        improved = val_total < self.best_val_loss
        if improved:
            self.best_val_loss, self.best_epoch, self.bad_validations = val_total, epoch, 0
        else:
            self.bad_validations += 1
        return improved

    def state_dict(self) -> dict:
        return dict(vars(self))

    @classmethod
    def restore(cls, state: dict):
        if state.get("checkpoint_selection", CHECKPOINT_SELECTION) != CHECKPOINT_SELECTION:
            raise ValueError("Resume requires a loss-selected checkpoint")
        # Migration checks only: evaluation metrics can never be a selector in
        # a new run. Reject the superseded PQ-selected checkpoint schema.
        if "best_metric_name" in state and (
            state["best_metric_name"] not in {"loss", "total", "val_total"}
            or state.get("checkpoint_mode") != "min"
        ):
            raise ValueError("Metric-selected checkpoints cannot resume loss-only training; start from pretraining")
        if "bad_validations" not in state:
            warnings.warn("Legacy loss checkpoint has no patience state; bad_validations resets to zero", stacklevel=2)
        return cls(
            best_val_loss=float(state["best_val_loss"]),
            best_epoch=int(state.get("best_epoch", -1)),
            bad_validations=int(state.get("bad_validations", 0)),
        )
