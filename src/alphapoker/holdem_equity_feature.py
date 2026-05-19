"""Helpers for using trained Hold'em equity models as scalar features."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from alphapoker.holdem import FixedLimitHoldemState
from alphapoker.holdem_features import adapt_holdem_features, encode_holdem_state

HoldemEquityEstimator = Callable[[FixedLimitHoldemState], float]


def resolve_equity_checkpoint_path(
    checkpoint_reference: str | Path,
    *,
    relative_to: Path | None = None,
) -> Path:
    path = Path(checkpoint_reference)
    if path.is_absolute() or path.exists() or relative_to is None:
        return path
    return relative_to.parent / path


def equity_estimator_from_checkpoint(checkpoint_path: Path) -> HoldemEquityEstimator:
    import torch

    from alphapoker.holdem_model import HoldemEquityNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    model = HoldemEquityNet(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def estimate(state: FixedLimitHoldemState) -> float:
        features = torch.tensor(
            [adapt_holdem_features(encode_holdem_state(state), input_dim)],
            dtype=torch.float32,
        )
        with torch.no_grad():
            return float(torch.sigmoid(model(features)).item())

    return estimate
