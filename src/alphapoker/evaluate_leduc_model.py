"""Evaluate a distilled Leduc policy/value checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alphapoker.leduc_cfr import (
    LeducStrategyProfile,
    expected_leduc_utility,
    leduc_exploitability,
)
from alphapoker.leduc_dataset import leduc_strategy_examples
from alphapoker.leduc_features import LEDUC_CANONICAL_ACTIONS
from alphapoker.train import write_json


def load_leduc_strategy(path: Path) -> tuple[LeducStrategyProfile, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("game") != "leduc_poker":
        raise ValueError("Expected a Leduc strategy JSON")
    return payload["strategy"], payload


def strategy_from_checkpoint(
    checkpoint_path: Path,
    reference_strategy: LeducStrategyProfile,
) -> tuple[LeducStrategyProfile, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    from alphapoker.leduc_model import LeducPolicyValueNet

    examples = leduc_strategy_examples(reference_strategy)
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    masks = torch.tensor([example.legal_mask for example in examples], dtype=torch.bool)
    target = torch.tensor([example.policy for example in examples], dtype=torch.float32)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LeducPolicyValueNet(input_dim=features.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits, values = model(features)
        masked_logits = logits.masked_fill(~masks, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = log_probs.exp()

    policy_cross_entropy = float((-(target * log_probs).sum(dim=-1)).mean().detach().cpu())
    target_safe = target.clamp_min(1e-12)
    probs_safe = probs.clamp_min(1e-12)
    policy_kl = float(
        (target * (target_safe.log() - probs_safe.log())).sum(dim=-1).mean().detach().cpu()
    )
    mean_abs_value = float(values.abs().mean().detach().cpu())

    neural_strategy: LeducStrategyProfile = {}
    prob_rows = probs.detach().cpu().tolist()
    for example, prob_row in zip(examples, prob_rows):
        neural_strategy[example.infoset] = {
            action: float(prob_row[index])
            for index, action in enumerate(LEDUC_CANONICAL_ACTIONS)
            if example.legal_mask[index]
        }

    return neural_strategy, {
        "policy_cross_entropy": policy_cross_entropy,
        "policy_kl": policy_kl,
        "mean_abs_value_head": mean_abs_value,
    }


def evaluate_model_policy(
    checkpoint_path: Path,
    strategy_json_path: Path,
) -> tuple[LeducStrategyProfile, dict[str, Any]]:
    reference_strategy, source_payload = load_leduc_strategy(strategy_json_path)
    neural_strategy, distill_metrics = strategy_from_checkpoint(checkpoint_path, reference_strategy)
    metrics: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "source_strategy_json": str(strategy_json_path),
        "source_metrics": source_payload.get("metrics", {}),
        "infosets": len(neural_strategy),
        "game_value_p0": expected_leduc_utility(neural_strategy, player=0),
        "exploitability": leduc_exploitability(neural_strategy),
    }
    metrics.update(distill_metrics)
    return neural_strategy, metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    neural_strategy, metrics = evaluate_model_policy(args.checkpoint, args.strategy_json)
    out_dir = Path(args.out)
    write_json(out_dir / "model_eval.json", metrics)
    write_json(
        out_dir / "model_strategy.json",
        {
            "game": "leduc_poker",
            "source": str(args.checkpoint),
            "metrics": metrics,
            "strategy": neural_strategy,
        },
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--strategy-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    metrics = run(build_parser().parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
