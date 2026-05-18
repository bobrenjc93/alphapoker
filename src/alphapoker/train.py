"""Train and evaluate AlphaPoker baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alphapoker.cfr import KuhnCFRTrainer, TrainingResult
from alphapoker.dataset import strategy_examples


def result_to_dict(result: TrainingResult) -> dict[str, float | int]:
    return {
        "iterations": result.iterations,
        "game_value_p0": result.game_value_p0,
        "best_response_p0": result.best_response_p0,
        "best_response_p1": result.best_response_p1,
        "exploitability": result.exploitability,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train_network(strategy: dict[str, dict[str, float]], out_dir: Path, epochs: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from alphapoker.model import KuhnPolicyValueNet

    out_dir.mkdir(parents=True, exist_ok=True)
    examples = strategy_examples(strategy)
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    policies = torch.tensor([example.policy for example in examples], dtype=torch.float32)
    masks = torch.tensor([example.legal_mask for example in examples], dtype=torch.bool)

    torch.manual_seed(0)
    model = KuhnPolicyValueNet(input_dim=features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    last_loss = 0.0
    for _ in range(epochs):
        logits, values = model(features)
        masked_logits = logits.masked_fill(~masks, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        policy_loss = -(policies * log_probs).sum(dim=-1).mean()
        value_loss = 0.01 * values.square().mean()
        loss = policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    checkpoint_path = out_dir / "kuhn_policy_value.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "infosets": [example.infoset for example in examples],
            "canonical_actions": ["check", "bet", "call", "fold"],
        },
        checkpoint_path,
    )
    return {
        "network_epochs": epochs,
        "network_final_loss": last_loss,
        "checkpoint": str(checkpoint_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    trainer = KuhnCFRTrainer(cfr_plus=not args.vanilla_cfr)
    result = trainer.train(args.iterations)
    strategy = trainer.average_strategy()

    out_dir = Path(args.out)
    metrics: dict[str, Any] = result_to_dict(result)

    strategy_payload: dict[str, Any] = {
        "game": "kuhn_poker",
        "algorithm": "cfr" if args.vanilla_cfr else "cfr_plus",
        "metrics": metrics,
        "strategy": strategy,
    }
    write_json(out_dir / "strategy.json", strategy_payload)

    if args.network_epochs > 0:
        metrics.update(train_network(strategy, out_dir, args.network_epochs))

    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--out", type=Path, default=Path("experiments/kuhn_cfr_baseline"))
    parser.add_argument("--network-epochs", type=int, default=0)
    parser.add_argument("--vanilla-cfr", action="store_true")
    return parser


def main() -> None:
    metrics = run(build_parser().parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
