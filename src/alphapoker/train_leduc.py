"""Train the tabular Leduc CFR baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alphapoker.leduc_dataset import leduc_strategy_examples
from alphapoker.leduc_features import LEDUC_CANONICAL_ACTIONS
from alphapoker.leduc_cfr import LeducCFRTrainer, LeducTrainingResult
from alphapoker.train import write_json


def result_to_dict(result: LeducTrainingResult) -> dict[str, float | int]:
    payload: dict[str, float | int] = {
        "iterations": result.iterations,
        "game_value_p0": result.game_value_p0,
        "infosets": result.infosets,
    }
    if result.exploitability is not None:
        payload["exploitability"] = result.exploitability
    return payload


def train_network(
    strategy: dict[str, dict[str, float]],
    out_dir: Path,
    epochs: int,
) -> dict[str, Any]:
    import copy

    import torch
    import torch.nn.functional as F

    from alphapoker.leduc_model import LeducPolicyValueNet

    out_dir.mkdir(parents=True, exist_ok=True)
    examples = leduc_strategy_examples(strategy)
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    policies = torch.tensor([example.policy for example in examples], dtype=torch.float32)
    masks = torch.tensor([example.legal_mask for example in examples], dtype=torch.bool)

    torch.manual_seed(0)
    model = LeducPolicyValueNet(input_dim=features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    last_loss = 0.0
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
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
        if last_loss < best_loss:
            best_loss = last_loss
            best_state = copy.deepcopy(model.state_dict())

    checkpoint_path = out_dir / "leduc_policy_value.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "infosets": [example.infoset for example in examples],
            "canonical_actions": list(LEDUC_CANONICAL_ACTIONS),
        },
        checkpoint_path,
    )
    return {
        "network_epochs": epochs,
        "network_final_loss": last_loss,
        "network_best_loss": best_loss,
        "checkpoint": str(checkpoint_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint_in is None:
        trainer = LeducCFRTrainer(cfr_plus=not args.vanilla_cfr)
    else:
        trainer = LeducCFRTrainer.load_checkpoint(args.checkpoint_in)

    result = trainer.train(args.iterations, compute_exploitability=args.best_response)
    strategy = trainer.average_strategy()

    out_dir = Path(args.out)
    metrics: dict[str, Any] = result_to_dict(result)
    if args.network_epochs > 0:
        metrics.update(train_network(strategy, out_dir, args.network_epochs))
    if args.checkpoint_out is not None:
        trainer.save_checkpoint(args.checkpoint_out)
        metrics["checkpoint_out"] = str(args.checkpoint_out)

    write_json(
        out_dir / "strategy.json",
        {
            "game": "leduc_poker",
            "algorithm": "cfr" if args.vanilla_cfr else "cfr_plus",
            "metrics": metrics,
            "strategy": strategy,
        },
    )
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("experiments/leduc_cfr_smoke"))
    parser.add_argument("--vanilla-cfr", action="store_true")
    parser.add_argument("--best-response", action="store_true")
    parser.add_argument("--network-epochs", type=int, default=0)
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    return parser


def main() -> None:
    metrics = run(build_parser().parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
