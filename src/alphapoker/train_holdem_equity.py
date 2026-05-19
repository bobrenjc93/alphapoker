"""Train a fixed-limit Hold'em equity predictor."""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from alphapoker.holdem_dataset import (
    HOLDEM_EQUITY_VALUE_OPPONENT_POLICIES,
    generate_equity_value_examples,
    read_equity_value_examples,
    write_equity_value_examples,
)
from alphapoker.train import write_json


def parse_player(value: str) -> int | None:
    if value == "both":
        return None
    try:
        player = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("player must be 0, 1, or both") from error
    if player not in (0, 1):
        raise argparse.ArgumentTypeError("player must be 0, 1, or both")
    return player


def player_label(player: int | None) -> int | str:
    return "both" if player is None else player


def _shard_hands(hands: int, jobs: int) -> list[int]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if hands <= 0:
        return []
    shards = min(hands, jobs)
    base = hands // shards
    extra = hands % shards
    return [base + (1 if index < extra else 0) for index in range(shards)]


def generate_equity_value_examples_shard(
    index: int,
    *,
    hands: int,
    seed: int,
    equity_sims: int,
    player: int | None,
    opponent_policy: str,
):
    return generate_equity_value_examples(
        hands=hands,
        seed=seed + index * 1_000_003,
        equity_sims=equity_sims,
        player=player,
        opponent_policy=opponent_policy,
    )


def generate_training_examples(
    *,
    hands: int,
    seed: int,
    equity_sims: int,
    player: int | None,
    opponent_policy: str,
    jobs: int,
):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if jobs == 1:
        return generate_equity_value_examples(
            hands=hands,
            seed=seed,
            equity_sims=equity_sims,
            player=player,
            opponent_policy=opponent_policy,
        )

    examples = []
    shard_hands = _shard_hands(hands, jobs)
    if not shard_hands:
        return examples
    with ProcessPoolExecutor(max_workers=min(jobs, len(shard_hands))) as executor:
        future_to_index = {
            executor.submit(
                generate_equity_value_examples_shard,
                index,
                hands=shard_size,
                seed=seed,
                equity_sims=equity_sims,
                player=player,
                opponent_policy=opponent_policy,
            ): index
            for index, shard_size in enumerate(shard_hands)
        }
        shard_results = {}
        for future in as_completed(future_to_index):
            shard_results[future_to_index[future]] = future.result()
    for index in sorted(shard_results):
        examples.extend(shard_results[index])
    return examples


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from alphapoker.holdem_model import HoldemEquityNet

    examples_in = getattr(args, "examples_in", None)
    examples_out = getattr(args, "examples_out", None)
    jobs = getattr(args, "jobs", 1)
    if examples_in is not None:
        examples = read_equity_value_examples(examples_in)
    else:
        examples = generate_training_examples(
            hands=args.hands,
            seed=args.seed,
            equity_sims=args.equity_sims,
            player=args.player,
            opponent_policy=args.opponent_policy,
            jobs=jobs,
        )
    if examples_out is not None:
        write_equity_value_examples(examples_out, examples)
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    targets = torch.tensor([example.equity for example in examples], dtype=torch.float32)

    torch.manual_seed(0)
    model = HoldemEquityNet(input_dim=features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    final_loss = 0.0
    for _ in range(args.epochs):
        logits = model(features)
        predictions = torch.sigmoid(logits)
        loss = F.mse_loss(predictions, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if final_loss < best_loss:
            best_loss = final_loss
            best_state = copy.deepcopy(model.state_dict())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "holdem_equity.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": features.shape[1],
        },
        checkpoint,
    )
    metrics: dict[str, Any] = {
        "hands": args.hands,
        "examples": len(examples),
        "equity_sims": args.equity_sims,
        "player": player_label(args.player),
        "opponent_policy": args.opponent_policy,
        "jobs": jobs,
        "epochs": args.epochs,
        "lr": args.lr,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
    }
    if examples_in is not None:
        metrics["examples_in"] = str(examples_in)
    if examples_out is not None:
        metrics["examples_out"] = str(examples_out)
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--equity-sims", type=int, default=16)
    parser.add_argument("--player", type=parse_player, default=0)
    parser.add_argument(
        "--opponent-policy",
        choices=HOLDEM_EQUITY_VALUE_OPPONENT_POLICIES,
        default="random",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-in", type=Path)
    parser.add_argument("--examples-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
