"""Train a Hold'em policy with REINFORCE against a fixed opponent."""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
from pathlib import Path
from typing import Any

from alphapoker.holdem import FixedLimitHoldemState, deal_fixed_limit_holdem
from alphapoker.holdem_features import (
    HOLDEM_CANONICAL_ACTIONS,
    encode_holdem_state,
    holdem_legal_action_mask,
)
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json


def sample_model_action(model, state: FixedLimitHoldemState):
    import torch

    features = torch.tensor([encode_holdem_state(state)], dtype=torch.float32)
    mask = torch.tensor(holdem_legal_action_mask(state), dtype=torch.bool)
    logits = model(features).squeeze(0).masked_fill(~mask, -1e9)
    distribution = torch.distributions.Categorical(logits=logits)
    action_index = distribution.sample()
    return (
        HOLDEM_CANONICAL_ACTIONS[int(action_index.item())],
        distribution.log_prob(action_index),
        distribution.entropy(),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from alphapoker.holdem_model import HoldemPolicyNet

    torch.manual_seed(args.seed)
    input_dim = len(encode_holdem_state(deal_fixed_limit_holdem(random.Random(args.seed + 3))))
    deal_rng = random.Random(args.seed + 1)
    opponent_rng = random.Random(args.seed + 2)
    opponent_policy = make_policy(args.opponent_policy, opponent_rng, args.equity_sims)

    model = HoldemPolicyNet(input_dim=input_dim)
    if args.init_checkpoint is not None:
        checkpoint_data = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_input_dim = int(checkpoint_data["input_dim"])
        if checkpoint_input_dim != input_dim:
            raise ValueError(
                f"init checkpoint input_dim {checkpoint_input_dim} does not match current {input_dim}"
            )
        model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    utilities: list[float] = []
    best_batch_avg_utility = float("-inf")
    best_state = copy.deepcopy(model.state_dict())

    hands_played = 0
    while hands_played < args.hands:
        state_before_batch = copy.deepcopy(model.state_dict())
        batch_terms = []
        batch_utilities = []
        for _ in range(min(args.batch_hands, args.hands - hands_played)):
            state = deal_fixed_limit_holdem(deal_rng)
            log_probs = []
            entropies = []
            while not state.is_terminal():
                player = state.current_player()
                if player == args.model_player:
                    action, log_prob, entropy = sample_model_action(model, state)
                    log_probs.append(log_prob)
                    entropies.append(entropy)
                else:
                    action = opponent_policy(state)
                state = state.apply(action)

            reward = state.utility(args.model_player)
            batch_utilities.append(reward)
            utilities.append(reward)
            if log_probs:
                batch_terms.append((torch.stack(log_probs).sum(), torch.stack(entropies).sum(), reward))
            hands_played += 1

        if not batch_terms:
            continue
        baseline = sum(batch_utilities) / len(batch_utilities)
        if baseline > best_batch_avg_utility:
            best_batch_avg_utility = baseline
            best_state = state_before_batch
        losses = [
            -log_prob_sum * (reward - baseline) - args.entropy_coef * entropy_sum
            for log_prob_sum, entropy_sum, reward in batch_terms
        ]
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.load_state_dict(best_state)
    utility_stdev = statistics.stdev(utilities) if len(utilities) > 1 else 0.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "holdem_policy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "canonical_actions": list(HOLDEM_CANONICAL_ACTIONS),
            "input_dim": input_dim,
        },
        checkpoint,
    )
    metrics: dict[str, Any] = {
        "hands": args.hands,
        "batch_hands": args.batch_hands,
        "model_player": args.model_player,
        "opponent_policy": args.opponent_policy,
        "equity_sims": args.equity_sims,
        "lr": args.lr,
        "entropy_coef": args.entropy_coef,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "best_batch_avg_utility_model": best_batch_avg_utility,
        "train_avg_utility_model": sum(utilities) / len(utilities) if utilities else 0.0,
        "train_utility_stdev_model": utility_stdev,
        "train_utility_stderr_model": utility_stdev / (len(utilities) ** 0.5) if utilities else 0.0,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
    }
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--batch-hands", type=int, default=50)
    parser.add_argument("--model-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="random")
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
