"""Train a Hold'em policy with actor-critic against fixed opponents."""

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
from alphapoker.train_holdem_policy_gradient import (
    choose_weighted_index,
    parse_policy_mix,
    parse_policy_weights,
)


def sample_actor_action(policy_model, value_model, state: FixedLimitHoldemState):
    import torch

    features = torch.tensor([encode_holdem_state(state)], dtype=torch.float32)
    mask = torch.tensor(holdem_legal_action_mask(state), dtype=torch.bool)
    logits = policy_model(features).squeeze(0).masked_fill(~mask, -1e9)
    distribution = torch.distributions.Categorical(logits=logits)
    action_index = distribution.sample()
    value = value_model(features).squeeze(0)
    return (
        HOLDEM_CANONICAL_ACTIONS[int(action_index.item())],
        distribution.log_prob(action_index),
        distribution.entropy(),
        value,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from alphapoker.holdem_model import HoldemPolicyNet, HoldemValueNet

    torch.manual_seed(args.seed)
    input_dim = len(encode_holdem_state(deal_fixed_limit_holdem(random.Random(args.seed + 3))))
    deal_rng = random.Random(args.seed + 1)
    opponent_selector_rng = random.Random(args.seed + 2)
    opponent_policy_names = args.opponent_policies or (args.opponent_policy,)
    opponent_policy_weights = args.opponent_policy_weights
    if opponent_policy_weights is None:
        opponent_policy_weights = tuple(1.0 for _ in opponent_policy_names)
    if len(opponent_policy_weights) != len(opponent_policy_names):
        raise ValueError("opponent policy weights must match opponent policies")
    opponent_policies = [
        make_policy(name, random.Random(args.seed + 100 + index), args.equity_sims)
        for index, name in enumerate(opponent_policy_names)
    ]

    policy_model = HoldemPolicyNet(input_dim=input_dim)
    if args.init_checkpoint is not None:
        checkpoint_data = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_input_dim = int(checkpoint_data["input_dim"])
        if checkpoint_input_dim != input_dim:
            raise ValueError(
                f"init checkpoint input_dim {checkpoint_input_dim} does not match current {input_dim}"
            )
        policy_model.load_state_dict(checkpoint_data["model_state_dict"])
    value_model = HoldemValueNet(input_dim=input_dim)
    optimizer = torch.optim.AdamW(
        [*policy_model.parameters(), *value_model.parameters()],
        lr=args.lr,
        weight_decay=1e-4,
    )
    utilities: list[float] = []
    best_batch_avg_utility = float("-inf")
    best_policy_state = copy.deepcopy(policy_model.state_dict())
    best_value_state = copy.deepcopy(value_model.state_dict())

    hands_played = 0
    while hands_played < args.hands:
        policy_state_before_batch = copy.deepcopy(policy_model.state_dict())
        value_state_before_batch = copy.deepcopy(value_model.state_dict())
        batch_terms = []
        batch_utilities = []
        for _ in range(min(args.batch_hands, args.hands - hands_played)):
            state = deal_fixed_limit_holdem(deal_rng)
            opponent_policy = opponent_policies[
                choose_weighted_index(opponent_selector_rng, opponent_policy_weights)
            ]
            action_terms = []
            while not state.is_terminal():
                player = state.current_player()
                if player == args.model_player:
                    action, log_prob, entropy, value = sample_actor_action(
                        policy_model,
                        value_model,
                        state,
                    )
                    action_terms.append((log_prob, entropy, value))
                else:
                    action = opponent_policy(state)
                state = state.apply(action)

            reward = state.utility(args.model_player)
            batch_utilities.append(reward)
            utilities.append(reward)
            for log_prob, entropy, value in action_terms:
                batch_terms.append((log_prob, entropy, value, reward))
            hands_played += 1

        if not batch_terms:
            continue
        baseline = sum(batch_utilities) / len(batch_utilities)
        if baseline > best_batch_avg_utility:
            best_batch_avg_utility = baseline
            best_policy_state = policy_state_before_batch
            best_value_state = value_state_before_batch

        losses = []
        for log_prob, entropy, value, reward in batch_terms:
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            advantage = reward_tensor - value.detach()
            policy_loss = -log_prob * advantage - args.entropy_coef * entropy
            value_loss = F.mse_loss(value, reward_tensor)
            losses.append(policy_loss + args.value_loss_coef * value_loss)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    policy_model.load_state_dict(best_policy_state)
    value_model.load_state_dict(best_value_state)
    utility_stdev = statistics.stdev(utilities) if len(utilities) > 1 else 0.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_checkpoint = out_dir / "holdem_policy.pt"
    value_checkpoint = out_dir / "holdem_value.pt"
    torch.save(
        {
            "model_state_dict": policy_model.state_dict(),
            "canonical_actions": list(HOLDEM_CANONICAL_ACTIONS),
            "input_dim": input_dim,
        },
        policy_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": value_model.state_dict(),
            "input_dim": input_dim,
        },
        value_checkpoint,
    )
    metrics: dict[str, Any] = {
        "hands": args.hands,
        "batch_hands": args.batch_hands,
        "model_player": args.model_player,
        "opponent_policy": args.opponent_policy,
        "opponent_policies": list(opponent_policy_names),
        "opponent_policy_weights": list(opponent_policy_weights),
        "equity_sims": args.equity_sims,
        "lr": args.lr,
        "entropy_coef": args.entropy_coef,
        "value_loss_coef": args.value_loss_coef,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "best_batch_avg_utility_model": best_batch_avg_utility,
        "train_avg_utility_model": sum(utilities) / len(utilities) if utilities else 0.0,
        "train_utility_stdev_model": utility_stdev,
        "train_utility_stderr_model": utility_stdev / (len(utilities) ** 0.5) if utilities else 0.0,
        "checkpoint": str(policy_checkpoint),
        "value_checkpoint": str(value_checkpoint),
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
    parser.add_argument("--opponent-policies", type=parse_policy_mix)
    parser.add_argument("--opponent-policy-weights", type=parse_policy_weights)
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
