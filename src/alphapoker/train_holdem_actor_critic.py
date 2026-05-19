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
from alphapoker.holdem_policy_features import (
    HoldemPolicyFeatureEncoder,
    POLICY_FEATURE_EQUITY_MODES,
    policy_feature_encoder_for_training,
)
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json
from alphapoker.train_holdem_policy_gradient import (
    choose_weighted_index,
    evaluate_trained_policy,
    evaluate_selection_checkpoint,
    compact_selection_evaluation,
    model_player_label,
    normalize_model_players,
    parse_model_players,
    parse_policy_mix,
    parse_policy_weights,
    save_policy_checkpoint,
    selection_evaluation_metadata,
    should_run_selection_evaluation,
    validate_checkpoint_selection_args,
)


def sample_actor_action(
    policy_model,
    value_model,
    state: FixedLimitHoldemState,
    feature_encoder: HoldemPolicyFeatureEncoder,
):
    import torch

    features = torch.tensor([feature_encoder.encode(state)], dtype=torch.float32)
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
    checkpoint_selection = getattr(args, "checkpoint_selection", "best-batch")
    validate_checkpoint_selection_args(args, checkpoint_selection)
    model_players = normalize_model_players(args.model_player)
    model_player_weights = args.model_player_weights
    if model_player_weights is not None and len(model_player_weights) != len(model_players):
        raise ValueError("model player weights must match model players")
    init_checkpoint_data = None
    base_input_dim = len(encode_holdem_state(deal_fixed_limit_holdem(random.Random(args.seed + 3))))
    if args.init_checkpoint is not None:
        init_checkpoint_data = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    feature_encoder = policy_feature_encoder_for_training(
        base_input_dim=base_input_dim,
        checkpoint=init_checkpoint_data,
        checkpoint_path=args.init_checkpoint,
        feature_seed=args.seed + 5,
        feature_equity_sims=args.feature_equity_sims,
        feature_equity_mode=args.feature_equity_mode,
    )
    input_dim = feature_encoder.input_dim
    deal_rng = random.Random(args.seed + 1)
    opponent_selector_rng = random.Random(args.seed + 2)
    model_player_selector_rng = random.Random(args.seed + 4)
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
    if init_checkpoint_data is not None:
        policy_model.load_state_dict(init_checkpoint_data["model_state_dict"])
    value_model = HoldemValueNet(input_dim=input_dim)
    optimizer = torch.optim.AdamW(
        [*policy_model.parameters(), *value_model.parameters()],
        lr=args.lr,
        weight_decay=1e-4,
    )
    utilities: list[float] = []
    utilities_by_model_player: dict[int, list[float]] = {player: [] for player in model_players}
    best_batch_avg_utility = float("-inf")
    best_policy_state = copy.deepcopy(policy_model.state_dict())
    best_value_state = copy.deepcopy(value_model.state_dict())
    best_selection_eval_avg_utility = float("-inf")
    best_selection_eval_hands_played: int | None = None
    best_selection_policy_state = copy.deepcopy(policy_model.state_dict())
    best_selection_value_state = copy.deepcopy(value_model.state_dict())
    selection_evaluations: list[dict[str, Any]] = []
    last_selection_eval_hands: int | None = None
    out_dir = Path(args.out)
    selection_candidate_checkpoint = out_dir / "holdem_policy_selection_candidate.pt"

    def run_selection_evaluation() -> None:
        nonlocal best_selection_eval_avg_utility
        nonlocal best_selection_eval_hands_played
        nonlocal best_selection_policy_state
        nonlocal best_selection_value_state
        nonlocal last_selection_eval_hands
        out_dir.mkdir(parents=True, exist_ok=True)
        save_policy_checkpoint(
            path=selection_candidate_checkpoint,
            model_state_dict=policy_model.state_dict(),
            input_dim=input_dim,
            feature_encoder=feature_encoder,
        )
        eval_metrics = evaluate_selection_checkpoint(
            args,
            selection_candidate_checkpoint,
            model_players,
        )
        summary = compact_selection_evaluation(
            hands_played=hands_played,
            metrics=eval_metrics,
        )
        selection_evaluations.append(summary)
        avg_utility = float(summary["avg_utility_model"])
        if avg_utility > best_selection_eval_avg_utility:
            best_selection_eval_avg_utility = avg_utility
            best_selection_eval_hands_played = hands_played
            best_selection_policy_state = copy.deepcopy(policy_model.state_dict())
            best_selection_value_state = copy.deepcopy(value_model.state_dict())
        last_selection_eval_hands = hands_played

    hands_played = 0
    if checkpoint_selection == "evaluation":
        run_selection_evaluation()
    while hands_played < args.hands:
        policy_state_before_batch = copy.deepcopy(policy_model.state_dict())
        value_state_before_batch = copy.deepcopy(value_model.state_dict())
        batch_terms = []
        batch_utilities = []
        for _ in range(min(args.batch_hands, args.hands - hands_played)):
            state = deal_fixed_limit_holdem(deal_rng)
            if model_player_weights is None:
                model_player = model_players[hands_played % len(model_players)]
            else:
                model_player = model_players[
                    choose_weighted_index(model_player_selector_rng, model_player_weights)
                ]
            opponent_policy = opponent_policies[
                choose_weighted_index(opponent_selector_rng, opponent_policy_weights)
            ]
            action_terms = []
            while not state.is_terminal():
                player = state.current_player()
                if player == model_player:
                    action, log_prob, entropy, value = sample_actor_action(
                        policy_model,
                        value_model,
                        state,
                        feature_encoder,
                    )
                    action_terms.append((log_prob, entropy, value))
                else:
                    action = opponent_policy(state)
                state = state.apply(action)

            reward = state.utility(model_player)
            batch_utilities.append(reward)
            utilities.append(reward)
            utilities_by_model_player[model_player].append(reward)
            for log_prob, entropy, value in action_terms:
                batch_terms.append((log_prob, entropy, value, reward))
            hands_played += 1

        if not batch_terms:
            if checkpoint_selection == "evaluation" and should_run_selection_evaluation(
                args,
                hands_played=hands_played,
                last_evaluated_hands=last_selection_eval_hands,
            ):
                run_selection_evaluation()
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
        if checkpoint_selection == "evaluation" and should_run_selection_evaluation(
            args,
            hands_played=hands_played,
            last_evaluated_hands=last_selection_eval_hands,
        ):
            run_selection_evaluation()

    final_policy_state = copy.deepcopy(policy_model.state_dict())
    final_value_state = copy.deepcopy(value_model.state_dict())
    if checkpoint_selection == "final":
        selected_policy_state = final_policy_state
        selected_value_state = final_value_state
    elif checkpoint_selection == "evaluation":
        selected_policy_state = best_selection_policy_state
        selected_value_state = best_selection_value_state
    else:
        selected_policy_state = best_policy_state
        selected_value_state = best_value_state
    policy_model.load_state_dict(selected_policy_state)
    value_model.load_state_dict(selected_value_state)
    utility_stdev = statistics.stdev(utilities) if len(utilities) > 1 else 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_checkpoint = out_dir / "holdem_policy.pt"
    value_checkpoint = out_dir / "holdem_value.pt"
    final_policy_checkpoint = out_dir / "holdem_policy_final.pt"
    final_value_checkpoint = out_dir / "holdem_value_final.pt"
    save_policy_checkpoint(
        path=policy_checkpoint,
        model_state_dict=policy_model.state_dict(),
        input_dim=input_dim,
        feature_encoder=feature_encoder,
    )
    save_policy_checkpoint(
        path=final_policy_checkpoint,
        model_state_dict=final_policy_state,
        input_dim=input_dim,
        feature_encoder=feature_encoder,
    )
    torch.save(
        {
            "model_state_dict": value_model.state_dict(),
            "input_dim": input_dim,
            **feature_encoder.checkpoint_metadata(),
        },
        value_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": final_value_state,
            "input_dim": input_dim,
            **feature_encoder.checkpoint_metadata(),
        },
        final_value_checkpoint,
    )
    selection_candidate_checkpoint.unlink(missing_ok=True)
    feature_metadata = feature_encoder.checkpoint_metadata()
    metrics: dict[str, Any] = {
        "hands": args.hands,
        "batch_hands": args.batch_hands,
        "model_player": model_player_label(model_players),
        "model_players": list(model_players),
        "model_player_weights": list(model_player_weights) if model_player_weights is not None else None,
        "opponent_policy": args.opponent_policy,
        "opponent_policies": list(opponent_policy_names),
        "opponent_policy_weights": list(opponent_policy_weights),
        "equity_sims": args.equity_sims,
        "lr": args.lr,
        "entropy_coef": args.entropy_coef,
        "value_loss_coef": args.value_loss_coef,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        **feature_metadata,
        "checkpoint_selection": checkpoint_selection,
        "best_batch_avg_utility_model": best_batch_avg_utility,
        "train_avg_utility_model": sum(utilities) / len(utilities) if utilities else 0.0,
        "train_utility_stdev_model": utility_stdev,
        "train_utility_stderr_model": utility_stdev / (len(utilities) ** 0.5) if utilities else 0.0,
        "train_avg_utility_by_model_player": {
            str(player): sum(player_utilities) / len(player_utilities)
            for player, player_utilities in utilities_by_model_player.items()
            if player_utilities
        },
        "checkpoint": str(policy_checkpoint),
        "value_checkpoint": str(value_checkpoint),
        "final_checkpoint": str(final_policy_checkpoint),
        "final_value_checkpoint": str(final_value_checkpoint),
        "seed": args.seed,
    }
    if checkpoint_selection == "evaluation":
        metrics.update(
            {
                **selection_evaluation_metadata(args, model_players),
                "selection_evaluations": selection_evaluations,
                "best_selection_eval_avg_utility_model": best_selection_eval_avg_utility,
                "best_selection_eval_hands_played": best_selection_eval_hands_played,
            }
        )
    eval_metrics = evaluate_trained_policy(args, policy_checkpoint, model_players)
    if eval_metrics is not None:
        metrics["evaluation"] = eval_metrics
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--batch-hands", type=int, default=50)
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--model-player-weights", type=parse_policy_weights)
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="random")
    parser.add_argument("--opponent-policies", type=parse_policy_mix)
    parser.add_argument("--opponent-policy-weights", type=parse_policy_weights)
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("best-batch", "final", "evaluation"),
        default="best-batch",
    )
    parser.add_argument("--selection-eval-hands", type=int, default=0)
    parser.add_argument("--selection-eval-interval-hands", type=int, default=0)
    parser.add_argument("--selection-eval-opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES)
    parser.add_argument("--selection-eval-equity-sims", type=int)
    parser.add_argument("--selection-eval-rollout-sims", type=int)
    parser.add_argument("--selection-eval-model-player", type=parse_model_players)
    parser.add_argument("--selection-eval-jobs", type=int, default=1)
    parser.add_argument("--selection-eval-paired-seats", action="store_true")
    parser.add_argument("--selection-eval-seed", type=int)
    parser.add_argument("--feature-equity-sims", type=int)
    parser.add_argument("--feature-equity-mode", choices=POLICY_FEATURE_EQUITY_MODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-hands", type=int, default=0)
    parser.add_argument("--eval-opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--eval-equity-sims", type=int)
    parser.add_argument("--eval-rollout-sims", type=int)
    parser.add_argument("--eval-model-player", type=parse_model_players)
    parser.add_argument("--eval-jobs", type=int, default=1)
    parser.add_argument("--eval-paired-seats", action="store_true")
    parser.add_argument("--eval-seed", type=int)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
