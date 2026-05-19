"""Train a Hold'em policy with REINFORCE against a fixed opponent."""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
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


def parse_policy_mix(value: str) -> tuple[str, ...]:
    policies = tuple(item.strip() for item in value.split(",") if item.strip())
    if not policies:
        raise argparse.ArgumentTypeError("at least one opponent policy is required")
    unknown = [policy for policy in policies if policy not in HOLDEM_SELF_PLAY_POLICIES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown opponent policy: {unknown[0]}")
    return policies


def parse_policy_weights(value: str) -> tuple[float, ...]:
    try:
        weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("opponent policy weights must be numeric") from error
    if not weights:
        raise argparse.ArgumentTypeError("at least one opponent policy weight is required")
    if any(weight < 0.0 for weight in weights):
        raise argparse.ArgumentTypeError("opponent policy weights must be non-negative")
    if sum(weights) <= 0.0:
        raise argparse.ArgumentTypeError("at least one opponent policy weight must be positive")
    return weights


def choose_weighted_index(rng: random.Random, weights: tuple[float, ...]) -> int:
    draw = rng.random() * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if draw <= cumulative:
            return index
    return len(weights) - 1


def parse_model_players(value: str) -> tuple[int, ...]:
    if value == "both":
        return (0, 1)
    try:
        model_player = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("model-player must be 0, 1, or both") from error
    if model_player not in (0, 1):
        raise argparse.ArgumentTypeError("model-player must be 0, 1, or both")
    return (model_player,)


def normalize_model_players(value: int | str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return parse_model_players(str(value))


def model_player_label(model_players: tuple[int, ...]) -> int | str:
    return model_players[0] if len(model_players) == 1 else "both"


def opponent_policy_label(opponent_policies: tuple[str, ...]) -> str:
    return opponent_policies[0] if len(opponent_policies) == 1 else ",".join(opponent_policies)


def evaluate_trained_policy(
    args: argparse.Namespace,
    checkpoint: Path,
    model_players: tuple[int, ...],
) -> dict[str, Any] | None:
    eval_hands = getattr(args, "eval_hands", 0)
    if eval_hands <= 0:
        return None

    from alphapoker.evaluate_holdem_model import run as evaluate_model

    eval_model_players = getattr(args, "eval_model_player", None) or model_players
    eval_seed = getattr(args, "eval_seed", None)
    eval_equity_sims = getattr(args, "eval_equity_sims", None)
    eval_rollout_margin = getattr(args, "eval_rollout_margin", None)
    return evaluate_model(
        argparse.Namespace(
            checkpoint=checkpoint,
            hands=eval_hands,
            seed=eval_seed if eval_seed is not None else args.seed + 10_000,
            opponent_policy=getattr(args, "eval_opponent_policy", "pot-odds"),
            equity_sims=eval_equity_sims if eval_equity_sims is not None else args.equity_sims,
            rollout_sims=getattr(args, "eval_rollout_sims", None),
            rollout_margin=(
                eval_rollout_margin
                if eval_rollout_margin is not None
                else getattr(args, "rollout_margin", 1.0)
            ),
            model_player=eval_model_players,
            jobs=getattr(args, "eval_jobs", 1),
            paired_seats=getattr(args, "eval_paired_seats", False),
            out=None,
        )
    )


def save_policy_checkpoint(
    *,
    path: Path,
    model_state_dict: dict[str, Any],
    input_dim: int,
    feature_encoder: HoldemPolicyFeatureEncoder,
) -> None:
    import torch

    torch.save(
        {
            "model_state_dict": model_state_dict,
            "canonical_actions": list(HOLDEM_CANONICAL_ACTIONS),
            "input_dim": input_dim,
            **feature_encoder.checkpoint_metadata(),
        },
        path,
    )


def validate_checkpoint_selection_args(args: argparse.Namespace, checkpoint_selection: str) -> None:
    selection_eval_hands = getattr(args, "selection_eval_hands", 0)
    selection_eval_interval_hands = getattr(args, "selection_eval_interval_hands", 0)
    selection_eval_jobs = getattr(args, "selection_eval_jobs", 1)
    selection_eval_opponent_policies = getattr(args, "selection_eval_opponent_policies", None)
    selection_eval_opponent_policy_weights = getattr(
        args,
        "selection_eval_opponent_policy_weights",
        None,
    )
    if checkpoint_selection == "evaluation":
        if selection_eval_hands <= 0:
            raise ValueError(
                "--checkpoint-selection evaluation requires --selection-eval-hands > 0"
            )
    elif selection_eval_hands > 0:
        raise ValueError("--selection-eval-hands requires --checkpoint-selection evaluation")
    if selection_eval_interval_hands < 0:
        raise ValueError("--selection-eval-interval-hands must be non-negative")
    if selection_eval_jobs < 1:
        raise ValueError("--selection-eval-jobs must be positive")
    if (
        selection_eval_opponent_policies is not None
        and getattr(args, "selection_eval_opponent_policy", None) is not None
    ):
        raise ValueError(
            "Set only one of --selection-eval-opponent-policy or "
            "--selection-eval-opponent-policies"
        )
    if selection_eval_opponent_policy_weights is not None:
        if selection_eval_opponent_policies is None:
            raise ValueError(
                "--selection-eval-opponent-policy-weights requires "
                "--selection-eval-opponent-policies"
            )
        if len(selection_eval_opponent_policy_weights) != len(selection_eval_opponent_policies):
            raise ValueError(
                "selection eval opponent policy weights must match selection eval opponents"
            )


def should_run_selection_evaluation(
    args: argparse.Namespace,
    *,
    hands_played: int,
    last_evaluated_hands: int | None,
) -> bool:
    if last_evaluated_hands == hands_played:
        return False
    if hands_played >= args.hands:
        return True
    interval_hands = getattr(args, "selection_eval_interval_hands", 0)
    if interval_hands <= 0 or last_evaluated_hands is None:
        return False
    return hands_played - last_evaluated_hands >= interval_hands


def resolve_selection_eval_opponent_policies(args: argparse.Namespace) -> tuple[str, ...]:
    opponent_policies = getattr(args, "selection_eval_opponent_policies", None)
    if opponent_policies is not None:
        return opponent_policies
    return (
        getattr(args, "selection_eval_opponent_policy", None)
        or getattr(args, "eval_opponent_policy", "pot-odds"),
    )


def resolve_selection_eval_opponent_weights(
    args: argparse.Namespace,
    opponent_policies: tuple[str, ...],
) -> tuple[float, ...]:
    weights = getattr(args, "selection_eval_opponent_policy_weights", None)
    if weights is None:
        return tuple(1.0 for _ in opponent_policies)
    return weights


def aggregate_selection_evaluation_metrics(
    *,
    component_metrics: list[dict[str, Any]],
    opponent_policies: tuple[str, ...],
    opponent_policy_weights: tuple[float, ...],
    aggregation: str,
) -> dict[str, Any]:
    if len(component_metrics) == 1:
        metrics = dict(component_metrics[0])
        metrics["selection_eval_score_model"] = metrics["avg_utility_model"]
        metrics["selection_eval_score_stderr_model"] = metrics["utility_stderr_model"]
        metrics["selection_eval_aggregation"] = aggregation
        metrics["opponent_policies"] = list(opponent_policies)
        metrics["opponent_policy_weights"] = list(opponent_policy_weights)
        return metrics

    total_weight = sum(opponent_policy_weights)
    normalized_weights = [weight / total_weight for weight in opponent_policy_weights]
    weighted_avg = sum(
        weight * float(metrics["avg_utility_model"])
        for weight, metrics in zip(normalized_weights, component_metrics, strict=True)
    )
    weighted_stderr = sum(
        (weight * float(metrics["utility_stderr_model"])) ** 2
        for weight, metrics in zip(normalized_weights, component_metrics, strict=True)
    ) ** 0.5
    min_index = min(
        range(len(component_metrics)),
        key=lambda index: float(component_metrics[index]["avg_utility_model"]),
    )
    if aggregation == "min":
        score = float(component_metrics[min_index]["avg_utility_model"])
        score_stderr = float(component_metrics[min_index]["utility_stderr_model"])
    else:
        score = weighted_avg
        score_stderr = weighted_stderr
    paired_deals = [
        metrics["paired_deals"]
        for metrics in component_metrics
        if metrics.get("paired_deals") is not None
    ]
    return {
        "avg_utility_model": weighted_avg,
        "utility_stderr_model": weighted_stderr,
        "selection_eval_score_model": score,
        "selection_eval_score_stderr_model": score_stderr,
        "selection_eval_aggregation": aggregation,
        "selection_eval_min_opponent_policy": component_metrics[min_index]["opponent_policy"],
        "opponent_policy": opponent_policy_label(opponent_policies),
        "opponent_policies": list(opponent_policies),
        "opponent_policy_weights": list(opponent_policy_weights),
        "hands": sum(int(metrics["hands"]) for metrics in component_metrics),
        "paired_deals": sum(int(value) for value in paired_deals) if paired_deals else None,
        "seed": component_metrics[0]["seed"],
        "paired_seats": component_metrics[0].get("paired_seats", False),
        "rollout_sims": component_metrics[0].get("rollout_sims"),
        "rollout_margin": component_metrics[0].get("rollout_margin"),
        "selection_eval_components": component_metrics,
    }


def evaluate_selection_checkpoint(
    args: argparse.Namespace,
    checkpoint: Path,
    model_players: tuple[int, ...],
) -> dict[str, Any]:
    from alphapoker.evaluate_holdem_model import run as evaluate_model

    eval_model_players = (
        getattr(args, "selection_eval_model_player", None)
        or getattr(args, "eval_model_player", None)
        or model_players
    )
    eval_opponent_policies = resolve_selection_eval_opponent_policies(args)
    eval_opponent_policy_weights = resolve_selection_eval_opponent_weights(
        args,
        eval_opponent_policies,
    )
    eval_aggregation = getattr(args, "selection_eval_aggregation", "mean")
    eval_equity_sims = getattr(args, "selection_eval_equity_sims", None)
    if eval_equity_sims is None:
        eval_equity_sims = getattr(args, "eval_equity_sims", None)
    eval_rollout_sims = getattr(args, "selection_eval_rollout_sims", None)
    if eval_rollout_sims is None:
        eval_rollout_sims = getattr(args, "eval_rollout_sims", None)
    eval_rollout_margin = getattr(args, "selection_eval_rollout_margin", None)
    if eval_rollout_margin is None:
        eval_rollout_margin = getattr(args, "eval_rollout_margin", None)
    if eval_rollout_margin is None:
        eval_rollout_margin = getattr(args, "rollout_margin", 1.0)
    eval_seed = getattr(args, "selection_eval_seed", None)
    component_metrics = [
        evaluate_model(
            argparse.Namespace(
                checkpoint=checkpoint,
                hands=getattr(args, "selection_eval_hands"),
                seed=eval_seed if eval_seed is not None else args.seed + 20_000,
                opponent_policy=opponent_policy,
                equity_sims=eval_equity_sims if eval_equity_sims is not None else args.equity_sims,
                rollout_sims=eval_rollout_sims,
                rollout_margin=eval_rollout_margin,
                model_player=eval_model_players,
                jobs=getattr(args, "selection_eval_jobs", 1),
                paired_seats=getattr(args, "selection_eval_paired_seats", False),
                out=None,
            )
        )
        for opponent_policy in eval_opponent_policies
    ]
    return aggregate_selection_evaluation_metrics(
        component_metrics=component_metrics,
        opponent_policies=eval_opponent_policies,
        opponent_policy_weights=eval_opponent_policy_weights,
        aggregation=eval_aggregation,
    )


def compact_selection_evaluation(
    *,
    hands_played: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    compact = {
        "hands_played": hands_played,
        "avg_utility_model": metrics["avg_utility_model"],
        "utility_stderr_model": metrics["utility_stderr_model"],
        "opponent_policy": metrics["opponent_policy"],
        "hands": metrics["hands"],
        "seed": metrics["seed"],
        "paired_seats": metrics.get("paired_seats", False),
    }
    if "paired_deals" in metrics:
        compact["paired_deals"] = metrics["paired_deals"]
    if "rollout_sims" in metrics:
        compact["rollout_sims"] = metrics["rollout_sims"]
    if "rollout_margin" in metrics:
        compact["rollout_margin"] = metrics["rollout_margin"]
    if "selection_eval_score_model" in metrics:
        compact["selection_eval_score_model"] = metrics["selection_eval_score_model"]
    if "selection_eval_score_stderr_model" in metrics:
        compact["selection_eval_score_stderr_model"] = metrics["selection_eval_score_stderr_model"]
    if "selection_eval_aggregation" in metrics:
        compact["selection_eval_aggregation"] = metrics["selection_eval_aggregation"]
    if "selection_eval_min_opponent_policy" in metrics:
        compact["selection_eval_min_opponent_policy"] = metrics[
            "selection_eval_min_opponent_policy"
        ]
    if "opponent_policies" in metrics:
        compact["opponent_policies"] = metrics["opponent_policies"]
    if "opponent_policy_weights" in metrics:
        compact["opponent_policy_weights"] = metrics["opponent_policy_weights"]
    if "selection_eval_components" in metrics:
        compact["selection_eval_components"] = [
            {
                "opponent_policy": component["opponent_policy"],
                "avg_utility_model": component["avg_utility_model"],
                "utility_stderr_model": component["utility_stderr_model"],
                "hands": component["hands"],
                "paired_deals": component.get("paired_deals"),
            }
            for component in metrics["selection_eval_components"]
        ]
    return compact


def selection_evaluation_metadata(
    args: argparse.Namespace,
    model_players: tuple[int, ...],
) -> dict[str, Any]:
    eval_opponent_policies = resolve_selection_eval_opponent_policies(args)
    eval_opponent_policy_weights = resolve_selection_eval_opponent_weights(
        args,
        eval_opponent_policies,
    )
    eval_equity_sims = getattr(args, "selection_eval_equity_sims", None)
    if eval_equity_sims is None:
        eval_equity_sims = getattr(args, "eval_equity_sims", None)
    eval_rollout_sims = getattr(args, "selection_eval_rollout_sims", None)
    if eval_rollout_sims is None:
        eval_rollout_sims = getattr(args, "eval_rollout_sims", None)
    eval_rollout_margin = getattr(args, "selection_eval_rollout_margin", None)
    if eval_rollout_margin is None:
        eval_rollout_margin = getattr(args, "eval_rollout_margin", None)
    if eval_rollout_margin is None:
        eval_rollout_margin = getattr(args, "rollout_margin", 1.0)
    eval_model_players = (
        getattr(args, "selection_eval_model_player", None)
        or getattr(args, "eval_model_player", None)
        or model_players
    )
    eval_seed = getattr(args, "selection_eval_seed", None)
    return {
        "selection_eval_hands": getattr(args, "selection_eval_hands", 0),
        "selection_eval_interval_hands": getattr(args, "selection_eval_interval_hands", 0),
        "selection_eval_opponent_policy": opponent_policy_label(eval_opponent_policies),
        "selection_eval_opponent_policies": list(eval_opponent_policies),
        "selection_eval_opponent_policy_weights": list(eval_opponent_policy_weights),
        "selection_eval_aggregation": getattr(args, "selection_eval_aggregation", "mean"),
        "selection_eval_equity_sims": (
            eval_equity_sims if eval_equity_sims is not None else args.equity_sims
        ),
        "selection_eval_rollout_sims": eval_rollout_sims,
        "selection_eval_rollout_margin": eval_rollout_margin,
        "selection_eval_model_player": model_player_label(eval_model_players),
        "selection_eval_jobs": getattr(args, "selection_eval_jobs", 1),
        "selection_eval_paired_seats": getattr(args, "selection_eval_paired_seats", False),
        "selection_eval_seed": eval_seed if eval_seed is not None else args.seed + 20_000,
    }


def report_training_progress(
    enabled: bool,
    *,
    hands_played: int,
    batch_avg_utility: float,
    train_avg_utility: float,
) -> None:
    if not enabled:
        return
    print(
        f"hands={hands_played} "
        f"batch_avg_utility_model={batch_avg_utility:.3f} "
        f"train_avg_utility_model={train_avg_utility:.3f}",
        file=sys.stderr,
        flush=True,
    )


def sample_model_action(
    model,
    state: FixedLimitHoldemState,
    feature_encoder: HoldemPolicyFeatureEncoder,
):
    import torch

    features = torch.tensor([feature_encoder.encode(state)], dtype=torch.float32)
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
    progress = bool(getattr(args, "progress", False))
    checkpoint_selection = getattr(args, "checkpoint_selection", "best-batch")
    validate_checkpoint_selection_args(args, checkpoint_selection)
    model_players = normalize_model_players(args.model_player)
    model_player_weights = getattr(args, "model_player_weights", None)
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
    opponent_policy_weights = getattr(args, "opponent_policy_weights", None)
    if opponent_policy_weights is None:
        opponent_policy_weights = tuple(1.0 for _ in opponent_policy_names)
    if len(opponent_policy_weights) != len(opponent_policy_names):
        raise ValueError("opponent policy weights must match opponent policies")
    rollout_sims = getattr(args, "rollout_sims", None)
    rollout_margin = float(getattr(args, "rollout_margin", 1.0))
    opponent_policies = [
        make_policy(
            name,
            random.Random(args.seed + 100 + index),
            args.equity_sims,
            rollout_sims,
            rollout_margin,
        )
        for index, name in enumerate(opponent_policy_names)
    ]

    model = HoldemPolicyNet(input_dim=input_dim)
    if init_checkpoint_data is not None:
        model.load_state_dict(init_checkpoint_data["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    utilities: list[float] = []
    utilities_by_model_player: dict[int, list[float]] = {player: [] for player in model_players}
    best_batch_avg_utility = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    best_selection_eval_avg_utility = float("-inf")
    best_selection_eval_hands_played: int | None = None
    best_selection_state = copy.deepcopy(model.state_dict())
    selection_evaluations: list[dict[str, Any]] = []
    last_selection_eval_hands: int | None = None
    out_dir = Path(args.out)
    selection_candidate_checkpoint = out_dir / "holdem_policy_selection_candidate.pt"

    def run_selection_evaluation() -> None:
        nonlocal best_selection_eval_avg_utility
        nonlocal best_selection_eval_hands_played
        nonlocal best_selection_state
        nonlocal last_selection_eval_hands
        out_dir.mkdir(parents=True, exist_ok=True)
        save_policy_checkpoint(
            path=selection_candidate_checkpoint,
            model_state_dict=model.state_dict(),
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
        selection_score = float(
            summary.get("selection_eval_score_model", summary["avg_utility_model"])
        )
        if selection_score > best_selection_eval_avg_utility:
            best_selection_eval_avg_utility = selection_score
            best_selection_eval_hands_played = hands_played
            best_selection_state = copy.deepcopy(model.state_dict())
        last_selection_eval_hands = hands_played

    hands_played = 0
    if checkpoint_selection == "evaluation":
        run_selection_evaluation()
    while hands_played < args.hands:
        state_before_batch = copy.deepcopy(model.state_dict())
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
            log_probs = []
            entropies = []
            while not state.is_terminal():
                player = state.current_player()
                if player == model_player:
                    action, log_prob, entropy = sample_model_action(
                        model,
                        state,
                        feature_encoder,
                    )
                    log_probs.append(log_prob)
                    entropies.append(entropy)
                else:
                    action = opponent_policy(state)
                state = state.apply(action)

            reward = state.utility(model_player)
            batch_utilities.append(reward)
            utilities.append(reward)
            utilities_by_model_player[model_player].append(reward)
            if log_probs:
                batch_terms.append((torch.stack(log_probs).sum(), torch.stack(entropies).sum(), reward))
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
            best_state = state_before_batch
        report_training_progress(
            progress,
            hands_played=hands_played,
            batch_avg_utility=baseline,
            train_avg_utility=sum(utilities) / len(utilities),
        )
        losses = [
            -log_prob_sum * (reward - baseline) - args.entropy_coef * entropy_sum
            for log_prob_sum, entropy_sum, reward in batch_terms
        ]
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

    final_state = copy.deepcopy(model.state_dict())
    if checkpoint_selection == "final":
        selected_state = final_state
    elif checkpoint_selection == "evaluation":
        selected_state = best_selection_state
    else:
        selected_state = best_state
    model.load_state_dict(selected_state)
    utility_stdev = statistics.stdev(utilities) if len(utilities) > 1 else 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "holdem_policy.pt"
    final_checkpoint = out_dir / "holdem_policy_final.pt"
    save_policy_checkpoint(
        path=checkpoint,
        model_state_dict=model.state_dict(),
        input_dim=input_dim,
        feature_encoder=feature_encoder,
    )
    save_policy_checkpoint(
        path=final_checkpoint,
        model_state_dict=final_state,
        input_dim=input_dim,
        feature_encoder=feature_encoder,
    )
    selection_candidate_checkpoint.unlink(missing_ok=True)
    feature_metadata = feature_encoder.checkpoint_metadata()
    metrics: dict[str, Any] = {
        "hands": args.hands,
        "batch_hands": args.batch_hands,
        "model_player": model_player_label(model_players),
        "model_players": list(model_players),
        "model_player_weights": list(model_player_weights) if model_player_weights is not None else None,
        "opponent_policy": opponent_policy_label(opponent_policy_names),
        "opponent_policies": list(opponent_policy_names),
        "opponent_policy_weights": list(opponent_policy_weights),
        "equity_sims": args.equity_sims,
        "rollout_sims": rollout_sims,
        "rollout_margin": rollout_margin,
        "lr": args.lr,
        "entropy_coef": args.entropy_coef,
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
        "checkpoint": str(checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "seed": args.seed,
    }
    if checkpoint_selection == "evaluation":
        metrics.update(
            {
                **selection_evaluation_metadata(args, model_players),
                "selection_evaluations": selection_evaluations,
                "best_selection_eval_score_model": best_selection_eval_avg_utility,
                "best_selection_eval_avg_utility_model": best_selection_eval_avg_utility,
                "best_selection_eval_hands_played": best_selection_eval_hands_played,
            }
        )
    eval_metrics = evaluate_trained_policy(args, checkpoint, model_players)
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
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--rollout-margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("best-batch", "final", "evaluation"),
        default="best-batch",
    )
    parser.add_argument("--selection-eval-hands", type=int, default=0)
    parser.add_argument("--selection-eval-interval-hands", type=int, default=0)
    parser.add_argument("--selection-eval-opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES)
    parser.add_argument("--selection-eval-opponent-policies", type=parse_policy_mix)
    parser.add_argument("--selection-eval-opponent-policy-weights", type=parse_policy_weights)
    parser.add_argument(
        "--selection-eval-aggregation",
        choices=("mean", "min"),
        default="mean",
    )
    parser.add_argument("--selection-eval-equity-sims", type=int)
    parser.add_argument("--selection-eval-rollout-sims", type=int)
    parser.add_argument("--selection-eval-rollout-margin", type=float)
    parser.add_argument("--selection-eval-model-player", type=parse_model_players)
    parser.add_argument("--selection-eval-jobs", type=int, default=1)
    parser.add_argument("--selection-eval-paired-seats", action="store_true")
    parser.add_argument("--selection-eval-seed", type=int)
    parser.add_argument("--feature-equity-sims", type=int)
    parser.add_argument("--feature-equity-mode", choices=POLICY_FEATURE_EQUITY_MODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--eval-hands", type=int, default=0)
    parser.add_argument("--eval-opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--eval-equity-sims", type=int)
    parser.add_argument("--eval-rollout-sims", type=int)
    parser.add_argument("--eval-rollout-margin", type=float)
    parser.add_argument("--eval-model-player", type=parse_model_players)
    parser.add_argument("--eval-jobs", type=int, default=1)
    parser.add_argument("--eval-paired-seats", action="store_true")
    parser.add_argument("--eval-seed", type=int)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
