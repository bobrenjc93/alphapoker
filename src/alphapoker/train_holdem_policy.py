"""Distill the fixed-limit Hold'em equity policy into a neural policy."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from alphapoker.holdem_dataset import (
    HOLDEM_FEATURE_EQUITY_MODES,
    HOLDEM_DATASET_OPPONENT_POLICIES,
    HOLDEM_EXPERT_POLICIES,
    generate_equity_policy_examples,
)
from alphapoker.holdem_dataset import read_policy_examples, write_policy_examples
from alphapoker.holdem_equity_feature import equity_estimator_from_checkpoint
from alphapoker.holdem_features import HOLDEM_CANONICAL_ACTIONS
from alphapoker.train import write_json

CLASS_WEIGHTING_MODES = ("none", "balanced", "sqrt-balanced")


def class_weight_exponent_for_mode(mode: str, exponent: float | None = None) -> float | None:
    if mode not in CLASS_WEIGHTING_MODES:
        raise ValueError(f"Unknown class weighting mode: {mode}")
    if mode == "none":
        if exponent is not None:
            raise ValueError("class weight exponent requires class weighting")
        return None
    resolved = 0.5 if mode == "sqrt-balanced" else 1.0
    if exponent is not None:
        if exponent <= 0.0:
            raise ValueError("class weight exponent must be positive")
        resolved = exponent
    return resolved


def class_weights_from_targets(
    targets,
    n_actions: int,
    mode: str,
    exponent: float | None = None,
):
    resolved_exponent = class_weight_exponent_for_mode(mode, exponent)
    if resolved_exponent is None:
        return None
    import torch

    counts = torch.bincount(targets, minlength=n_actions).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights.pow(resolved_exponent)
    return weights / weights.mean()


def example_weights_from_masks(masks, facing_bet_weight: float):
    if facing_bet_weight <= 0.0:
        raise ValueError("facing bet weight must be positive")
    import torch

    weights = torch.ones(masks.shape[0], dtype=torch.float32, device=masks.device)
    if facing_bet_weight == 1.0:
        return weights
    call_index = HOLDEM_CANONICAL_ACTIONS.index("call")
    fold_index = HOLDEM_CANONICAL_ACTIONS.index("fold")
    facing_bet = masks[:, call_index] & masks[:, fold_index]
    return torch.where(
        facing_bet,
        torch.full_like(weights, facing_bet_weight),
        weights,
    )


def load_policy_checkpoint_state(
    model,
    checkpoint_data: dict[str, Any],
    *,
    target_input_dim: int,
    allow_input_expansion: bool = False,
) -> tuple[int, bool]:
    state_dict = copy.deepcopy(checkpoint_data["model_state_dict"])
    first_layer_key = "net.0.weight"
    init_input_dim = int(
        checkpoint_data.get("input_dim", state_dict[first_layer_key].shape[1])
    )
    if init_input_dim == target_input_dim:
        model.load_state_dict(state_dict)
        return init_input_dim, False
    if init_input_dim > target_input_dim or not allow_input_expansion:
        raise ValueError(
            "init checkpoint input_dim does not match generated feature dimension: "
            f"{init_input_dim} != {target_input_dim}"
        )
    first_layer_weight = state_dict[first_layer_key]
    if first_layer_weight.shape[1] != init_input_dim:
        raise ValueError(
            "init checkpoint first layer width does not match checkpoint input_dim: "
            f"{first_layer_weight.shape[1]} != {init_input_dim}"
        )
    expanded_weight = model.state_dict()[first_layer_key].clone()
    expanded_weight.zero_()
    expanded_weight[:, : first_layer_weight.shape[1]] = first_layer_weight
    state_dict[first_layer_key] = expanded_weight
    model.load_state_dict(state_dict)
    return init_input_dim, True


def _shard_hands(hands: int, jobs: int) -> list[int]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if hands <= 0:
        return []
    shards = min(hands, jobs)
    base = hands // shards
    extra = hands % shards
    return [base + (1 if index < extra else 0) for index in range(shards)]


def _split_train_validation_indices(
    example_count: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")
    indices = list(range(example_count))
    if validation_fraction == 0.0 or example_count <= 1:
        return indices, []

    validation_count = max(1, round(example_count * validation_fraction))
    validation_count = min(validation_count, example_count - 1)
    shuffled = indices.copy()
    random.Random(seed).shuffle(shuffled)
    validation_indices = set(shuffled[:validation_count])
    return (
        [index for index in indices if index not in validation_indices],
        [index for index in indices if index in validation_indices],
    )


def generate_policy_examples_shard(
    index: int,
    *,
    hands: int,
    seed: int,
    equity_sims: int,
    expert_player: int | None,
    expert_policy: str,
    opponent_policy: str,
    rollout_sims: int | None,
    rollout_margin: float,
    feature_equity_sims: int | None,
    feature_equity_mode: str,
    feature_equity_checkpoint: Path | None,
    behavior_checkpoint: Path | None,
    action_history_features: bool,
):
    feature_equity_fn = None
    if feature_equity_checkpoint is not None:
        feature_equity_fn = equity_estimator_from_checkpoint(feature_equity_checkpoint)

    behavior_policy = None
    if behavior_checkpoint is not None:
        from alphapoker.evaluate_holdem_model import model_policy_from_checkpoint

        behavior_policy = model_policy_from_checkpoint(behavior_checkpoint)

    return generate_equity_policy_examples(
        hands=hands,
        seed=seed + index * 1_000_003,
        equity_sims=equity_sims,
        expert_player=expert_player,
        expert_policy=expert_policy,
        opponent_policy=opponent_policy,
        rollout_sims=rollout_sims,
        rollout_margin=rollout_margin,
        feature_equity_sims=feature_equity_sims,
        feature_equity_mode=feature_equity_mode,
        feature_equity_fn=feature_equity_fn,
        expert_behavior_policy=behavior_policy,
        action_history_features=action_history_features,
    )


def generate_policy_training_examples(
    *,
    hands: int,
    seed: int,
    equity_sims: int,
    expert_player: int | None,
    expert_policy: str,
    opponent_policy: str,
    rollout_sims: int | None,
    rollout_margin: float,
    feature_equity_sims: int | None,
    feature_equity_mode: str,
    feature_equity_checkpoint: Path | None,
    behavior_checkpoint: Path | None,
    action_history_features: bool,
    jobs: int,
    progress: bool = False,
):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if jobs == 1:
        examples = generate_policy_examples_shard(
            0,
            hands=hands,
            seed=seed,
            equity_sims=equity_sims,
            expert_player=expert_player,
            expert_policy=expert_policy,
            opponent_policy=opponent_policy,
            rollout_sims=rollout_sims,
            rollout_margin=rollout_margin,
            feature_equity_sims=feature_equity_sims,
            feature_equity_mode=feature_equity_mode,
            feature_equity_checkpoint=feature_equity_checkpoint,
            behavior_checkpoint=behavior_checkpoint,
            action_history_features=action_history_features,
        )
        if progress:
            print(
                f"examples shard 0: hands={hands} examples={len(examples)}",
                file=sys.stderr,
                flush=True,
            )
        return examples

    examples = []
    shard_hands = _shard_hands(hands, jobs)
    if not shard_hands:
        return examples
    with ProcessPoolExecutor(max_workers=min(jobs, len(shard_hands))) as executor:
        future_to_index = {
            executor.submit(
                generate_policy_examples_shard,
                index,
                hands=shard_size,
                seed=seed,
                equity_sims=equity_sims,
                expert_player=expert_player,
                expert_policy=expert_policy,
                opponent_policy=opponent_policy,
                rollout_sims=rollout_sims,
                rollout_margin=rollout_margin,
                feature_equity_sims=feature_equity_sims,
                feature_equity_mode=feature_equity_mode,
                feature_equity_checkpoint=feature_equity_checkpoint,
                behavior_checkpoint=behavior_checkpoint,
                action_history_features=action_history_features,
            ): index
            for index, shard_size in enumerate(shard_hands)
        }
        shard_results = {}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            result = future.result()
            shard_results[index] = result
            if progress:
                print(
                    f"examples shard {index}: hands={shard_hands[index]} "
                    f"examples={len(result)}",
                    file=sys.stderr,
                    flush=True,
                )
    for index in sorted(shard_results):
        examples.extend(shard_results[index])
    return examples


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from alphapoker.holdem_model import HoldemPolicyNet

    if args.feature_equity_sims is not None and args.feature_equity_checkpoint is not None:
        raise ValueError("Set only one of --feature-equity-sims or --feature-equity-checkpoint")
    if args.feature_equity_mode != "random" and args.feature_equity_sims is None:
        raise ValueError("--feature-equity-mode requires --feature-equity-sims")
    if args.feature_equity_checkpoint is not None and args.feature_equity_mode != "random":
        raise ValueError("--feature-equity-mode is only used with --feature-equity-sims")

    examples_in = getattr(args, "examples_in", None)
    examples_out = getattr(args, "examples_out", None)
    jobs = getattr(args, "jobs", 1)
    rollout_margin = float(getattr(args, "rollout_margin", 1.0))
    if examples_in is not None:
        examples = read_policy_examples(examples_in)
    else:
        examples = generate_policy_training_examples(
            hands=args.hands,
            seed=args.seed,
            equity_sims=args.equity_sims,
            expert_player=args.expert_player,
            expert_policy=args.expert_policy,
            opponent_policy=args.opponent_policy,
            rollout_sims=args.rollout_sims,
            rollout_margin=rollout_margin,
            feature_equity_sims=args.feature_equity_sims,
            feature_equity_mode=args.feature_equity_mode,
            feature_equity_checkpoint=args.feature_equity_checkpoint,
            behavior_checkpoint=args.behavior_checkpoint,
            action_history_features=bool(getattr(args, "action_history_features", False)),
            jobs=jobs,
            progress=bool(getattr(args, "progress", False)),
        )
    if examples_out is not None:
        write_policy_examples(examples_out, examples)
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    targets = torch.tensor([example.action_index for example in examples], dtype=torch.long)
    masks = torch.tensor([example.legal_mask for example in examples], dtype=torch.bool)
    validation_fraction = getattr(args, "validation_fraction", 0.0)
    train_indices, validation_indices = _split_train_validation_indices(
        len(examples),
        validation_fraction,
        args.seed,
    )
    train_features = features[train_indices]
    train_targets = targets[train_indices]
    train_masks = masks[train_indices]
    validation_features = features[validation_indices] if validation_indices else None
    validation_targets = targets[validation_indices] if validation_indices else None
    validation_masks = masks[validation_indices] if validation_indices else None
    facing_bet_weight = float(getattr(args, "facing_bet_weight", 1.0))
    example_weights = example_weights_from_masks(masks, facing_bet_weight)
    use_example_weights = facing_bet_weight != 1.0
    train_example_weights = example_weights[train_indices] if use_example_weights else None
    validation_example_weights = (
        example_weights[validation_indices]
        if use_example_weights and validation_indices
        else None
    )
    call_index = HOLDEM_CANONICAL_ACTIONS.index("call")
    fold_index = HOLDEM_CANONICAL_ACTIONS.index("fold")
    facing_bet_mask = masks[:, call_index] & masks[:, fold_index]

    torch.manual_seed(0)
    model = HoldemPolicyNet(input_dim=features.shape[1])
    init_checkpoint = getattr(args, "init_checkpoint", None)
    init_kl_weight = float(getattr(args, "init_kl_weight", 0.0))
    init_allow_input_expansion = bool(getattr(args, "init_allow_input_expansion", False))
    init_input_dim = None
    init_input_expanded = False
    if init_kl_weight < 0.0:
        raise ValueError("--init-kl-weight must be non-negative")
    if init_kl_weight > 0.0 and init_checkpoint is None:
        raise ValueError("--init-kl-weight requires --init-checkpoint")
    if init_checkpoint is not None:
        init_data = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        init_input_dim, init_input_expanded = load_policy_checkpoint_state(
            model,
            init_data,
            target_input_dim=features.shape[1],
            allow_input_expansion=init_allow_input_expansion,
        )
    anchor_model = None
    if init_kl_weight > 0.0:
        anchor_model = copy.deepcopy(model)
        anchor_model.eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weighting = getattr(args, "class_weighting", "none")
    class_weight_exponent = getattr(args, "class_weight_exponent", None)
    class_weights = class_weights_from_targets(
        train_targets,
        len(HOLDEM_CANONICAL_ACTIONS),
        class_weighting,
        class_weight_exponent,
    )
    resolved_class_weight_exponent = class_weight_exponent_for_mode(
        class_weighting,
        class_weight_exponent,
    )

    def loss_for(batch_features, batch_targets, batch_masks, batch_weights=None):
        logits = model(batch_features)
        masked_logits = logits.masked_fill(~batch_masks, -1e9)
        if batch_weights is None:
            loss = F.cross_entropy(masked_logits, batch_targets, weight=class_weights)
        else:
            losses = F.cross_entropy(
                masked_logits,
                batch_targets,
                weight=class_weights,
                reduction="none",
            )
            loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)
        if anchor_model is None:
            return loss
        with torch.no_grad():
            anchor_logits = anchor_model(batch_features).masked_fill(~batch_masks, -1e9)
            anchor_probs = F.softmax(anchor_logits, dim=1)
        if batch_weights is None:
            policy_kl = F.kl_div(
                F.log_softmax(masked_logits, dim=1),
                anchor_probs,
                reduction="batchmean",
            )
        else:
            kl_losses = F.kl_div(
                F.log_softmax(masked_logits, dim=1),
                anchor_probs,
                reduction="none",
            ).sum(dim=1)
            policy_kl = (
                (kl_losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)
            )
        return loss + init_kl_weight * policy_kl

    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    final_loss = 0.0
    final_validation_loss: float | None = None
    best_train_loss = float("inf")
    best_validation_loss: float | None = None
    for epoch in range(args.epochs):
        loss = loss_for(train_features, train_targets, train_masks, train_example_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss = loss_for(
                train_features,
                train_targets,
                train_masks,
                train_example_weights,
            )
            final_loss = float(train_loss.detach().cpu())
            validation_loss_value = None
            selection_loss = final_loss
            if validation_features is not None:
                validation_loss = loss_for(
                    validation_features,
                    validation_targets,
                    validation_masks,
                    validation_example_weights,
                )
                validation_loss_value = float(validation_loss.detach().cpu())
                final_validation_loss = validation_loss_value
                selection_loss = validation_loss_value

        if selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch + 1
            best_train_loss = final_loss
            best_validation_loss = validation_loss_value
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    with torch.no_grad():
        best_logits = model(features).masked_fill(~masks, -1e9)
        predictions = best_logits.argmax(dim=1)
        overall_accuracy = float((predictions == targets).float().mean().cpu())
        train_logits = model(train_features).masked_fill(~train_masks, -1e9)
        train_predictions = train_logits.argmax(dim=1)
        train_accuracy = float((train_predictions == train_targets).float().mean().cpu())
        validation_accuracy = None
        if validation_features is not None:
            validation_logits = model(validation_features).masked_fill(~validation_masks, -1e9)
            validation_predictions = validation_logits.argmax(dim=1)
            validation_accuracy = float(
                (validation_predictions == validation_targets).float().mean().cpu()
            )
    target_action_counts = {
        action: int((targets == index).sum().item())
        for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
    }
    predicted_action_counts = {
        action: int((predictions == index).sum().item())
        for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "holdem_policy.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "canonical_actions": list(HOLDEM_CANONICAL_ACTIONS),
            "input_dim": features.shape[1],
            "feature_equity_sims": args.feature_equity_sims,
            "feature_equity_mode": (
                args.feature_equity_mode if args.feature_equity_sims is not None else None
            ),
            "feature_equity_checkpoint": (
                str(args.feature_equity_checkpoint)
                if args.feature_equity_checkpoint is not None
                else None
            ),
            "action_history_features": bool(getattr(args, "action_history_features", False)),
        },
        checkpoint,
    )
    metrics: dict[str, Any] = {
        "hands": args.hands,
        "examples": len(examples),
        "equity_sims": args.equity_sims,
        "expert_player": args.expert_player,
        "expert_policy": args.expert_policy,
        "opponent_policy": args.opponent_policy,
        "rollout_sims": args.rollout_sims,
        "rollout_margin": rollout_margin,
        "feature_equity_sims": args.feature_equity_sims,
        "feature_equity_mode": (
            args.feature_equity_mode if args.feature_equity_sims is not None else None
        ),
        "feature_equity_checkpoint": (
            str(args.feature_equity_checkpoint)
            if args.feature_equity_checkpoint is not None
            else None
        ),
        "action_history_features": bool(getattr(args, "action_history_features", False)),
        "jobs": jobs,
        "epochs": args.epochs,
        "lr": args.lr,
        "init_kl_weight": init_kl_weight,
        "class_weighting": class_weighting,
        "class_weight_exponent": resolved_class_weight_exponent,
        "facing_bet_weight": facing_bet_weight,
        "facing_bet_examples": int(facing_bet_mask.sum().item()),
        "facing_bet_train_examples": int(facing_bet_mask[train_indices].sum().item()),
        "facing_bet_validation_examples": int(
            facing_bet_mask[validation_indices].sum().item()
        )
        if validation_indices
        else 0,
        "validation_fraction": validation_fraction,
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "selection_metric": "validation_loss" if validation_indices else "train_loss",
        "best_epoch": best_epoch,
        "final_loss": final_loss,
        "final_train_loss": final_loss,
        "final_validation_loss": final_validation_loss,
        "best_loss": best_loss,
        "best_train_loss": best_train_loss,
        "best_validation_loss": best_validation_loss,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "overall_accuracy": overall_accuracy,
        "target_action_counts": target_action_counts,
        "predicted_action_counts": predicted_action_counts,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
    }
    if args.behavior_checkpoint is not None:
        metrics["behavior_checkpoint"] = str(args.behavior_checkpoint)
    if init_checkpoint is not None:
        metrics["init_checkpoint"] = str(init_checkpoint)
        metrics["init_input_dim"] = init_input_dim
        metrics["init_input_expanded"] = init_input_expanded
    if examples_in is not None:
        metrics["examples_in"] = str(examples_in)
    if examples_out is not None:
        metrics["examples_out"] = str(examples_out)
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--expert-player", type=int, choices=[0, 1])
    parser.add_argument("--expert-policy", choices=HOLDEM_EXPERT_POLICIES, default="equity")
    parser.add_argument(
        "--opponent-policy",
        choices=HOLDEM_DATASET_OPPONENT_POLICIES,
        default="equity",
    )
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--rollout-margin", type=float, default=1.0)
    parser.add_argument("--feature-equity-sims", type=int)
    parser.add_argument(
        "--feature-equity-mode",
        choices=HOLDEM_FEATURE_EQUITY_MODES,
        default="random",
    )
    parser.add_argument("--feature-equity-checkpoint", type=Path)
    parser.add_argument("--action-history-features", action="store_true")
    parser.add_argument("--behavior-checkpoint", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--init-kl-weight", type=float, default=0.0)
    parser.add_argument("--init-allow-input-expansion", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--class-weighting", choices=CLASS_WEIGHTING_MODES, default="none")
    parser.add_argument("--class-weight-exponent", type=float)
    parser.add_argument("--facing-bet-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-in", type=Path)
    parser.add_argument("--examples-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
