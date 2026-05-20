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
from alphapoker.holdem_features import (
    HOLDEM_ACTION_HISTORY_FEATURE_DIM,
    HOLDEM_CANONICAL_ACTIONS,
    HOLDEM_FEATURE_DIM,
    HOLDEM_PLAYER_FEATURE_DIM,
    HOLDEM_PLAYER_FEATURE_OFFSET,
)
from alphapoker.holdem_mccfr import (
    HOLDEM_MCCFR_STRATEGY_MODES,
    HOLDEM_MCCFR_SUPPORT_MODES,
)
from alphapoker.train import write_json

CLASS_WEIGHTING_MODES = ("none", "balanced", "sqrt-balanced")


def action_weight_overrides_from_specs(specs: list[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError("action weight must use ACTION=WEIGHT")
        action, value_text = spec.split("=", 1)
        if action not in HOLDEM_CANONICAL_ACTIONS:
            raise ValueError(f"unknown Hold'em action for weight: {action}")
        try:
            value = float(value_text)
        except ValueError as error:
            raise ValueError(f"invalid weight for action {action}: {value_text}") from error
        if value <= 0.0:
            raise ValueError("action weight must be positive")
        overrides[action] = value
    return overrides


def apply_action_weight_overrides(class_weights, overrides: dict[str, float]):
    if not overrides:
        return class_weights
    import torch

    if class_weights is None:
        weights = torch.ones(len(HOLDEM_CANONICAL_ACTIONS), dtype=torch.float32)
    else:
        weights = class_weights.clone()
    for action, weight in overrides.items():
        weights[HOLDEM_CANONICAL_ACTIONS.index(action)] *= weight
    return weights / weights.mean()


def player_action_weight_overrides_from_specs(
    specs: list[str] | None,
) -> dict[tuple[int, str], float]:
    overrides: dict[tuple[int, str], float] = {}
    for spec in specs or []:
        if ":" not in spec or "=" not in spec:
            raise ValueError("player action weight must use PLAYER:ACTION=WEIGHT")
        player_text, rest = spec.split(":", 1)
        action, value_text = rest.split("=", 1)
        try:
            player = int(player_text)
        except ValueError as error:
            raise ValueError(f"invalid player for action weight: {player_text}") from error
        if player not in (0, 1):
            raise ValueError("player action weight player must be 0 or 1")
        if action not in HOLDEM_CANONICAL_ACTIONS:
            raise ValueError(f"unknown Hold'em action for player weight: {action}")
        try:
            value = float(value_text)
        except ValueError as error:
            raise ValueError(
                f"invalid weight for player {player} action {action}: {value_text}"
            ) from error
        if value <= 0.0:
            raise ValueError("player action weight must be positive")
        overrides[(player, action)] = value
    return overrides


def player_value_weight_overrides_from_specs(
    specs: list[str] | None,
) -> dict[int, float]:
    overrides: dict[int, float] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError("player value weight must use PLAYER=WEIGHT")
        player_text, value_text = spec.split("=", 1)
        try:
            player = int(player_text)
        except ValueError as error:
            raise ValueError(f"invalid player for value weight: {player_text}") from error
        if player not in (0, 1):
            raise ValueError("player value weight player must be 0 or 1")
        try:
            value = float(value_text)
        except ValueError as error:
            raise ValueError(
                f"invalid weight for player {player}: {value_text}"
            ) from error
        if value <= 0.0:
            raise ValueError("player value weight must be positive")
        overrides[player] = value
    return overrides


def player_indices_from_features(features):
    if features.shape[1] < HOLDEM_PLAYER_FEATURE_OFFSET + HOLDEM_PLAYER_FEATURE_DIM:
        raise ValueError("player action weights require full Hold'em state features")
    return features[
        :,
        HOLDEM_PLAYER_FEATURE_OFFSET : HOLDEM_PLAYER_FEATURE_OFFSET
        + HOLDEM_PLAYER_FEATURE_DIM,
    ].argmax(dim=1)


def player_action_weights_from_features_targets(
    features,
    targets,
    overrides: dict[tuple[int, str], float],
):
    import torch

    weights = torch.ones(targets.shape[0], dtype=torch.float32, device=targets.device)
    if not overrides:
        return weights
    players = player_indices_from_features(features)
    for (player, action), value in overrides.items():
        action_index = HOLDEM_CANONICAL_ACTIONS.index(action)
        selected = (players == player) & (targets == action_index)
        weights = torch.where(selected, torch.full_like(weights, value), weights)
    return weights


def facing_bet_mask_from_masks(masks):
    call_index = HOLDEM_CANONICAL_ACTIONS.index("call")
    fold_index = HOLDEM_CANONICAL_ACTIONS.index("fold")
    return masks[:, call_index] & masks[:, fold_index]


def opponent_aggression_count_mask_from_features(features, minimum: int):
    if minimum < 1:
        raise ValueError("opponent aggression threshold must be positive")
    if features.shape[1] < HOLDEM_FEATURE_DIM + HOLDEM_ACTION_HISTORY_FEATURE_DIM:
        raise ValueError("opponent aggression gating requires action-history features")
    import torch

    opponent_aggression_feature = features[:, -HOLDEM_ACTION_HISTORY_FEATURE_DIM + 1]
    max_total_aggressions = 16.0
    opponent_aggressions = torch.round(
        opponent_aggression_feature * max_total_aggressions
    ).long()
    return opponent_aggressions >= minimum


def facing_bet_action_weights_from_masks_targets(
    masks,
    targets,
    overrides: dict[str, float],
    *,
    features=None,
    after_opponent_aggressions: int | None = None,
):
    import torch

    weights = torch.ones(targets.shape[0], dtype=torch.float32, device=targets.device)
    if not overrides:
        return weights
    facing_bet = facing_bet_mask_from_masks(masks)
    if after_opponent_aggressions is not None:
        if features is None:
            raise ValueError("opponent aggression gating requires features")
        facing_bet = facing_bet & opponent_aggression_count_mask_from_features(
            features,
            after_opponent_aggressions,
        )
    for action, value in overrides.items():
        action_index = HOLDEM_CANONICAL_ACTIONS.index(action)
        selected = facing_bet & (targets == action_index)
        weights = torch.where(selected, torch.full_like(weights, value), weights)
    return weights


def player_facing_bet_action_weights_from_features_masks_targets(
    features,
    masks,
    targets,
    overrides: dict[tuple[int, str], float],
    *,
    after_opponent_aggressions: int | None = None,
):
    import torch

    weights = torch.ones(targets.shape[0], dtype=torch.float32, device=targets.device)
    if not overrides:
        return weights
    players = player_indices_from_features(features)
    facing_bet = facing_bet_mask_from_masks(masks)
    if after_opponent_aggressions is not None:
        facing_bet = facing_bet & opponent_aggression_count_mask_from_features(
            features,
            after_opponent_aggressions,
        )
    for (player, action), value in overrides.items():
        action_index = HOLDEM_CANONICAL_ACTIONS.index(action)
        selected = (players == player) & facing_bet & (targets == action_index)
        weights = torch.where(selected, torch.full_like(weights, value), weights)
    return weights


def action_value_example_weights_from_mask(action_value_mask, weight: float):
    if weight <= 0.0:
        raise ValueError("action value example weight must be positive")
    import torch

    weights = torch.ones(
        action_value_mask.shape[0],
        dtype=torch.float32,
        device=action_value_mask.device,
    )
    if weight == 1.0:
        return weights
    return torch.where(action_value_mask, torch.full_like(weights, weight), weights)


def player_action_value_weights_from_features_mask(
    features,
    action_value_mask,
    overrides: dict[int, float],
):
    import torch

    weights = torch.ones(
        action_value_mask.shape[0],
        dtype=torch.float32,
        device=action_value_mask.device,
    )
    if not overrides:
        return weights
    players = player_indices_from_features(features)
    for player, value in overrides.items():
        selected = (players == player) & action_value_mask
        weights = torch.where(selected, torch.full_like(weights, value), weights)
    return weights


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
    facing_bet = facing_bet_mask_from_masks(masks)
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
    expert_mccfr_checkpoint: Path | None = None,
    expert_mccfr_fallback_policy: str = "tight-turn-river-exact-pot-odds",
    expert_mccfr_min_strategy_weight: float = 0.0,
    expert_mccfr_strategy_mode: str = "average",
    expert_mccfr_strategy_support_mode: str = "count",
    action_history_features: bool,
    soft_target_temperature: float | None,
    record_facing_bet_only: bool,
    record_min_opponent_aggressions: int | None,
):
    feature_equity_fn = None
    if feature_equity_checkpoint is not None:
        feature_equity_fn = equity_estimator_from_checkpoint(feature_equity_checkpoint)

    behavior_policy = None
    if behavior_checkpoint is not None:
        from alphapoker.evaluate_holdem_model import model_policy_from_checkpoint

        behavior_policy = model_policy_from_checkpoint(behavior_checkpoint)

    expert_policy_override = None
    if expert_mccfr_checkpoint is not None:
        from alphapoker.holdem_mccfr import (
            HoldemAbstractionCFRTrainer,
            holdem_policy_from_trainer,
        )
        from alphapoker.holdem_self_play import make_policy

        trainer = HoldemAbstractionCFRTrainer.load_checkpoint(expert_mccfr_checkpoint)
        shard_seed = seed + index * 1_000_003
        fallback_policy = make_policy(
            expert_mccfr_fallback_policy,
            random.Random(shard_seed + 11),
            equity_sims,
            rollout_sims,
            rollout_margin,
        )
        expert_policy_override = holdem_policy_from_trainer(
            trainer,
            random.Random(shard_seed + 13),
            fallback_policy=fallback_policy,
            min_strategy_weight=expert_mccfr_min_strategy_weight,
            strategy_mode=expert_mccfr_strategy_mode,
            strategy_support_mode=expert_mccfr_strategy_support_mode,
        )

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
        expert_policy_override=expert_policy_override,
        action_history_features=action_history_features,
        soft_target_temperature=soft_target_temperature,
        record_facing_bet_only=record_facing_bet_only,
        record_min_opponent_aggressions=record_min_opponent_aggressions,
    )


def _policy_example_shard_cache_path(shard_cache_dir: Path, index: int) -> Path:
    return shard_cache_dir / f"shard_{index:04d}.json"


def _policy_example_shard_cache_manifest(
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
    expert_mccfr_checkpoint: Path | None,
    expert_mccfr_fallback_policy: str,
    expert_mccfr_min_strategy_weight: float,
    expert_mccfr_strategy_mode: str,
    expert_mccfr_strategy_support_mode: str,
    action_history_features: bool,
    soft_target_temperature: float | None,
    record_facing_bet_only: bool,
    record_min_opponent_aggressions: int | None,
    jobs: int,
    shard_hands: list[int],
) -> dict[str, Any]:
    return {
        "version": 1,
        "hands": hands,
        "seed": seed,
        "equity_sims": equity_sims,
        "expert_player": expert_player,
        "expert_policy": expert_policy,
        "opponent_policy": opponent_policy,
        "rollout_sims": rollout_sims,
        "rollout_margin": rollout_margin,
        "feature_equity_sims": feature_equity_sims,
        "feature_equity_mode": feature_equity_mode,
        "feature_equity_checkpoint": (
            str(feature_equity_checkpoint)
            if feature_equity_checkpoint is not None
            else None
        ),
        "behavior_checkpoint": (
            str(behavior_checkpoint) if behavior_checkpoint is not None else None
        ),
        "expert_mccfr_checkpoint": (
            str(expert_mccfr_checkpoint)
            if expert_mccfr_checkpoint is not None
            else None
        ),
        "expert_mccfr_fallback_policy": expert_mccfr_fallback_policy,
        "expert_mccfr_min_strategy_weight": expert_mccfr_min_strategy_weight,
        "expert_mccfr_strategy_mode": expert_mccfr_strategy_mode,
        "expert_mccfr_strategy_support_mode": expert_mccfr_strategy_support_mode,
        "action_history_features": action_history_features,
        "soft_target_temperature": soft_target_temperature,
        "record_facing_bet_only": record_facing_bet_only,
        "record_min_opponent_aggressions": record_min_opponent_aggressions,
        "jobs": jobs,
        "shard_hands": shard_hands,
    }


def _prepare_policy_example_shard_cache(
    shard_cache_dir: Path,
    manifest: dict[str, Any],
) -> None:
    shard_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_cache_dir / "manifest.json"
    if manifest_path.exists():
        cached_manifest = json.loads(manifest_path.read_text())
        if cached_manifest != manifest:
            raise ValueError(
                "--examples-shard-cache-dir manifest does not match current "
                "generation arguments"
            )
        return
    write_json(manifest_path, manifest)


def _write_policy_examples_shard_cache(
    shard_cache_dir: Path,
    index: int,
    examples: list[Any],
) -> None:
    path = _policy_example_shard_cache_path(shard_cache_dir, index)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_policy_examples(tmp_path, examples)
    tmp_path.replace(path)


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
    expert_mccfr_checkpoint: Path | None = None,
    expert_mccfr_fallback_policy: str = "tight-turn-river-exact-pot-odds",
    expert_mccfr_min_strategy_weight: float = 0.0,
    expert_mccfr_strategy_mode: str = "average",
    expert_mccfr_strategy_support_mode: str = "count",
    action_history_features: bool,
    soft_target_temperature: float | None,
    record_facing_bet_only: bool,
    record_min_opponent_aggressions: int | None,
    jobs: int,
    progress: bool = False,
    shard_cache_dir: Path | None = None,
):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    shard_hands = _shard_hands(hands, jobs)
    if shard_cache_dir is not None:
        _prepare_policy_example_shard_cache(
            shard_cache_dir,
            _policy_example_shard_cache_manifest(
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
                expert_mccfr_checkpoint=expert_mccfr_checkpoint,
                expert_mccfr_fallback_policy=expert_mccfr_fallback_policy,
                expert_mccfr_min_strategy_weight=expert_mccfr_min_strategy_weight,
                expert_mccfr_strategy_mode=expert_mccfr_strategy_mode,
                expert_mccfr_strategy_support_mode=expert_mccfr_strategy_support_mode,
                action_history_features=action_history_features,
                soft_target_temperature=soft_target_temperature,
                record_facing_bet_only=record_facing_bet_only,
                record_min_opponent_aggressions=record_min_opponent_aggressions,
                jobs=jobs,
                shard_hands=shard_hands,
            ),
        )
    if not shard_hands:
        return []
    if jobs == 1:
        if shard_cache_dir is not None:
            shard_path = _policy_example_shard_cache_path(shard_cache_dir, 0)
            if shard_path.exists():
                examples = read_policy_examples(shard_path)
                if progress:
                    print(
                        f"examples shard 0: hands={hands} examples={len(examples)} cached",
                        file=sys.stderr,
                        flush=True,
                    )
                return examples
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
            expert_mccfr_checkpoint=expert_mccfr_checkpoint,
            expert_mccfr_fallback_policy=expert_mccfr_fallback_policy,
            expert_mccfr_min_strategy_weight=expert_mccfr_min_strategy_weight,
            expert_mccfr_strategy_mode=expert_mccfr_strategy_mode,
            expert_mccfr_strategy_support_mode=expert_mccfr_strategy_support_mode,
            action_history_features=action_history_features,
            soft_target_temperature=soft_target_temperature,
            record_facing_bet_only=record_facing_bet_only,
            record_min_opponent_aggressions=record_min_opponent_aggressions,
        )
        if progress:
            print(
                f"examples shard 0: hands={hands} examples={len(examples)}",
                file=sys.stderr,
                flush=True,
            )
        if shard_cache_dir is not None:
            _write_policy_examples_shard_cache(shard_cache_dir, 0, examples)
        return examples

    examples = []
    shard_results = {}
    missing_shards = []
    if shard_cache_dir is not None:
        for index, shard_size in enumerate(shard_hands):
            shard_path = _policy_example_shard_cache_path(shard_cache_dir, index)
            if shard_path.exists():
                result = read_policy_examples(shard_path)
                shard_results[index] = result
                if progress:
                    print(
                        f"examples shard {index}: hands={shard_size} "
                        f"examples={len(result)} cached",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                missing_shards.append((index, shard_size))
    else:
        missing_shards = list(enumerate(shard_hands))

    if missing_shards:
        with ProcessPoolExecutor(max_workers=min(jobs, len(missing_shards))) as executor:
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
                    expert_mccfr_checkpoint=expert_mccfr_checkpoint,
                    expert_mccfr_fallback_policy=expert_mccfr_fallback_policy,
                    expert_mccfr_min_strategy_weight=expert_mccfr_min_strategy_weight,
                    expert_mccfr_strategy_mode=expert_mccfr_strategy_mode,
                    expert_mccfr_strategy_support_mode=expert_mccfr_strategy_support_mode,
                    action_history_features=action_history_features,
                    soft_target_temperature=soft_target_temperature,
                    record_facing_bet_only=record_facing_bet_only,
                    record_min_opponent_aggressions=record_min_opponent_aggressions,
                ): index
                for index, shard_size in missing_shards
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                shard_results[index] = result
                if shard_cache_dir is not None:
                    _write_policy_examples_shard_cache(shard_cache_dir, index, result)
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
    if (
        args.expert_mccfr_checkpoint is not None
        and args.soft_target_temperature is not None
    ):
        raise ValueError("--expert-mccfr-checkpoint does not support soft targets")

    examples_in = getattr(args, "examples_in", None)
    extra_examples_in = getattr(args, "extra_examples_in", None) or []
    examples_out = getattr(args, "examples_out", None)
    examples_shard_cache_dir = getattr(args, "examples_shard_cache_dir", None)
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
            expert_mccfr_checkpoint=args.expert_mccfr_checkpoint,
            expert_mccfr_fallback_policy=args.expert_mccfr_fallback_policy,
            expert_mccfr_min_strategy_weight=args.expert_mccfr_min_strategy_weight,
            expert_mccfr_strategy_mode=args.expert_mccfr_strategy_mode,
            expert_mccfr_strategy_support_mode=args.expert_mccfr_strategy_support_mode,
            action_history_features=bool(getattr(args, "action_history_features", False)),
            soft_target_temperature=getattr(args, "soft_target_temperature", None),
            record_facing_bet_only=bool(getattr(args, "record_facing_bet_only", False)),
            record_min_opponent_aggressions=getattr(
                args,
                "record_min_opponent_aggressions",
                None,
            ),
            jobs=jobs,
            progress=bool(getattr(args, "progress", False)),
            shard_cache_dir=examples_shard_cache_dir,
        )
    extra_example_count = 0
    for extra_examples_path in extra_examples_in:
        extra_examples = read_policy_examples(extra_examples_path)
        extra_example_count += len(extra_examples)
        examples.extend(extra_examples)
    if examples_out is not None:
        write_policy_examples(examples_out, examples)
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    targets = torch.tensor([example.action_index for example in examples], dtype=torch.long)
    masks = torch.tensor([example.legal_mask for example in examples], dtype=torch.bool)
    soft_target_examples = sum(example.action_probs is not None for example in examples)
    action_value_target_examples = sum(example.action_values is not None for example in examples)
    soft_targets = None
    if soft_target_examples:
        soft_target_rows = []
        for example in examples:
            if example.action_probs is None:
                row = [0.0 for _ in HOLDEM_CANONICAL_ACTIONS]
                row[example.action_index] = 1.0
            else:
                if len(example.action_probs) != len(HOLDEM_CANONICAL_ACTIONS):
                    raise ValueError("soft target action_probs length does not match actions")
                row = [float(value) for value in example.action_probs]
            soft_target_rows.append(row)
        soft_targets = torch.tensor(soft_target_rows, dtype=torch.float32)
    action_value_targets = None
    action_value_target_mask = None
    if action_value_target_examples:
        action_value_rows = []
        action_value_mask_rows = []
        for example in examples:
            if example.action_values is None:
                action_value_rows.append([0.0 for _ in HOLDEM_CANONICAL_ACTIONS])
                action_value_mask_rows.append(False)
            else:
                if len(example.action_values) != len(HOLDEM_CANONICAL_ACTIONS):
                    raise ValueError("action_values length does not match actions")
                action_value_rows.append([float(value) for value in example.action_values])
                action_value_mask_rows.append(True)
        action_value_targets = torch.tensor(action_value_rows, dtype=torch.float32)
        action_value_target_mask = torch.tensor(action_value_mask_rows, dtype=torch.bool)
    action_value_loss_weight = float(getattr(args, "action_value_loss_weight", 0.0))
    action_value_target_scale = float(getattr(args, "action_value_target_scale", 1.0))
    if action_value_loss_weight < 0.0:
        raise ValueError("--action-value-loss-weight must be non-negative")
    if action_value_target_scale <= 0.0:
        raise ValueError("--action-value-target-scale must be positive")
    if action_value_loss_weight > 0.0 and action_value_targets is None:
        raise ValueError("--action-value-loss-weight requires cached action_values")
    action_value_example_weight = float(getattr(args, "action_value_example_weight", 1.0))
    if action_value_example_weight <= 0.0:
        raise ValueError("--action-value-example-weight must be positive")
    if action_value_example_weight != 1.0 and action_value_targets is None:
        raise ValueError("--action-value-example-weight requires cached action_values")
    player_action_value_weight_overrides = player_value_weight_overrides_from_specs(
        getattr(args, "player_action_value_weight", None)
    )
    if player_action_value_weight_overrides and action_value_targets is None:
        raise ValueError("--player-action-value-weight requires cached action_values")
    validation_fraction = getattr(args, "validation_fraction", 0.0)
    train_indices, validation_indices = _split_train_validation_indices(
        len(examples),
        validation_fraction,
        args.seed,
    )
    train_features = features[train_indices]
    train_targets = targets[train_indices]
    train_masks = masks[train_indices]
    train_soft_targets = soft_targets[train_indices] if soft_targets is not None else None
    train_action_value_targets = (
        action_value_targets[train_indices] if action_value_targets is not None else None
    )
    train_action_value_target_mask = (
        action_value_target_mask[train_indices]
        if action_value_target_mask is not None
        else None
    )
    validation_features = features[validation_indices] if validation_indices else None
    validation_targets = targets[validation_indices] if validation_indices else None
    validation_masks = masks[validation_indices] if validation_indices else None
    validation_soft_targets = (
        soft_targets[validation_indices]
        if soft_targets is not None and validation_indices
        else None
    )
    validation_action_value_targets = (
        action_value_targets[validation_indices]
        if action_value_targets is not None and validation_indices
        else None
    )
    validation_action_value_target_mask = (
        action_value_target_mask[validation_indices]
        if action_value_target_mask is not None and validation_indices
        else None
    )
    facing_bet_weight = float(getattr(args, "facing_bet_weight", 1.0))
    player_action_weight_overrides = player_action_weight_overrides_from_specs(
        getattr(args, "player_action_weight", None)
    )
    facing_bet_action_weight_overrides = action_weight_overrides_from_specs(
        getattr(args, "facing_bet_action_weight", None)
    )
    facing_bet_action_weight_after_opponent_aggressions = getattr(
        args,
        "facing_bet_action_weight_after_opponent_aggressions",
        None,
    )
    player_facing_bet_action_weight_overrides = player_action_weight_overrides_from_specs(
        getattr(args, "player_facing_bet_action_weight", None)
    )
    player_facing_bet_action_weight_after_opponent_aggressions = getattr(
        args,
        "player_facing_bet_action_weight_after_opponent_aggressions",
        None,
    )
    facing_bet_weights = example_weights_from_masks(masks, facing_bet_weight)
    player_action_weights = player_action_weights_from_features_targets(
        features,
        targets,
        player_action_weight_overrides,
    )
    facing_bet_action_weights = facing_bet_action_weights_from_masks_targets(
        masks,
        targets,
        facing_bet_action_weight_overrides,
        features=features,
        after_opponent_aggressions=(
            facing_bet_action_weight_after_opponent_aggressions
        ),
    )
    player_facing_bet_action_weights = (
        player_facing_bet_action_weights_from_features_masks_targets(
            features,
            masks,
            targets,
            player_facing_bet_action_weight_overrides,
            after_opponent_aggressions=(
                player_facing_bet_action_weight_after_opponent_aggressions
            ),
        )
    )
    if action_value_target_mask is None:
        action_value_example_weights = torch.ones_like(facing_bet_weights)
        player_action_value_weights = torch.ones_like(facing_bet_weights)
    else:
        action_value_example_weights = action_value_example_weights_from_mask(
            action_value_target_mask,
            action_value_example_weight,
        )
        player_action_value_weights = player_action_value_weights_from_features_mask(
            features,
            action_value_target_mask,
            player_action_value_weight_overrides,
        )
    example_weights = (
        facing_bet_weights
        * player_action_weights
        * facing_bet_action_weights
        * player_facing_bet_action_weights
        * action_value_example_weights
        * player_action_value_weights
    )
    use_example_weights = (
        facing_bet_weight != 1.0
        or bool(player_action_weight_overrides)
        or bool(facing_bet_action_weight_overrides)
        or bool(player_facing_bet_action_weight_overrides)
        or action_value_example_weight != 1.0
        or bool(player_action_value_weight_overrides)
    )
    train_example_weights = example_weights[train_indices] if use_example_weights else None
    validation_example_weights = (
        example_weights[validation_indices]
        if use_example_weights and validation_indices
        else None
    )
    facing_bet_mask = facing_bet_mask_from_masks(masks)
    init_kl_weight = float(getattr(args, "init_kl_weight", 0.0))
    init_kl_example_weighting = getattr(args, "init_kl_example_weighting", "example")
    if init_kl_example_weighting not in ("example", "state", "uniform"):
        raise ValueError("--init-kl-example-weighting must be example, state, or uniform")
    if init_kl_example_weighting == "example":
        kl_example_weights = example_weights if use_example_weights else None
    elif init_kl_example_weighting == "state":
        kl_example_weights = (
            facing_bet_weights
            * action_value_example_weights
            * player_action_value_weights
        )
    else:
        kl_example_weights = None
    train_kl_example_weights = (
        kl_example_weights[train_indices] if kl_example_weights is not None else None
    )
    validation_kl_example_weights = (
        kl_example_weights[validation_indices]
        if kl_example_weights is not None and validation_indices
        else None
    )

    torch.manual_seed(0)
    model = HoldemPolicyNet(input_dim=features.shape[1])
    init_checkpoint = getattr(args, "init_checkpoint", None)
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
    action_weight_overrides = action_weight_overrides_from_specs(
        getattr(args, "action_weight", None)
    )
    class_weights = apply_action_weight_overrides(class_weights, action_weight_overrides)
    resolved_class_weight_exponent = class_weight_exponent_for_mode(
        class_weighting,
        class_weight_exponent,
    )
    effective_class_weights = (
        None
        if class_weights is None
        else {
            action: float(class_weights[index].detach().cpu())
            for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
        }
    )

    def loss_for(
        batch_features,
        batch_targets,
        batch_masks,
        batch_soft_targets=None,
        batch_action_value_targets=None,
        batch_action_value_target_mask=None,
        batch_weights=None,
        batch_kl_weights=None,
    ):
        logits = model(batch_features)
        masked_logits = logits.masked_fill(~batch_masks, -1e9)
        if batch_soft_targets is None and batch_weights is None:
            loss = F.cross_entropy(masked_logits, batch_targets, weight=class_weights)
        elif batch_soft_targets is None:
            losses = F.cross_entropy(
                masked_logits,
                batch_targets,
                weight=class_weights,
                reduction="none",
            )
            loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)
        else:
            log_probs = F.log_softmax(masked_logits, dim=1)
            if class_weights is None:
                weighted_targets = batch_soft_targets
                class_normalizer = torch.ones(
                    batch_soft_targets.shape[0],
                    dtype=torch.float32,
                    device=batch_soft_targets.device,
                )
            else:
                weighted_targets = batch_soft_targets * class_weights
                class_normalizer = weighted_targets.sum(dim=1).clamp_min(1e-12)
            losses = -(weighted_targets * log_probs).sum(dim=1)
            if batch_weights is None:
                loss = losses.sum() / class_normalizer.sum().clamp_min(1e-12)
            else:
                loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)
        if (
            action_value_loss_weight > 0.0
            and batch_action_value_targets is not None
            and batch_action_value_target_mask is not None
            and bool(batch_action_value_target_mask.any())
        ):
            legal_mask_float = batch_masks.float()
            legal_counts = legal_mask_float.sum(dim=1).clamp_min(1.0)
            logit_mean = (logits.masked_fill(~batch_masks, 0.0).sum(dim=1) / legal_counts).unsqueeze(1)
            logit_advantages = (logits - logit_mean).masked_fill(~batch_masks, 0.0)
            target_mean = (
                (batch_action_value_targets * legal_mask_float).sum(dim=1) / legal_counts
            ).unsqueeze(1)
            target_advantages = (
                (batch_action_value_targets - target_mean) / action_value_target_scale
            ).masked_fill(~batch_masks, 0.0)
            value_losses = (
                ((logit_advantages - target_advantages).pow(2) * legal_mask_float).sum(dim=1)
                / legal_counts
            )
            value_losses = value_losses[batch_action_value_target_mask]
            if batch_weights is None:
                action_value_loss = value_losses.mean()
            else:
                value_weights = batch_weights[batch_action_value_target_mask]
                action_value_loss = (
                    (value_losses * value_weights).sum()
                    / value_weights.sum().clamp_min(1e-12)
                )
            loss = loss + action_value_loss_weight * action_value_loss
        if anchor_model is None:
            return loss
        with torch.no_grad():
            anchor_logits = anchor_model(batch_features).masked_fill(~batch_masks, -1e9)
            anchor_probs = F.softmax(anchor_logits, dim=1)
        if batch_kl_weights is None:
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
                (kl_losses * batch_kl_weights).sum()
                / batch_kl_weights.sum().clamp_min(1e-12)
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
        loss = loss_for(
            train_features,
            train_targets,
            train_masks,
            train_soft_targets,
            train_action_value_targets,
            train_action_value_target_mask,
            train_example_weights,
            train_kl_example_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss = loss_for(
                train_features,
                train_targets,
                train_masks,
                train_soft_targets,
                train_action_value_targets,
                train_action_value_target_mask,
                train_example_weights,
                train_kl_example_weights,
            )
            final_loss = float(train_loss.detach().cpu())
            validation_loss_value = None
            selection_loss = final_loss
            if validation_features is not None:
                validation_loss = loss_for(
                    validation_features,
                    validation_targets,
                    validation_masks,
                    validation_soft_targets,
                    validation_action_value_targets,
                    validation_action_value_target_mask,
                    validation_example_weights,
                    validation_kl_example_weights,
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
    facing_bet_target_action_counts = {
        action: int(((targets == index) & facing_bet_mask).sum().item())
        for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
    }
    facing_bet_predicted_action_counts = {
        action: int(((predictions == index) & facing_bet_mask).sum().item())
        for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
    }
    soft_target_action_mass = None
    if soft_targets is not None:
        soft_target_action_mass = {
            action: float(soft_targets[:, index].sum().item())
            for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
        }
    player_target_action_counts = None
    player_predicted_action_counts = None
    player_facing_bet_target_action_counts = None
    player_facing_bet_predicted_action_counts = None
    if features.shape[1] >= HOLDEM_PLAYER_FEATURE_OFFSET + HOLDEM_PLAYER_FEATURE_DIM:
        players = player_indices_from_features(features)
        player_target_action_counts = {}
        player_predicted_action_counts = {}
        player_facing_bet_target_action_counts = {}
        player_facing_bet_predicted_action_counts = {}
        for player in (0, 1):
            selected = players == player
            selected_facing_bet = selected & facing_bet_mask
            player_target_action_counts[str(player)] = {
                action: int(((targets == index) & selected).sum().item())
                for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
            }
            player_predicted_action_counts[str(player)] = {
                action: int(((predictions == index) & selected).sum().item())
                for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
            }
            player_facing_bet_target_action_counts[str(player)] = {
                action: int(((targets == index) & selected_facing_bet).sum().item())
                for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
            }
            player_facing_bet_predicted_action_counts[str(player)] = {
                action: int(((predictions == index) & selected_facing_bet).sum().item())
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
        "expert_mccfr_checkpoint": (
            str(args.expert_mccfr_checkpoint)
            if args.expert_mccfr_checkpoint is not None
            else None
        ),
        "expert_mccfr_fallback_policy": args.expert_mccfr_fallback_policy,
        "expert_mccfr_min_strategy_weight": args.expert_mccfr_min_strategy_weight,
        "expert_mccfr_strategy_mode": args.expert_mccfr_strategy_mode,
        "expert_mccfr_strategy_support_mode": args.expert_mccfr_strategy_support_mode,
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
        "soft_target_temperature": getattr(args, "soft_target_temperature", None),
        "record_facing_bet_only": bool(getattr(args, "record_facing_bet_only", False)),
        "record_min_opponent_aggressions": getattr(
            args,
            "record_min_opponent_aggressions",
            None,
        ),
        "soft_target_examples": int(soft_target_examples),
        "soft_target_action_mass": soft_target_action_mass,
        "action_value_target_examples": int(action_value_target_examples),
        "action_value_loss_weight": action_value_loss_weight,
        "action_value_target_scale": action_value_target_scale,
        "action_value_example_weight": action_value_example_weight,
        "action_value_weighted_examples": int(
            action_value_target_mask.sum().item()
        )
        if action_value_target_mask is not None and action_value_example_weight != 1.0
        else 0,
        "jobs": jobs,
        "epochs": args.epochs,
        "lr": args.lr,
        "init_kl_weight": init_kl_weight,
        "init_kl_example_weighting": init_kl_example_weighting,
        "class_weighting": class_weighting,
        "class_weight_exponent": resolved_class_weight_exponent,
        "action_weight_overrides": action_weight_overrides,
        "effective_class_weights": effective_class_weights,
        "player_action_weight_overrides": {
            f"{player}:{action}": weight
            for (player, action), weight in player_action_weight_overrides.items()
        },
        "player_action_weighted_examples": int(
            (player_action_weights != 1.0).sum().item()
        )
        if player_action_weight_overrides
        else 0,
        "facing_bet_action_weight_overrides": facing_bet_action_weight_overrides,
        "facing_bet_action_weight_after_opponent_aggressions": (
            facing_bet_action_weight_after_opponent_aggressions
        ),
        "facing_bet_action_weighted_examples": int(
            (facing_bet_action_weights != 1.0).sum().item()
        )
        if facing_bet_action_weight_overrides
        else 0,
        "player_facing_bet_action_weight_overrides": {
            f"{player}:{action}": weight
            for (player, action), weight in player_facing_bet_action_weight_overrides.items()
        },
        "player_facing_bet_action_weight_after_opponent_aggressions": (
            player_facing_bet_action_weight_after_opponent_aggressions
        ),
        "player_facing_bet_action_weighted_examples": int(
            (player_facing_bet_action_weights != 1.0).sum().item()
        )
        if player_facing_bet_action_weight_overrides
        else 0,
        "player_action_value_weight_overrides": {
            str(player): weight
            for player, weight in player_action_value_weight_overrides.items()
        },
        "player_action_value_weighted_examples": int(
            (player_action_value_weights != 1.0).sum().item()
        )
        if player_action_value_weight_overrides
        else 0,
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
        "facing_bet_target_action_counts": facing_bet_target_action_counts,
        "facing_bet_predicted_action_counts": facing_bet_predicted_action_counts,
        "player_target_action_counts": player_target_action_counts,
        "player_predicted_action_counts": player_predicted_action_counts,
        "player_facing_bet_target_action_counts": player_facing_bet_target_action_counts,
        "player_facing_bet_predicted_action_counts": (
            player_facing_bet_predicted_action_counts
        ),
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
    metrics["extra_examples_in"] = [str(path) for path in extra_examples_in]
    metrics["extra_examples"] = int(extra_example_count)
    if examples_out is not None:
        metrics["examples_out"] = str(examples_out)
    if examples_shard_cache_dir is not None:
        metrics["examples_shard_cache_dir"] = str(examples_shard_cache_dir)
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--expert-player", type=int, choices=[0, 1])
    parser.add_argument("--expert-policy", choices=HOLDEM_EXPERT_POLICIES, default="equity")
    parser.add_argument(
        "--expert-mccfr-checkpoint",
        type=Path,
        help="Use a Hold'em MCCFR checkpoint as the supervised expert policy.",
    )
    parser.add_argument(
        "--expert-mccfr-fallback-policy",
        choices=HOLDEM_DATASET_OPPONENT_POLICIES,
        default="tight-turn-river-exact-pot-odds",
        help="Fallback policy for unsupported MCCFR infosets while labeling examples.",
    )
    parser.add_argument(
        "--expert-mccfr-min-strategy-weight",
        type=float,
        default=0.0,
        help="Minimum MCCFR support before falling back while labeling examples.",
    )
    parser.add_argument(
        "--expert-mccfr-strategy-mode",
        choices=HOLDEM_MCCFR_STRATEGY_MODES,
        default="average",
    )
    parser.add_argument(
        "--expert-mccfr-strategy-support-mode",
        choices=HOLDEM_MCCFR_SUPPORT_MODES,
        default="count",
    )
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
    parser.add_argument("--soft-target-temperature", type=float)
    parser.add_argument(
        "--record-facing-bet-only",
        action="store_true",
        help="Record supervised examples only when the expert player is facing a bet.",
    )
    parser.add_argument(
        "--record-min-opponent-aggressions",
        type=int,
        help=(
            "Record supervised examples only after at least this many prior "
            "opponent bets or raises."
        ),
    )
    parser.add_argument("--action-value-loss-weight", type=float, default=0.0)
    parser.add_argument("--action-value-target-scale", type=float, default=1.0)
    parser.add_argument("--action-value-example-weight", type=float, default=1.0)
    parser.add_argument("--behavior-checkpoint", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--init-kl-weight", type=float, default=0.0)
    parser.add_argument(
        "--init-kl-example-weighting",
        choices=("example", "state", "uniform"),
        default="example",
        help=(
            "Use all example weights, only state-level example weights, or no "
            "example weights for initialized-policy KL anchoring."
        ),
    )
    parser.add_argument("--init-allow-input-expansion", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--class-weighting", choices=CLASS_WEIGHTING_MODES, default="none")
    parser.add_argument("--class-weight-exponent", type=float)
    parser.add_argument(
        "--action-weight",
        action="append",
        default=[],
        metavar="ACTION=WEIGHT",
        help="Multiply one action's loss weight, for example --action-weight raise=2.0.",
    )
    parser.add_argument(
        "--player-action-weight",
        action="append",
        default=[],
        metavar="PLAYER:ACTION=WEIGHT",
        help=(
            "Multiply examples for one current-player/action target, "
            "for example --player-action-weight 1:raise=2.0."
        ),
    )
    parser.add_argument(
        "--facing-bet-action-weight",
        action="append",
        default=[],
        metavar="ACTION=WEIGHT",
        help=(
            "Multiply examples for one target action only when facing a bet, "
            "for example --facing-bet-action-weight call=2.0."
        ),
    )
    parser.add_argument(
        "--facing-bet-action-weight-after-opponent-aggressions",
        type=int,
        help=(
            "Apply --facing-bet-action-weight only when action-history features show "
            "at least this many prior opponent bets or raises."
        ),
    )
    parser.add_argument(
        "--player-facing-bet-action-weight",
        action="append",
        default=[],
        metavar="PLAYER:ACTION=WEIGHT",
        help=(
            "Multiply examples for one current-player/action target only when facing "
            "a bet, for example --player-facing-bet-action-weight 1:call=2.0."
        ),
    )
    parser.add_argument(
        "--player-facing-bet-action-weight-after-opponent-aggressions",
        type=int,
        help=(
            "Apply --player-facing-bet-action-weight only when action-history "
            "features show at least this many prior opponent bets or raises."
        ),
    )
    parser.add_argument(
        "--player-action-value-weight",
        action="append",
        default=[],
        metavar="PLAYER=WEIGHT",
        help=(
            "Multiply cached action-value examples for one current player, "
            "for example --player-action-value-weight 1=3.0."
        ),
    )
    parser.add_argument("--facing-bet-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-in", type=Path)
    parser.add_argument(
        "--extra-examples-in",
        action="append",
        default=[],
        type=Path,
        help="Append cached examples to the primary generated or cached examples.",
    )
    parser.add_argument("--examples-out", type=Path)
    parser.add_argument(
        "--examples-shard-cache-dir",
        type=Path,
        help=(
            "Write generated example shards as they finish and reuse matching "
            "cached shards on rerun."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
