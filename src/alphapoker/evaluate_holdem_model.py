"""Evaluate a trained fixed-limit Hold'em policy checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from alphapoker.holdem import FixedLimitHoldemState, HoldemPolicy, policy_rollout_action_values
from alphapoker.holdem_evaluation import (
    ACTION_COUNT_KEYS,
    add_action_counts,
    aggregate_policy_match_shards,
    empty_action_counts,
    evaluate_policy_match,
    evaluate_policy_match_paired_seats,
)
from alphapoker.holdem_features import (
    HOLDEM_CANONICAL_ACTIONS,
    holdem_legal_action_mask,
    opponent_aggressions_before_current_decision,
)
from alphapoker.holdem_policy_features import (
    policy_feature_encoder_from_checkpoint_data,
)
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json


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


def player_checkpoints_from_args(args: argparse.Namespace) -> tuple[Path, Path]:
    return (
        getattr(args, "player0_checkpoint", None) or args.checkpoint,
        getattr(args, "player1_checkpoint", None) or args.checkpoint,
    )


def action_logit_biases_from_specs(specs: list[str] | None) -> dict[str, float]:
    biases: dict[str, float] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError("logit bias must use ACTION=BIAS")
        action, value_text = spec.split("=", 1)
        if action not in HOLDEM_CANONICAL_ACTIONS:
            raise ValueError(f"unknown Hold'em action for logit bias: {action}")
        try:
            biases[action] = float(value_text)
        except ValueError as error:
            raise ValueError(f"invalid logit bias for action {action}: {value_text}") from error
    return biases


def player_action_logit_biases_from_specs(
    specs: list[str] | None,
) -> dict[tuple[int, str], float]:
    biases: dict[tuple[int, str], float] = {}
    for spec in specs or []:
        if ":" not in spec or "=" not in spec:
            raise ValueError("player logit bias must use PLAYER:ACTION=BIAS")
        player_text, rest = spec.split(":", 1)
        action, value_text = rest.split("=", 1)
        try:
            player = int(player_text)
        except ValueError as error:
            raise ValueError(f"invalid player for logit bias: {player_text}") from error
        if player not in (0, 1):
            raise ValueError("player logit bias player must be 0 or 1")
        if action not in HOLDEM_CANONICAL_ACTIONS:
            raise ValueError(f"unknown Hold'em action for player logit bias: {action}")
        try:
            biases[(player, action)] = float(value_text)
        except ValueError as error:
            raise ValueError(
                f"invalid logit bias for player {player} action {action}: {value_text}"
            ) from error
    return biases


def serialize_player_action_biases(
    biases: dict[tuple[int, str], float],
) -> dict[str, float]:
    return {f"{player}:{action}": bias for (player, action), bias in biases.items()}


def _weighted_mean(metrics: list[dict[str, Any]], key: str) -> float:
    total_hands = sum(int(item["hands"]) for item in metrics)
    return (
        sum(float(item[key]) * int(item["hands"]) for item in metrics) / total_hands
        if total_hands
        else 0.0
    )


def _pooled_stdev(metrics: list[dict[str, Any]], mean_key: str, stdev_key: str) -> float:
    total_hands = sum(int(item["hands"]) for item in metrics)
    if total_hands <= 1:
        return 0.0
    mean = _weighted_mean(metrics, mean_key)
    sum_squares = 0.0
    for item in metrics:
        hands = int(item["hands"])
        stdev = float(item[stdev_key])
        item_mean = float(item[mean_key])
        sum_squares += max(0, hands - 1) * stdev * stdev
        sum_squares += hands * (item_mean - mean) * (item_mean - mean)
    return math.sqrt(sum_squares / (total_hands - 1))


def _summed_action_counts(metrics: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = empty_action_counts()
    for item in metrics:
        add_action_counts(counts, item[key])
    return counts


def _empty_model_decision_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "action_counts": {action: 0 for action in HOLDEM_CANONICAL_ACTIONS},
        "prob_sums": {action: 0.0 for action in HOLDEM_CANONICAL_ACTIONS},
        "legal_logit_sums": {action: 0.0 for action in HOLDEM_CANONICAL_ACTIONS},
        "legal_logit_counts": {action: 0 for action in HOLDEM_CANONICAL_ACTIONS},
        "chosen_prob_sum": 0.0,
        "top_logit_margin_sum": 0.0,
        "raise_vs_call_logit_sum": 0.0,
        "raise_vs_call_count": 0,
        "raise_vs_fold_logit_sum": 0.0,
        "raise_vs_fold_count": 0,
    }


class ModelDecisionDiagnostics:
    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, Any]] = defaultdict(_empty_model_decision_bucket)

    def record(self, state: FixedLimitHoldemState, logits, mask, action_index: int) -> None:
        import torch

        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(masked_logits, dim=0)
        player = state.current_player()
        facing_bet = bool(
            mask[HOLDEM_CANONICAL_ACTIONS.index("call")]
            and mask[HOLDEM_CANONICAL_ACTIONS.index("fold")]
        )
        bucket_names = ["all", f"player_{player}"]
        bucket_names.append("facing_bet" if facing_bet else "not_facing_bet")
        bucket_names.append(
            f"player_{player}_{'facing_bet' if facing_bet else 'not_facing_bet'}"
        )
        if facing_bet:
            opponent_aggressions = opponent_aggressions_before_current_decision(state)
            capped_aggressions = min(opponent_aggressions, 2)
            bucket_names.append(f"facing_bet_opp_aggr_{capped_aggressions}")
            bucket_names.append(
                f"player_{player}_facing_bet_opp_aggr_{capped_aggressions}"
            )

        legal_logits = [
            float(logits[index].detach().cpu())
            for index, legal in enumerate(mask)
            if bool(legal)
        ]
        top_margin = 0.0
        if len(legal_logits) >= 2:
            ordered_logits = sorted(legal_logits, reverse=True)
            top_margin = ordered_logits[0] - ordered_logits[1]

        for name in bucket_names:
            self._record_bucket(name, logits, mask, probs, action_index, top_margin)

    def _record_bucket(
        self,
        name: str,
        logits,
        mask,
        probs,
        action_index: int,
        top_margin: float,
    ) -> None:
        bucket = self._buckets[name]
        bucket["count"] += 1
        action = HOLDEM_CANONICAL_ACTIONS[action_index]
        bucket["action_counts"][action] += 1
        bucket["chosen_prob_sum"] += float(probs[action_index].detach().cpu())
        bucket["top_logit_margin_sum"] += top_margin
        for index, action_name in enumerate(HOLDEM_CANONICAL_ACTIONS):
            bucket["prob_sums"][action_name] += float(probs[index].detach().cpu())
            if bool(mask[index]):
                bucket["legal_logit_sums"][action_name] += float(
                    logits[index].detach().cpu()
                )
                bucket["legal_logit_counts"][action_name] += 1
        self._record_logit_diff(bucket, logits, mask, "raise", "call")
        self._record_logit_diff(bucket, logits, mask, "raise", "fold")

    @staticmethod
    def _record_logit_diff(
        bucket: dict[str, Any],
        logits,
        mask,
        left_action: str,
        right_action: str,
    ) -> None:
        left_index = HOLDEM_CANONICAL_ACTIONS.index(left_action)
        right_index = HOLDEM_CANONICAL_ACTIONS.index(right_action)
        if bool(mask[left_index]) and bool(mask[right_index]):
            key = f"{left_action}_vs_{right_action}"
            bucket[f"{key}_logit_sum"] += float(
                (logits[left_index] - logits[right_index]).detach().cpu()
            )
            bucket[f"{key}_count"] += 1

    def as_raw(self) -> dict[str, Any]:
        return dict(self._buckets)


def merge_model_decision_diagnostics_raw(
    raw_items: list[dict[str, Any]],
) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    for raw in raw_items:
        for name, bucket in raw.items():
            target = merged.setdefault(name, _empty_model_decision_bucket())
            target["count"] += int(bucket["count"])
            target["chosen_prob_sum"] += float(bucket["chosen_prob_sum"])
            target["top_logit_margin_sum"] += float(bucket["top_logit_margin_sum"])
            target["raise_vs_call_logit_sum"] += float(
                bucket["raise_vs_call_logit_sum"]
            )
            target["raise_vs_call_count"] += int(bucket["raise_vs_call_count"])
            target["raise_vs_fold_logit_sum"] += float(
                bucket["raise_vs_fold_logit_sum"]
            )
            target["raise_vs_fold_count"] += int(bucket["raise_vs_fold_count"])
            for action in HOLDEM_CANONICAL_ACTIONS:
                target["action_counts"][action] += int(bucket["action_counts"][action])
                target["prob_sums"][action] += float(bucket["prob_sums"][action])
                target["legal_logit_sums"][action] += float(
                    bucket["legal_logit_sums"][action]
                )
                target["legal_logit_counts"][action] += int(
                    bucket["legal_logit_counts"][action]
                )
    return merged


def format_model_decision_diagnostics(raw: dict[str, Any]) -> dict[str, Any]:
    formatted = {}
    for name, bucket in sorted(raw.items()):
        count = int(bucket["count"])
        if count == 0:
            continue
        legal_logit_counts = bucket["legal_logit_counts"]
        formatted[name] = {
            "count": count,
            "action_counts": bucket["action_counts"],
            "avg_action_probs": {
                action: float(bucket["prob_sums"][action]) / count
                for action in HOLDEM_CANONICAL_ACTIONS
            },
            "avg_legal_logits": {
                action: (
                    float(bucket["legal_logit_sums"][action])
                    / int(legal_logit_counts[action])
                )
                for action in HOLDEM_CANONICAL_ACTIONS
                if int(legal_logit_counts[action]) > 0
            },
            "avg_chosen_prob": float(bucket["chosen_prob_sum"]) / count,
            "avg_top_logit_margin": float(bucket["top_logit_margin_sum"]) / count,
            "avg_raise_vs_call_logit": (
                float(bucket["raise_vs_call_logit_sum"])
                / int(bucket["raise_vs_call_count"])
                if int(bucket["raise_vs_call_count"]) > 0
                else None
            ),
            "avg_raise_vs_fold_logit": (
                float(bucket["raise_vs_fold_logit_sum"])
                / int(bucket["raise_vs_fold_count"])
                if int(bucket["raise_vs_fold_count"]) > 0
                else None
            ),
        }
    return formatted


def attach_model_decision_diagnostics(
    metrics: dict[str, Any],
    raw_items: list[dict[str, Any]],
) -> None:
    if not raw_items:
        return
    raw = merge_model_decision_diagnostics_raw(raw_items)
    metrics["model_decision_diagnostics_raw"] = raw
    metrics["model_decision_diagnostics"] = format_model_decision_diagnostics(raw)


def aggregate_model_player_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if len(metrics) == 1:
        return metrics[0]
    total_hands = sum(int(item["hands"]) for item in metrics)
    model_stdev = _pooled_stdev(metrics, "avg_utility_model", "utility_stdev_model")
    p0_stdev = _pooled_stdev(metrics, "avg_utility_p0", "utility_stdev_p0")
    first = metrics[0]
    aggregated = {
        "hands": total_hands,
        "hands_per_model_player": first["hands"],
        "model_player": "both",
        "avg_utility_model": _weighted_mean(metrics, "avg_utility_model"),
        "utility_stdev_model": model_stdev,
        "utility_stderr_model": model_stdev / (total_hands**0.5) if total_hands else 0.0,
        "avg_utility_p0": _weighted_mean(metrics, "avg_utility_p0"),
        "utility_stdev_p0": p0_stdev,
        "utility_stderr_p0": p0_stdev / (total_hands**0.5) if total_hands else 0.0,
        "avg_actions": _weighted_mean(metrics, "avg_actions"),
        "folds": sum(int(item["folds"]) for item in metrics),
        "showdowns": sum(int(item["showdowns"]) for item in metrics),
        "seed": first["seed"],
        "seat_metrics": metrics,
    }
    for key in ACTION_COUNT_KEYS:
        if key in first:
            aggregated[key] = _summed_action_counts(metrics, key)
    for key in (
        "checkpoint",
        "policy",
        "opponent_policy",
        "equity_sims",
        "rollout_sims",
        "rollout_margin",
        "blend_checkpoint",
        "blend_weight",
        "blend_after_opponent_aggressions",
        "blend_facing_bet_only",
        "blend_players",
        "facing_bet_logit_biases",
        "facing_bet_logit_bias_after_opponent_aggressions",
        "facing_bet_logit_bias_min_raise_prob",
        "player_facing_bet_logit_biases",
        "player_facing_bet_logit_bias_after_opponent_aggressions",
        "player_facing_bet_logit_bias_min_raise_prob",
        "player0_checkpoint",
        "player1_checkpoint",
        "player_checkpoint",
        "fallback_policy",
        "min_strategy_weight",
        "bet_threshold",
        "raise_threshold",
        "call_margin",
    ):
        if key in first:
            aggregated[key] = first[key]
    attach_model_decision_diagnostics(
        aggregated,
        [
            item["model_decision_diagnostics_raw"]
            for item in metrics
            if "model_decision_diagnostics_raw" in item
        ],
    )
    return aggregated


def split_hands(hands: int, jobs: int) -> list[int]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if hands < 0:
        raise ValueError("hands must be non-negative")
    if hands == 0:
        return [0]
    shard_count = min(hands, jobs)
    base_hands, extra_hands = divmod(hands, shard_count)
    return [base_hands + (1 if shard < extra_hands else 0) for shard in range(shard_count)]


def evaluation_process_context() -> mp.context.BaseContext:
    start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    return mp.get_context(start_method)


def checkpoint_feature_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_dim": int(checkpoint["input_dim"]),
        "feature_equity_sims": checkpoint.get("feature_equity_sims"),
        "feature_equity_mode": checkpoint.get("feature_equity_mode", "random"),
        "feature_equity_checkpoint": checkpoint.get("feature_equity_checkpoint"),
        "action_history_features": bool(checkpoint.get("action_history_features", False)),
    }


def validate_blend_checkpoint_compatibility(
    primary_checkpoint: dict[str, Any],
    blend_checkpoint: dict[str, Any],
) -> None:
    primary_metadata = checkpoint_feature_metadata(primary_checkpoint)
    blend_metadata = checkpoint_feature_metadata(blend_checkpoint)
    if primary_metadata != blend_metadata:
        raise ValueError(
            "blended checkpoints must have compatible feature metadata: "
            f"{primary_metadata} != {blend_metadata}"
        )


def model_policy_from_checkpoint(
    checkpoint_path: Path,
    *,
    feature_seed: int = 0,
    blend_checkpoint_path: Path | None = None,
    blend_weight: float = 0.5,
    blend_after_opponent_aggressions: int | None = None,
    blend_facing_bet_only: bool = False,
    blend_players: tuple[int, ...] | None = None,
    facing_bet_logit_biases: dict[str, float] | None = None,
    facing_bet_logit_bias_after_opponent_aggressions: int | None = None,
    facing_bet_logit_bias_min_raise_prob: float | None = None,
    player_facing_bet_logit_biases: dict[tuple[int, str], float] | None = None,
    player_facing_bet_logit_bias_after_opponent_aggressions: int | None = None,
    player_facing_bet_logit_bias_min_raise_prob: float | None = None,
    model_rollout_sims: int | None = None,
    model_rollout_margin: float = 0.0,
    model_rollout_opponent_policy: str | None = None,
    model_rollout_opponent_equity_sims: int = 8,
    model_rollout_opponent_rollout_sims: int | None = None,
    model_rollout_opponent_rollout_margin: float = 1.0,
    decision_diagnostics: ModelDecisionDiagnostics | None = None,
) -> HoldemPolicy:
    import torch

    from alphapoker.holdem_model import HoldemPolicyNet

    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend_weight must be between 0.0 and 1.0")
    if blend_after_opponent_aggressions is not None:
        if blend_after_opponent_aggressions < 1:
            raise ValueError("blend_after_opponent_aggressions must be positive")
        if blend_checkpoint_path is None:
            raise ValueError("blend_after_opponent_aggressions requires a blend checkpoint")
    if blend_facing_bet_only and blend_checkpoint_path is None:
        raise ValueError("blend_facing_bet_only requires a blend checkpoint")
    if blend_players is not None:
        if blend_checkpoint_path is None:
            raise ValueError("blend_players requires a blend checkpoint")
        invalid_players = [player for player in blend_players if player not in (0, 1)]
        if invalid_players:
            raise ValueError("blend_players must contain only players 0 and 1")
    if facing_bet_logit_bias_after_opponent_aggressions is not None:
        if facing_bet_logit_bias_after_opponent_aggressions < 1:
            raise ValueError(
                "facing_bet_logit_bias_after_opponent_aggressions must be positive"
            )
    if facing_bet_logit_bias_min_raise_prob is not None:
        if not 0.0 <= facing_bet_logit_bias_min_raise_prob <= 1.0:
            raise ValueError("facing_bet_logit_bias_min_raise_prob must be in [0, 1]")
    if player_facing_bet_logit_bias_after_opponent_aggressions is not None:
        if player_facing_bet_logit_bias_after_opponent_aggressions < 1:
            raise ValueError(
                "player_facing_bet_logit_bias_after_opponent_aggressions must be positive"
            )
    if player_facing_bet_logit_bias_min_raise_prob is not None:
        if not 0.0 <= player_facing_bet_logit_bias_min_raise_prob <= 1.0:
            raise ValueError(
                "player_facing_bet_logit_bias_min_raise_prob must be in [0, 1]"
            )
    if model_rollout_sims is not None and model_rollout_sims <= 0:
        raise ValueError("model_rollout_sims must be positive")
    if model_rollout_opponent_equity_sims <= 0:
        raise ValueError("model_rollout_opponent_equity_sims must be positive")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_encoder = policy_feature_encoder_from_checkpoint_data(
        checkpoint,
        checkpoint_path=checkpoint_path,
        feature_seed=feature_seed,
    )
    model = HoldemPolicyNet(input_dim=feature_encoder.input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    blend_model = None
    if blend_checkpoint_path is not None:
        blend_checkpoint = torch.load(blend_checkpoint_path, map_location="cpu", weights_only=False)
        validate_blend_checkpoint_compatibility(checkpoint, blend_checkpoint)
        blend_model = HoldemPolicyNet(input_dim=feature_encoder.input_dim)
        blend_model.load_state_dict(blend_checkpoint["model_state_dict"])
        blend_model.eval()
    facing_bet_logit_biases = facing_bet_logit_biases or {}
    player_facing_bet_logit_biases = player_facing_bet_logit_biases or {}

    def select_model_action(
        state: FixedLimitHoldemState,
        *,
        record_diagnostics: bool = True,
    ) -> str:
        features = torch.tensor([feature_encoder.encode(state)], dtype=torch.float32)
        mask = torch.tensor(holdem_legal_action_mask(state), dtype=torch.bool)
        facing_bet = bool(
            mask[HOLDEM_CANONICAL_ACTIONS.index("call")]
            and mask[HOLDEM_CANONICAL_ACTIONS.index("fold")]
        )
        current_player = state.current_player()
        with torch.no_grad():
            logits = model(features).squeeze(0)
            if blend_model is not None:
                blend_logits = blend_model(features).squeeze(0)
                active_blend_weight = blend_weight
                if blend_facing_bet_only and not facing_bet:
                    active_blend_weight = 0.0
                if blend_players is not None and current_player not in blend_players:
                    active_blend_weight = 0.0
                if blend_after_opponent_aggressions is not None:
                    opponent_aggressions = opponent_aggressions_before_current_decision(state)
                    active_blend_weight = (
                        active_blend_weight
                        if opponent_aggressions >= blend_after_opponent_aggressions
                        else 0.0
                    )
                logits = (
                    (1.0 - active_blend_weight) * logits
                    + active_blend_weight * blend_logits
                )
            if facing_bet:
                visible_opponent_aggressions: int | None = None
                pre_bias_raise_prob = None
                if (
                    facing_bet_logit_bias_min_raise_prob is not None
                    or player_facing_bet_logit_bias_min_raise_prob is not None
                ):
                    masked_pre_bias_logits = logits.masked_fill(~mask, -1e9)
                    pre_bias_probs = torch.softmax(masked_pre_bias_logits, dim=0)
                    pre_bias_raise_prob = float(
                        pre_bias_probs[HOLDEM_CANONICAL_ACTIONS.index("raise")]
                        .detach()
                        .cpu()
                    )
                apply_global_biases = True
                if facing_bet_logit_bias_after_opponent_aggressions is not None:
                    visible_opponent_aggressions = (
                        opponent_aggressions_before_current_decision(state)
                    )
                    apply_global_biases = (
                        visible_opponent_aggressions
                        >= facing_bet_logit_bias_after_opponent_aggressions
                    )
                if (
                    apply_global_biases
                    and facing_bet_logit_bias_min_raise_prob is not None
                ):
                    apply_global_biases = (
                        pre_bias_raise_prob is not None
                        and pre_bias_raise_prob >= facing_bet_logit_bias_min_raise_prob
                    )
                if apply_global_biases:
                    for action, bias in facing_bet_logit_biases.items():
                        logits[HOLDEM_CANONICAL_ACTIONS.index(action)] += bias
                apply_player_biases = True
                if (
                    player_facing_bet_logit_bias_after_opponent_aggressions
                    is not None
                ):
                    if visible_opponent_aggressions is None:
                        visible_opponent_aggressions = (
                            opponent_aggressions_before_current_decision(state)
                        )
                    apply_player_biases = (
                        visible_opponent_aggressions
                        >= player_facing_bet_logit_bias_after_opponent_aggressions
                    )
                if (
                    apply_player_biases
                    and player_facing_bet_logit_bias_min_raise_prob is not None
                ):
                    apply_player_biases = (
                        pre_bias_raise_prob is not None
                        and pre_bias_raise_prob
                        >= player_facing_bet_logit_bias_min_raise_prob
                    )
                if apply_player_biases:
                    for (player, action), bias in player_facing_bet_logit_biases.items():
                        if player == current_player:
                            logits[HOLDEM_CANONICAL_ACTIONS.index(action)] += bias
            logits = logits.masked_fill(~mask, -1e9)
            action_index = int(logits.argmax().item())
        if decision_diagnostics is not None and record_diagnostics:
            decision_diagnostics.record(state, logits, mask, action_index)
        return HOLDEM_CANONICAL_ACTIONS[action_index]

    if model_rollout_sims is None:
        return select_model_action

    rollout_rng = random.Random(feature_seed + 90_000_019)
    rollout_opponent_policy = (
        model_rollout_opponent_policy or "tight-turn-river-exact-pot-odds"
    )

    def rollout_select_action(state: FixedLimitHoldemState) -> str:
        action_values = policy_rollout_action_values(
            state,
            rollout_rng,
            simulations=model_rollout_sims,
            continuation_policy_factory=lambda _: (
                lambda rollout_state: select_model_action(
                    rollout_state,
                    record_diagnostics=False,
                )
            ),
            opponent_policy_factory=lambda rng: make_policy(
                rollout_opponent_policy,
                rng,
                model_rollout_opponent_equity_sims,
                model_rollout_opponent_rollout_sims,
                model_rollout_opponent_rollout_margin,
            ),
        )
        legal_actions = state.legal_actions()
        best_action = max(legal_actions, key=lambda action: action_values[action])
        default_action = select_model_action(state)
        if default_action not in legal_actions:
            return best_action
        if action_values[best_action] >= action_values[default_action] + model_rollout_margin:
            return best_action
        return default_action

    return rollout_select_action


def make_opponent_policy(
    name: str,
    rng: random.Random,
    equity_sims: int,
    rollout_sims: int | None = None,
    rollout_margin: float = 1.0,
) -> HoldemPolicy:
    return make_policy(name, rng, equity_sims, rollout_sims, rollout_margin)


def evaluate_model_shard(
    *,
    checkpoint: Path,
    player_checkpoints: tuple[Path, Path],
    hands: int,
    seed: int,
    opponent_policy: str,
    equity_sims: int,
    rollout_sims: int | None,
    rollout_margin: float,
    blend_checkpoint: Path | None,
    blend_weight: float,
    blend_after_opponent_aggressions: int | None,
    blend_facing_bet_only: bool,
    blend_players: tuple[int, ...] | None,
    facing_bet_logit_biases: dict[str, float],
    facing_bet_logit_bias_after_opponent_aggressions: int | None,
    facing_bet_logit_bias_min_raise_prob: float | None,
    player_facing_bet_logit_biases: dict[tuple[int, str], float],
    player_facing_bet_logit_bias_after_opponent_aggressions: int | None,
    player_facing_bet_logit_bias_min_raise_prob: float | None,
    model_rollout_sims: int | None,
    model_rollout_margin: float,
    model_rollout_opponent_policy: str | None,
    model_rollout_opponent_equity_sims: int,
    model_rollout_opponent_rollout_sims: int | None,
    model_rollout_opponent_rollout_margin: float,
    model_decision_diagnostics: bool,
    model_player: int,
    shard_index: int,
) -> dict[str, Any]:
    eval_seed = seed + shard_index * 1_000_003
    opponent_rng = random.Random(eval_seed + 1)
    player_checkpoint = player_checkpoints[model_player]
    decision_diagnostics = (
        ModelDecisionDiagnostics() if model_decision_diagnostics else None
    )
    metrics = {
        "checkpoint": str(checkpoint),
        "player_checkpoint": str(player_checkpoint),
        "player0_checkpoint": str(player_checkpoints[0]),
        "player1_checkpoint": str(player_checkpoints[1]),
        **evaluate_policy_match(
            model_policy=model_policy_from_checkpoint(
                player_checkpoint,
                feature_seed=eval_seed,
                blend_checkpoint_path=blend_checkpoint,
                blend_weight=blend_weight,
                blend_after_opponent_aggressions=blend_after_opponent_aggressions,
                blend_facing_bet_only=blend_facing_bet_only,
                blend_players=blend_players,
                facing_bet_logit_biases=facing_bet_logit_biases,
                facing_bet_logit_bias_after_opponent_aggressions=(
                    facing_bet_logit_bias_after_opponent_aggressions
                ),
                facing_bet_logit_bias_min_raise_prob=(
                    facing_bet_logit_bias_min_raise_prob
                ),
                player_facing_bet_logit_biases=player_facing_bet_logit_biases,
                player_facing_bet_logit_bias_after_opponent_aggressions=(
                    player_facing_bet_logit_bias_after_opponent_aggressions
                ),
                player_facing_bet_logit_bias_min_raise_prob=(
                    player_facing_bet_logit_bias_min_raise_prob
                ),
                model_rollout_sims=model_rollout_sims,
                model_rollout_margin=model_rollout_margin,
                model_rollout_opponent_policy=model_rollout_opponent_policy,
                model_rollout_opponent_equity_sims=model_rollout_opponent_equity_sims,
                model_rollout_opponent_rollout_sims=model_rollout_opponent_rollout_sims,
                model_rollout_opponent_rollout_margin=(
                    model_rollout_opponent_rollout_margin
                ),
                decision_diagnostics=decision_diagnostics,
            ),
            opponent_policy=make_opponent_policy(
                opponent_policy,
                opponent_rng,
                equity_sims,
                rollout_sims,
                rollout_margin,
            ),
            hands=hands,
            seed=eval_seed,
            model_player=model_player,
        ),
        "opponent_policy": opponent_policy,
        "equity_sims": equity_sims,
        "rollout_sims": rollout_sims,
        "rollout_margin": rollout_margin,
        "blend_checkpoint": str(blend_checkpoint) if blend_checkpoint is not None else None,
        "blend_weight": blend_weight if blend_checkpoint is not None else None,
        "blend_after_opponent_aggressions": blend_after_opponent_aggressions,
        "blend_facing_bet_only": blend_facing_bet_only,
        "blend_players": list(blend_players) if blend_players is not None else None,
        "facing_bet_logit_biases": facing_bet_logit_biases,
        "facing_bet_logit_bias_after_opponent_aggressions": (
            facing_bet_logit_bias_after_opponent_aggressions
        ),
        "facing_bet_logit_bias_min_raise_prob": facing_bet_logit_bias_min_raise_prob,
        "player_facing_bet_logit_biases": serialize_player_action_biases(
            player_facing_bet_logit_biases
        ),
        "player_facing_bet_logit_bias_after_opponent_aggressions": (
            player_facing_bet_logit_bias_after_opponent_aggressions
        ),
        "player_facing_bet_logit_bias_min_raise_prob": (
            player_facing_bet_logit_bias_min_raise_prob
        ),
        "model_rollout_sims": model_rollout_sims,
        "model_rollout_margin": model_rollout_margin if model_rollout_sims is not None else None,
        "model_rollout_opponent_policy": (
            model_rollout_opponent_policy if model_rollout_sims is not None else None
        ),
        "model_rollout_opponent_equity_sims": (
            model_rollout_opponent_equity_sims if model_rollout_sims is not None else None
        ),
        "model_rollout_opponent_rollout_sims": (
            model_rollout_opponent_rollout_sims if model_rollout_sims is not None else None
        ),
        "model_rollout_opponent_rollout_margin": (
            model_rollout_opponent_rollout_margin if model_rollout_sims is not None else None
        ),
        "shard_index": shard_index,
    }
    if decision_diagnostics is not None:
        attach_model_decision_diagnostics(metrics, [decision_diagnostics.as_raw()])
    return metrics


def evaluate_model_paired_shard(
    *,
    checkpoint: Path,
    player_checkpoints: tuple[Path, Path],
    hands: int,
    seed: int,
    opponent_policy: str,
    equity_sims: int,
    rollout_sims: int | None,
    rollout_margin: float,
    blend_checkpoint: Path | None,
    blend_weight: float,
    blend_after_opponent_aggressions: int | None,
    blend_facing_bet_only: bool,
    blend_players: tuple[int, ...] | None,
    facing_bet_logit_biases: dict[str, float],
    facing_bet_logit_bias_after_opponent_aggressions: int | None,
    facing_bet_logit_bias_min_raise_prob: float | None,
    player_facing_bet_logit_biases: dict[tuple[int, str], float],
    player_facing_bet_logit_bias_after_opponent_aggressions: int | None,
    player_facing_bet_logit_bias_min_raise_prob: float | None,
    model_rollout_sims: int | None,
    model_rollout_margin: float,
    model_rollout_opponent_policy: str | None,
    model_rollout_opponent_equity_sims: int,
    model_rollout_opponent_rollout_sims: int | None,
    model_rollout_opponent_rollout_margin: float,
    model_decision_diagnostics: bool,
    shard_index: int,
) -> dict[str, Any]:
    eval_seed = seed + shard_index * 1_000_003
    decision_diagnostics = (
        ModelDecisionDiagnostics() if model_decision_diagnostics else None
    )
    model_policies = tuple(
        model_policy_from_checkpoint(
            player_checkpoints[model_player],
            feature_seed=eval_seed + model_player,
            blend_checkpoint_path=blend_checkpoint,
            blend_weight=blend_weight,
            blend_after_opponent_aggressions=blend_after_opponent_aggressions,
            blend_facing_bet_only=blend_facing_bet_only,
            blend_players=blend_players,
            facing_bet_logit_biases=facing_bet_logit_biases,
            facing_bet_logit_bias_after_opponent_aggressions=(
                facing_bet_logit_bias_after_opponent_aggressions
            ),
            facing_bet_logit_bias_min_raise_prob=facing_bet_logit_bias_min_raise_prob,
            player_facing_bet_logit_biases=player_facing_bet_logit_biases,
            player_facing_bet_logit_bias_after_opponent_aggressions=(
                player_facing_bet_logit_bias_after_opponent_aggressions
            ),
            player_facing_bet_logit_bias_min_raise_prob=(
                player_facing_bet_logit_bias_min_raise_prob
            ),
            model_rollout_sims=model_rollout_sims,
            model_rollout_margin=model_rollout_margin,
            model_rollout_opponent_policy=model_rollout_opponent_policy,
            model_rollout_opponent_equity_sims=model_rollout_opponent_equity_sims,
            model_rollout_opponent_rollout_sims=model_rollout_opponent_rollout_sims,
            model_rollout_opponent_rollout_margin=(
                model_rollout_opponent_rollout_margin
            ),
            decision_diagnostics=decision_diagnostics,
        )
        for model_player in (0, 1)
    )
    opponent_policies = tuple(
        make_opponent_policy(
            opponent_policy,
            random.Random(eval_seed + 10 + model_player),
            equity_sims,
            rollout_sims,
            rollout_margin,
        )
        for model_player in (0, 1)
    )
    metrics = {
        "checkpoint": str(checkpoint),
        "player0_checkpoint": str(player_checkpoints[0]),
        "player1_checkpoint": str(player_checkpoints[1]),
        **evaluate_policy_match_paired_seats(
            model_policies=model_policies,
            opponent_policies=opponent_policies,
            hands=hands,
            seed=eval_seed,
        ),
        "opponent_policy": opponent_policy,
        "equity_sims": equity_sims,
        "rollout_sims": rollout_sims,
        "rollout_margin": rollout_margin,
        "blend_checkpoint": str(blend_checkpoint) if blend_checkpoint is not None else None,
        "blend_weight": blend_weight if blend_checkpoint is not None else None,
        "blend_after_opponent_aggressions": blend_after_opponent_aggressions,
        "blend_facing_bet_only": blend_facing_bet_only,
        "blend_players": list(blend_players) if blend_players is not None else None,
        "facing_bet_logit_biases": facing_bet_logit_biases,
        "facing_bet_logit_bias_after_opponent_aggressions": (
            facing_bet_logit_bias_after_opponent_aggressions
        ),
        "facing_bet_logit_bias_min_raise_prob": facing_bet_logit_bias_min_raise_prob,
        "player_facing_bet_logit_biases": serialize_player_action_biases(
            player_facing_bet_logit_biases
        ),
        "player_facing_bet_logit_bias_after_opponent_aggressions": (
            player_facing_bet_logit_bias_after_opponent_aggressions
        ),
        "player_facing_bet_logit_bias_min_raise_prob": (
            player_facing_bet_logit_bias_min_raise_prob
        ),
        "model_rollout_sims": model_rollout_sims,
        "model_rollout_margin": model_rollout_margin if model_rollout_sims is not None else None,
        "model_rollout_opponent_policy": (
            model_rollout_opponent_policy if model_rollout_sims is not None else None
        ),
        "model_rollout_opponent_equity_sims": (
            model_rollout_opponent_equity_sims if model_rollout_sims is not None else None
        ),
        "model_rollout_opponent_rollout_sims": (
            model_rollout_opponent_rollout_sims if model_rollout_sims is not None else None
        ),
        "model_rollout_opponent_rollout_margin": (
            model_rollout_opponent_rollout_margin if model_rollout_sims is not None else None
        ),
        "shard_index": shard_index,
    }
    if decision_diagnostics is not None:
        attach_model_decision_diagnostics(metrics, [decision_diagnostics.as_raw()])
    return metrics


def report_progress(enabled: bool, result: dict[str, Any]) -> None:
    if not enabled:
        return
    print(
        f"player {result['model_player']} shard {result['shard_index']}: "
        f"hands={result['hands']} "
        f"avg_utility_model={result['avg_utility_model']:.3f} "
        f"stderr={result['utility_stderr_model']:.3f}",
        file=sys.stderr,
        flush=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    progress = bool(getattr(args, "progress", False))
    rollout_margin = float(getattr(args, "rollout_margin", 1.0))
    blend_checkpoint = getattr(args, "blend_checkpoint", None)
    blend_weight = float(getattr(args, "blend_weight", 0.5))
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("--blend-weight must be between 0.0 and 1.0")
    blend_after_opponent_aggressions = getattr(args, "blend_after_opponent_aggressions", None)
    if blend_after_opponent_aggressions is not None:
        if blend_after_opponent_aggressions < 1:
            raise ValueError("--blend-after-opponent-aggressions must be positive")
        if blend_checkpoint is None:
            raise ValueError("--blend-after-opponent-aggressions requires --blend-checkpoint")
    blend_facing_bet_only = bool(getattr(args, "blend_facing_bet_only", False))
    if blend_facing_bet_only and blend_checkpoint is None:
        raise ValueError("--blend-facing-bet-only requires --blend-checkpoint")
    blend_players = tuple(getattr(args, "blend_player", []) or [])
    if blend_players and blend_checkpoint is None:
        raise ValueError("--blend-player requires --blend-checkpoint")
    if any(player not in (0, 1) for player in blend_players):
        raise ValueError("--blend-player values must be 0 or 1")
    blend_players_or_none = blend_players or None
    facing_bet_logit_biases = action_logit_biases_from_specs(
        getattr(args, "facing_bet_logit_bias", None)
    )
    facing_bet_logit_bias_after_opponent_aggressions = getattr(
        args,
        "facing_bet_logit_bias_after_opponent_aggressions",
        None,
    )
    if (
        facing_bet_logit_bias_after_opponent_aggressions is not None
        and facing_bet_logit_bias_after_opponent_aggressions < 1
    ):
        raise ValueError(
            "--facing-bet-logit-bias-after-opponent-aggressions must be positive"
        )
    facing_bet_logit_bias_min_raise_prob = getattr(
        args,
        "facing_bet_logit_bias_min_raise_prob",
        None,
    )
    if (
        facing_bet_logit_bias_min_raise_prob is not None
        and not 0.0 <= facing_bet_logit_bias_min_raise_prob <= 1.0
    ):
        raise ValueError("--facing-bet-logit-bias-min-raise-prob must be in [0, 1]")
    player_facing_bet_logit_biases = player_action_logit_biases_from_specs(
        getattr(args, "player_facing_bet_logit_bias", None)
    )
    player_bias_after_opponent_aggressions = getattr(
        args,
        "player_facing_bet_logit_bias_after_opponent_aggressions",
        None,
    )
    if (
        player_bias_after_opponent_aggressions is not None
        and player_bias_after_opponent_aggressions < 1
    ):
        raise ValueError(
            "--player-facing-bet-logit-bias-after-opponent-aggressions must be positive"
        )
    player_facing_bet_logit_bias_min_raise_prob = getattr(
        args,
        "player_facing_bet_logit_bias_min_raise_prob",
        None,
    )
    if (
        player_facing_bet_logit_bias_min_raise_prob is not None
        and not 0.0 <= player_facing_bet_logit_bias_min_raise_prob <= 1.0
    ):
        raise ValueError(
            "--player-facing-bet-logit-bias-min-raise-prob must be in [0, 1]"
        )
    model_rollout_sims = getattr(args, "model_rollout_sims", None)
    if model_rollout_sims is not None and model_rollout_sims <= 0:
        raise ValueError("--model-rollout-sims must be positive")
    model_rollout_margin = float(getattr(args, "model_rollout_margin", 0.0))
    model_rollout_opponent_policy = (
        getattr(args, "model_rollout_opponent_policy", None) or args.opponent_policy
    )
    model_rollout_opponent_equity_sims = (
        args.equity_sims
        if getattr(args, "model_rollout_opponent_equity_sims", None) is None
        else args.model_rollout_opponent_equity_sims
    )
    if model_rollout_opponent_equity_sims <= 0:
        raise ValueError("--model-rollout-opponent-equity-sims must be positive")
    model_rollout_opponent_rollout_sims = (
        args.rollout_sims
        if getattr(args, "model_rollout_opponent_rollout_sims", None) is None
        else args.model_rollout_opponent_rollout_sims
    )
    model_rollout_opponent_rollout_margin_arg = getattr(
        args,
        "model_rollout_opponent_rollout_margin",
        None,
    )
    model_rollout_opponent_rollout_margin = (
        rollout_margin
        if model_rollout_opponent_rollout_margin_arg is None
        else float(model_rollout_opponent_rollout_margin_arg)
    )
    model_decision_diagnostics = bool(
        getattr(args, "model_decision_diagnostics", False)
    )
    model_players = normalize_model_players(args.model_player)
    player_checkpoints = player_checkpoints_from_args(args)
    shard_hands = split_hands(args.hands, args.jobs)
    if args.paired_seats and model_players != (0, 1):
        raise ValueError("--paired-seats requires --model-player both")
    if args.paired_seats:
        shard_kwargs = [
            {
                "checkpoint": args.checkpoint,
                "player_checkpoints": player_checkpoints,
                "hands": shard_size,
                "seed": args.seed,
                "opponent_policy": args.opponent_policy,
                "equity_sims": args.equity_sims,
                "rollout_sims": args.rollout_sims,
                "rollout_margin": rollout_margin,
                "blend_checkpoint": blend_checkpoint,
                "blend_weight": blend_weight,
                "blend_after_opponent_aggressions": blend_after_opponent_aggressions,
                "blend_facing_bet_only": blend_facing_bet_only,
                "blend_players": blend_players_or_none,
                "facing_bet_logit_biases": facing_bet_logit_biases,
                "facing_bet_logit_bias_after_opponent_aggressions": (
                    facing_bet_logit_bias_after_opponent_aggressions
                ),
                "facing_bet_logit_bias_min_raise_prob": (
                    facing_bet_logit_bias_min_raise_prob
                ),
                "player_facing_bet_logit_biases": player_facing_bet_logit_biases,
                "player_facing_bet_logit_bias_after_opponent_aggressions": (
                    player_bias_after_opponent_aggressions
                ),
                "player_facing_bet_logit_bias_min_raise_prob": (
                    player_facing_bet_logit_bias_min_raise_prob
                ),
                "model_rollout_sims": model_rollout_sims,
                "model_rollout_margin": model_rollout_margin,
                "model_rollout_opponent_policy": model_rollout_opponent_policy,
                "model_rollout_opponent_equity_sims": model_rollout_opponent_equity_sims,
                "model_rollout_opponent_rollout_sims": (
                    model_rollout_opponent_rollout_sims
                ),
                "model_rollout_opponent_rollout_margin": (
                    model_rollout_opponent_rollout_margin
                ),
                "model_decision_diagnostics": model_decision_diagnostics,
                "shard_index": shard_index,
            }
            for shard_index, shard_size in enumerate(shard_hands)
        ]
        if args.jobs == 1:
            shard_metrics = []
            for kwargs in shard_kwargs:
                result = evaluate_model_paired_shard(**kwargs)
                shard_metrics.append(result)
                report_progress(progress, result)
        else:
            with ProcessPoolExecutor(
                max_workers=args.jobs,
                mp_context=evaluation_process_context(),
            ) as executor:
                futures = [
                    executor.submit(evaluate_model_paired_shard, **kwargs)
                    for kwargs in shard_kwargs
                ]
                shard_metrics = []
                for future in as_completed(futures):
                    result = future.result()
                    shard_metrics.append(result)
                    report_progress(progress, result)
            shard_metrics.sort(key=lambda item: item["shard_index"])
        metrics = aggregate_policy_match_shards(shard_metrics)
        attach_model_decision_diagnostics(
            metrics,
            [
                item["model_decision_diagnostics_raw"]
                for item in shard_metrics
                if "model_decision_diagnostics_raw" in item
            ],
        )
        metrics["jobs"] = args.jobs
        metrics["shard_hands"] = shard_hands
        metrics["paired_seats"] = True
        metrics["model_decision_diagnostics_enabled"] = model_decision_diagnostics
        if args.out is not None:
            write_json(args.out, metrics)
        return metrics

    shard_metrics_by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    shard_kwargs = [
        {
            "checkpoint": args.checkpoint,
            "player_checkpoints": player_checkpoints,
            "hands": shard_size,
            "seed": args.seed,
            "opponent_policy": args.opponent_policy,
            "equity_sims": args.equity_sims,
            "rollout_sims": args.rollout_sims,
            "rollout_margin": rollout_margin,
            "blend_checkpoint": blend_checkpoint,
            "blend_weight": blend_weight,
            "blend_after_opponent_aggressions": blend_after_opponent_aggressions,
            "blend_facing_bet_only": blend_facing_bet_only,
            "blend_players": blend_players_or_none,
            "facing_bet_logit_biases": facing_bet_logit_biases,
            "facing_bet_logit_bias_after_opponent_aggressions": (
                facing_bet_logit_bias_after_opponent_aggressions
            ),
            "facing_bet_logit_bias_min_raise_prob": (
                facing_bet_logit_bias_min_raise_prob
            ),
            "player_facing_bet_logit_biases": player_facing_bet_logit_biases,
            "player_facing_bet_logit_bias_after_opponent_aggressions": (
                player_bias_after_opponent_aggressions
            ),
            "player_facing_bet_logit_bias_min_raise_prob": (
                player_facing_bet_logit_bias_min_raise_prob
            ),
            "model_rollout_sims": model_rollout_sims,
            "model_rollout_margin": model_rollout_margin,
            "model_rollout_opponent_policy": model_rollout_opponent_policy,
            "model_rollout_opponent_equity_sims": model_rollout_opponent_equity_sims,
            "model_rollout_opponent_rollout_sims": model_rollout_opponent_rollout_sims,
            "model_rollout_opponent_rollout_margin": (
                model_rollout_opponent_rollout_margin
            ),
            "model_decision_diagnostics": model_decision_diagnostics,
            "model_player": model_player,
            "shard_index": shard_index,
        }
        for model_player in model_players
        for shard_index, shard_size in enumerate(shard_hands)
    ]

    if args.jobs == 1:
        for kwargs in shard_kwargs:
            result = evaluate_model_shard(**kwargs)
            shard_metrics_by_player[int(result["model_player"])].append(result)
            report_progress(progress, result)
    else:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=evaluation_process_context(),
        ) as executor:
            futures = [executor.submit(evaluate_model_shard, **kwargs) for kwargs in shard_kwargs]
            for future in as_completed(futures):
                result = future.result()
                shard_metrics_by_player[int(result["model_player"])].append(result)
                report_progress(progress, result)

    seat_metrics = []
    for model_player in model_players:
        player_shards = sorted(
            shard_metrics_by_player[model_player],
            key=lambda item: item["shard_index"],
        )
        seat_metric = aggregate_policy_match_shards(player_shards)
        attach_model_decision_diagnostics(
            seat_metric,
            [
                item["model_decision_diagnostics_raw"]
                for item in player_shards
                if "model_decision_diagnostics_raw" in item
            ],
        )
        seat_metrics.append(seat_metric)
    metrics = aggregate_model_player_metrics(seat_metrics)
    metrics["jobs"] = args.jobs
    metrics["shard_hands"] = shard_hands
    metrics["paired_seats"] = False
    metrics["model_decision_diagnostics_enabled"] = model_decision_diagnostics
    if args.out is not None:
        write_json(args.out, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--player0-checkpoint", type=Path)
    parser.add_argument("--player1-checkpoint", type=Path)
    parser.add_argument("--blend-checkpoint", type=Path)
    parser.add_argument("--blend-weight", type=float, default=0.5)
    parser.add_argument("--blend-after-opponent-aggressions", type=int)
    parser.add_argument(
        "--blend-facing-bet-only",
        action="store_true",
        help="Apply checkpoint blending only while the model is facing a bet or raise.",
    )
    parser.add_argument(
        "--blend-player",
        action="append",
        default=[],
        type=int,
        choices=[0, 1],
        help="Apply checkpoint blending only for this current player; repeat for both.",
    )
    parser.add_argument(
        "--facing-bet-logit-bias",
        action="append",
        default=[],
        metavar="ACTION=BIAS",
        help="Add a logit bias while facing a bet, for example call=0.5.",
    )
    parser.add_argument(
        "--facing-bet-logit-bias-after-opponent-aggressions",
        type=int,
        help=(
            "Apply global facing-bet logit biases only after at least this many "
            "visible opponent bets or raises."
        ),
    )
    parser.add_argument(
        "--facing-bet-logit-bias-min-raise-prob",
        type=float,
        help=(
            "Apply global facing-bet logit biases only when the pre-bias model "
            "raise probability is at least this value."
        ),
    )
    parser.add_argument(
        "--player-facing-bet-logit-bias",
        action="append",
        default=[],
        metavar="PLAYER:ACTION=BIAS",
        help=(
            "Add a player-specific logit bias while facing a bet, "
            "for example 0:raise=-1.0."
        ),
    )
    parser.add_argument(
        "--player-facing-bet-logit-bias-after-opponent-aggressions",
        type=int,
        help=(
            "Apply player-specific facing-bet logit biases only after at least "
            "this many visible opponent bets or raises."
        ),
    )
    parser.add_argument(
        "--player-facing-bet-logit-bias-min-raise-prob",
        type=float,
        help=(
            "Apply player-specific facing-bet logit biases only when the "
            "pre-bias model raise probability is at least this value."
        ),
    )
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="random")
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--rollout-margin", type=float, default=1.0)
    parser.add_argument(
        "--model-rollout-sims",
        type=int,
        help="Wrap the neural policy in one-step belief rollouts with this many sims/action.",
    )
    parser.add_argument(
        "--model-rollout-margin",
        type=float,
        default=0.0,
        help="Require this action-value improvement over the neural default action.",
    )
    parser.add_argument(
        "--model-rollout-opponent-policy",
        choices=HOLDEM_SELF_PLAY_POLICIES,
        help="Opponent model used inside neural policy rollouts; defaults to eval opponent.",
    )
    parser.add_argument("--model-rollout-opponent-equity-sims", type=int)
    parser.add_argument("--model-rollout-opponent-rollout-sims", type=int)
    parser.add_argument(
        "--model-rollout-opponent-rollout-margin",
        type=float,
        help="Rollout margin for the opponent policy used inside neural policy rollouts.",
    )
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument(
        "--model-decision-diagnostics",
        action="store_true",
        help="Record aggregate neural action probabilities and logit margins by state bucket.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--paired-seats", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
