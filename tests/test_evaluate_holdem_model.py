import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.evaluate_holdem_model import (  # noqa: E402
    action_logit_biases_from_specs,
    build_parser,
    evaluation_process_context,
    make_opponent_policy,
    model_policy_from_checkpoint,
    opponent_aggressions_before_current_decision,
    parse_model_players,
    player_action_logit_biases_from_specs,
    run,
    split_hands,
)
from alphapoker.holdem import RAISE, deal_fixed_limit_holdem  # noqa: E402
from alphapoker.holdem_features import (  # noqa: E402
    HOLDEM_ACTION_HISTORY_FEATURE_DIM,
    HOLDEM_CANONICAL_ACTIONS,
    HOLDEM_FEATURE_DIM,
)
from alphapoker.holdem_model import HoldemEquityNet, HoldemPolicyNet  # noqa: E402


def write_biased_policy_checkpoint(
    path,
    action: str,
    *,
    input_dim: int = HOLDEM_FEATURE_DIM,
    action_history_features: bool = False,
) -> None:
    model = HoldemPolicyNet(input_dim=input_dim)
    for parameter in model.parameters():
        parameter.data.zero_()
    bias = model.net[-1].bias
    assert bias is not None
    bias.data[HOLDEM_CANONICAL_ACTIONS.index(action)] = 10.0
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
    }
    if action_history_features:
        checkpoint["action_history_features"] = True
    torch.save(checkpoint, path)


def test_make_opponent_policy_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        make_opponent_policy("bad", __import__("random").Random(0), 8)


def test_holdem_model_eval_split_hands_balances_jobs() -> None:
    assert split_hands(10, 4) == [3, 3, 2, 2]


def test_holdem_model_eval_uses_safe_parallel_context() -> None:
    assert evaluation_process_context().get_start_method() in {"forkserver", "spawn"}


def test_holdem_model_eval_parser_accepts_model_player() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--blend-checkpoint",
            "blend.pt",
            "--player0-blend-checkpoint",
            "p0_blend.pt",
            "--player1-blend-checkpoint",
            "p1_blend.pt",
            "--blend-weight",
            "0.25",
            "--blend-after-opponent-aggressions",
            "1",
            "--blend-facing-bet-only",
            "--blend-player",
            "1",
            "--facing-bet-logit-bias",
            "call=0.5",
            "--facing-bet-logit-bias-after-opponent-aggressions",
            "2",
            "--facing-bet-logit-bias-min-raise-prob",
            "0.15",
            "--player-facing-bet-logit-bias",
            "0:raise=-1.0",
            "--player-facing-bet-logit-bias-after-opponent-aggressions",
            "2",
            "--player-facing-bet-logit-bias-min-raise-prob",
            "0.2",
            "--model-rollout-sims",
            "1",
            "--model-rollout-margin",
            "0.5",
            "--model-rollout-opponent-policy",
            "tight-range-pot-odds",
            "--model-rollout-opponent-equity-sims",
            "2",
            "--model-rollout-opponent-rollout-sims",
            "1",
            "--model-rollout-opponent-rollout-margin",
            "1.5",
            "--model-decision-diagnostics",
            "--model-player",
            "1",
        ]
    )

    assert args.model_player == (1,)
    assert str(args.blend_checkpoint) == "blend.pt"
    assert str(args.player0_blend_checkpoint) == "p0_blend.pt"
    assert str(args.player1_blend_checkpoint) == "p1_blend.pt"
    assert args.blend_weight == 0.25
    assert args.blend_after_opponent_aggressions == 1
    assert args.blend_facing_bet_only
    assert args.blend_player == [1]
    assert args.facing_bet_logit_bias == ["call=0.5"]
    assert args.facing_bet_logit_bias_after_opponent_aggressions == 2
    assert args.facing_bet_logit_bias_min_raise_prob == 0.15
    assert args.player_facing_bet_logit_bias == ["0:raise=-1.0"]
    assert args.player_facing_bet_logit_bias_after_opponent_aggressions == 2
    assert args.player_facing_bet_logit_bias_min_raise_prob == 0.2
    assert args.model_rollout_sims == 1
    assert args.model_rollout_margin == 0.5
    assert args.model_rollout_opponent_policy == "tight-range-pot-odds"
    assert args.model_rollout_opponent_equity_sims == 2
    assert args.model_rollout_opponent_rollout_sims == 1
    assert args.model_rollout_opponent_rollout_margin == 1.5
    assert args.model_decision_diagnostics


def test_holdem_model_eval_parser_accepts_both_model_players() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--model-player",
            "both",
            "--jobs",
            "3",
            "--paired-seats",
            "--progress",
        ]
    )

    assert args.model_player == (0, 1)
    assert args.jobs == 3
    assert args.paired_seats
    assert args.progress
    assert parse_model_players("both") == (0, 1)


def test_holdem_model_eval_parser_accepts_player_checkpoints() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "fallback.pt",
            "--player0-checkpoint",
            "p0.pt",
            "--player1-checkpoint",
            "p1.pt",
        ]
    )

    assert str(args.checkpoint) == "fallback.pt"
    assert str(args.player0_checkpoint) == "p0.pt"
    assert str(args.player1_checkpoint) == "p1.pt"


def test_holdem_model_eval_parser_accepts_pot_odds_opponent() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--opponent-policy",
            "pot-odds",
        ]
    )

    assert args.opponent_policy == "pot-odds"


def test_holdem_model_eval_parser_accepts_rollout_opponent() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--opponent-policy",
            "rollout-pot-odds",
            "--rollout-sims",
            "2",
            "--rollout-margin",
            "1.5",
        ]
    )

    assert args.opponent_policy == "rollout-pot-odds"
    assert args.rollout_sims == 2
    assert args.rollout_margin == 1.5


def test_holdem_model_eval_parser_accepts_tuned_pot_odds_opponent() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--opponent-policy",
            "tuned-pot-odds",
        ]
    )

    assert args.opponent_policy == "tuned-pot-odds"


def test_model_policy_loads_relative_feature_equity_checkpoint(tmp_path) -> None:
    equity_checkpoint = tmp_path / "equity.pt"
    policy_checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "model_state_dict": HoldemEquityNet().state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM,
        },
        equity_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM + 1).state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM + 1,
            "feature_equity_sims": None,
            "feature_equity_checkpoint": "equity.pt",
        },
        policy_checkpoint,
    )

    state = deal_fixed_limit_holdem()
    action = model_policy_from_checkpoint(policy_checkpoint)(state)

    assert action in state.legal_actions()


def test_model_policy_loads_turn_river_exact_feature_mode(tmp_path) -> None:
    policy_checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM + 1).state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM + 1,
            "feature_equity_sims": 2,
            "feature_equity_mode": "turn-river-exact",
            "feature_equity_checkpoint": None,
        },
        policy_checkpoint,
    )

    state = deal_fixed_limit_holdem()
    action = model_policy_from_checkpoint(policy_checkpoint)(state)

    assert action in state.legal_actions()


def test_model_policy_loads_tight_range_feature_mode(tmp_path) -> None:
    policy_checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM + 1).state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM + 1,
            "feature_equity_sims": 2,
            "feature_equity_mode": "tight-range",
            "feature_equity_checkpoint": None,
        },
        policy_checkpoint,
    )

    state = deal_fixed_limit_holdem()
    action = model_policy_from_checkpoint(policy_checkpoint)(state)

    assert action in state.legal_actions()


def test_model_policy_can_wrap_with_rollouts(tmp_path) -> None:
    policy_checkpoint = tmp_path / "policy.pt"
    write_biased_policy_checkpoint(policy_checkpoint, "call")
    policy = model_policy_from_checkpoint(
        policy_checkpoint,
        model_rollout_sims=1,
        model_rollout_margin=0.0,
        model_rollout_opponent_policy="tight-turn-river-exact-pot-odds",
        model_rollout_opponent_equity_sims=1,
    )

    state = deal_fixed_limit_holdem(__import__("random").Random(21))
    action = policy(state)

    assert action in state.legal_actions()


def test_model_policy_blends_compatible_checkpoints(tmp_path) -> None:
    state = deal_fixed_limit_holdem()
    legal_actions = state.legal_actions()
    primary_action = legal_actions[0]
    blend_action = legal_actions[-1]
    primary_checkpoint = tmp_path / "primary.pt"
    blend_checkpoint = tmp_path / "blend.pt"
    write_biased_policy_checkpoint(primary_checkpoint, primary_action)
    write_biased_policy_checkpoint(blend_checkpoint, blend_action)

    primary_policy = model_policy_from_checkpoint(
        primary_checkpoint,
        blend_checkpoint_path=blend_checkpoint,
        blend_weight=0.0,
    )
    blend_policy = model_policy_from_checkpoint(
        primary_checkpoint,
        blend_checkpoint_path=blend_checkpoint,
        blend_weight=1.0,
    )

    assert primary_policy(state) == primary_action
    assert blend_policy(state) == blend_action


def test_model_policy_can_blend_after_opponent_aggression(tmp_path) -> None:
    state = deal_fixed_limit_holdem()
    primary_action = state.legal_actions()[0]
    blend_action = state.legal_actions()[-1]
    primary_checkpoint = tmp_path / "primary.pt"
    blend_checkpoint = tmp_path / "blend.pt"
    write_biased_policy_checkpoint(primary_checkpoint, primary_action)
    write_biased_policy_checkpoint(blend_checkpoint, blend_action)

    policy = model_policy_from_checkpoint(
        primary_checkpoint,
        blend_checkpoint_path=blend_checkpoint,
        blend_weight=1.0,
        blend_after_opponent_aggressions=1,
    )

    assert opponent_aggressions_before_current_decision(state) == 0
    assert policy(state) == primary_action
    raised_state = state.apply(RAISE)
    assert opponent_aggressions_before_current_decision(raised_state) == 1
    assert policy(raised_state) == blend_action


def test_model_policy_can_blend_only_while_facing_bet(tmp_path) -> None:
    state = deal_fixed_limit_holdem()
    non_facing_state = state.apply("call")
    facing_state = state.apply(RAISE)
    primary_action = non_facing_state.legal_actions()[0]
    blend_action = facing_state.legal_actions()[-1]
    primary_checkpoint = tmp_path / "primary.pt"
    blend_checkpoint = tmp_path / "blend.pt"
    write_biased_policy_checkpoint(primary_checkpoint, primary_action)
    write_biased_policy_checkpoint(blend_checkpoint, blend_action)

    policy = model_policy_from_checkpoint(
        primary_checkpoint,
        blend_checkpoint_path=blend_checkpoint,
        blend_weight=1.0,
        blend_facing_bet_only=True,
    )

    assert policy(non_facing_state) == primary_action
    assert policy(facing_state) == blend_action


def test_model_policy_can_blend_only_for_selected_player(tmp_path) -> None:
    state = deal_fixed_limit_holdem().apply(RAISE)
    current_player = state.current_player()
    other_player = 1 - current_player
    primary_action = state.legal_actions()[0]
    blend_action = state.legal_actions()[-1]
    primary_checkpoint = tmp_path / "primary.pt"
    blend_checkpoint = tmp_path / "blend.pt"
    write_biased_policy_checkpoint(primary_checkpoint, primary_action)
    write_biased_policy_checkpoint(blend_checkpoint, blend_action)

    inactive_policy = model_policy_from_checkpoint(
        primary_checkpoint,
        blend_checkpoint_path=blend_checkpoint,
        blend_weight=1.0,
        blend_players=(other_player,),
    )
    active_policy = model_policy_from_checkpoint(
        primary_checkpoint,
        blend_checkpoint_path=blend_checkpoint,
        blend_weight=1.0,
        blend_players=(current_player,),
    )

    assert inactive_policy(state) == primary_action
    assert active_policy(state) == blend_action


def test_model_policy_applies_facing_bet_logit_bias(tmp_path) -> None:
    state = deal_fixed_limit_holdem()
    raised_state = state.apply(RAISE)
    policy_checkpoint = tmp_path / "policy.pt"
    write_biased_policy_checkpoint(policy_checkpoint, "fold")

    policy = model_policy_from_checkpoint(
        policy_checkpoint,
        facing_bet_logit_biases=action_logit_biases_from_specs(["call=20.0"]),
    )

    assert policy(raised_state) == "call"


def test_model_policy_can_delay_facing_bet_logit_bias(tmp_path) -> None:
    state = deal_fixed_limit_holdem()
    raised_state = state.apply(RAISE)
    policy_checkpoint = tmp_path / "policy.pt"
    write_biased_policy_checkpoint(policy_checkpoint, "fold")

    delayed_policy = model_policy_from_checkpoint(
        policy_checkpoint,
        facing_bet_logit_biases=action_logit_biases_from_specs(["call=20.0"]),
        facing_bet_logit_bias_after_opponent_aggressions=2,
    )
    active_policy = model_policy_from_checkpoint(
        policy_checkpoint,
        facing_bet_logit_biases=action_logit_biases_from_specs(["call=20.0"]),
        facing_bet_logit_bias_after_opponent_aggressions=1,
    )

    assert opponent_aggressions_before_current_decision(raised_state) == 1
    assert delayed_policy(raised_state) == "fold"
    assert active_policy(raised_state) == "call"


def test_model_policy_can_gate_facing_bet_logit_bias_by_raise_prob(tmp_path) -> None:
    state = deal_fixed_limit_holdem()
    raised_state = state.apply(RAISE)
    policy_checkpoint = tmp_path / "policy.pt"
    write_biased_policy_checkpoint(policy_checkpoint, "fold")

    blocked_policy = model_policy_from_checkpoint(
        policy_checkpoint,
        facing_bet_logit_biases=action_logit_biases_from_specs(["call=20.0"]),
        facing_bet_logit_bias_min_raise_prob=0.1,
    )
    active_policy = model_policy_from_checkpoint(
        policy_checkpoint,
        facing_bet_logit_biases=action_logit_biases_from_specs(["call=20.0"]),
        facing_bet_logit_bias_min_raise_prob=0.0,
    )

    assert blocked_policy(raised_state) == "fold"
    assert active_policy(raised_state) == "call"


def test_model_policy_applies_player_facing_bet_logit_bias(tmp_path) -> None:
    state = deal_fixed_limit_holdem()
    raised_state = state.apply(RAISE)
    current_player = raised_state.current_player()
    policy_checkpoint = tmp_path / "policy.pt"
    write_biased_policy_checkpoint(policy_checkpoint, "fold")

    policy = model_policy_from_checkpoint(
        policy_checkpoint,
        player_facing_bet_logit_biases=player_action_logit_biases_from_specs(
            [f"{current_player}:call=20.0"]
        ),
    )

    assert policy(raised_state) == "call"


def test_model_policy_can_delay_player_facing_bet_logit_bias(tmp_path) -> None:
    state = deal_fixed_limit_holdem()
    raised_state = state.apply(RAISE)
    current_player = raised_state.current_player()
    policy_checkpoint = tmp_path / "policy.pt"
    write_biased_policy_checkpoint(policy_checkpoint, "fold")

    delayed_policy = model_policy_from_checkpoint(
        policy_checkpoint,
        player_facing_bet_logit_biases=player_action_logit_biases_from_specs(
            [f"{current_player}:call=20.0"]
        ),
        player_facing_bet_logit_bias_after_opponent_aggressions=2,
    )
    active_policy = model_policy_from_checkpoint(
        policy_checkpoint,
        player_facing_bet_logit_biases=player_action_logit_biases_from_specs(
            [f"{current_player}:call=20.0"]
        ),
        player_facing_bet_logit_bias_after_opponent_aggressions=1,
    )

    assert opponent_aggressions_before_current_decision(raised_state) == 1
    assert delayed_policy(raised_state) == "fold"
    assert active_policy(raised_state) == "call"


def test_model_policy_can_gate_player_facing_bet_logit_bias_by_raise_prob(
    tmp_path,
) -> None:
    state = deal_fixed_limit_holdem()
    raised_state = state.apply(RAISE)
    current_player = raised_state.current_player()
    policy_checkpoint = tmp_path / "policy.pt"
    write_biased_policy_checkpoint(policy_checkpoint, "fold")

    blocked_policy = model_policy_from_checkpoint(
        policy_checkpoint,
        player_facing_bet_logit_biases=player_action_logit_biases_from_specs(
            [f"{current_player}:call=20.0"]
        ),
        player_facing_bet_logit_bias_min_raise_prob=0.1,
    )
    active_policy = model_policy_from_checkpoint(
        policy_checkpoint,
        player_facing_bet_logit_biases=player_action_logit_biases_from_specs(
            [f"{current_player}:call=20.0"]
        ),
        player_facing_bet_logit_bias_min_raise_prob=0.0,
    )

    assert blocked_policy(raised_state) == "fold"
    assert active_policy(raised_state) == "call"


def test_model_policy_blend_rejects_incompatible_features(tmp_path) -> None:
    state = deal_fixed_limit_holdem()
    legal_action = state.legal_actions()[0]
    primary_checkpoint = tmp_path / "primary.pt"
    blend_checkpoint = tmp_path / "blend.pt"
    write_biased_policy_checkpoint(primary_checkpoint, legal_action)
    write_biased_policy_checkpoint(
        blend_checkpoint,
        legal_action,
        input_dim=HOLDEM_FEATURE_DIM + 1,
    )

    with pytest.raises(ValueError, match="compatible feature metadata"):
        model_policy_from_checkpoint(
            primary_checkpoint,
            blend_checkpoint_path=blend_checkpoint,
        )(state)


def test_holdem_model_eval_run_smoke(tmp_path) -> None:
    policy_checkpoint = tmp_path / "policy.pt"
    blend_checkpoint = tmp_path / "blend.pt"
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM).state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM,
        },
        policy_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM).state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM,
        },
        blend_checkpoint,
    )

    metrics = run(
        build_parser().parse_args(
            [
                "--checkpoint",
                str(policy_checkpoint),
                "--player1-checkpoint",
                str(blend_checkpoint),
                "--blend-checkpoint",
                str(blend_checkpoint),
                "--blend-weight",
                "0.25",
                "--blend-after-opponent-aggressions",
                "1",
                "--blend-facing-bet-only",
                "--blend-player",
                "1",
                "--hands",
                "1",
                "--model-player",
                "both",
                "--opponent-policy",
                "random",
            ]
        )
    )

    assert metrics["hands"] == 2
    assert metrics["hands_per_model_player"] == 1
    assert metrics["jobs"] == 1
    assert metrics["shard_hands"] == [1]
    assert metrics["player0_checkpoint"] == str(policy_checkpoint)
    assert metrics["player1_checkpoint"] == str(blend_checkpoint)
    assert metrics["blend_checkpoint"] == str(blend_checkpoint)
    assert metrics["blend_weight"] == 0.25
    assert metrics["blend_after_opponent_aggressions"] == 1
    assert metrics["blend_facing_bet_only"]
    assert metrics["blend_players"] == [1]
    assert not metrics["paired_seats"]
    assert sum(metrics["model_action_counts"].values()) > 0
    assert sum(metrics["opponent_action_counts"].values()) > 0


def test_holdem_model_eval_records_model_decision_diagnostics(tmp_path) -> None:
    policy_checkpoint = tmp_path / "policy.pt"
    write_biased_policy_checkpoint(policy_checkpoint, "bet")

    metrics = run(
        build_parser().parse_args(
            [
                "--checkpoint",
                str(policy_checkpoint),
                "--hands",
                "1",
                "--model-player",
                "both",
                "--opponent-policy",
                "random",
                "--model-decision-diagnostics",
            ]
        )
    )

    diagnostics = metrics["model_decision_diagnostics"]
    all_bucket = diagnostics["all"]
    assert metrics["model_decision_diagnostics_enabled"]
    assert all_bucket["count"] == sum(metrics["model_action_counts"].values())
    assert all_bucket["action_counts"] == metrics["model_action_counts"]
    assert set(all_bucket["avg_action_probs"]) == set(HOLDEM_CANONICAL_ACTIONS)
    assert "avg_top_logit_margin" in all_bucket
    assert "player_0" in diagnostics
    assert "player_1" in diagnostics


def test_holdem_model_eval_run_paired_seats_smoke(tmp_path) -> None:
    policy_checkpoint = tmp_path / "policy.pt"
    blend_checkpoint = tmp_path / "blend.pt"
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM).state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM,
        },
        policy_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM).state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM,
        },
        blend_checkpoint,
    )

    metrics = run(
        build_parser().parse_args(
            [
                "--checkpoint",
                str(policy_checkpoint),
                "--player1-checkpoint",
                str(blend_checkpoint),
                "--blend-checkpoint",
                str(blend_checkpoint),
                "--blend-after-opponent-aggressions",
                "1",
                "--blend-facing-bet-only",
                "--blend-player",
                "1",
                "--hands",
                "2",
                "--model-player",
                "both",
                "--paired-seats",
                "--opponent-policy",
                "random",
                "--jobs",
                "2",
            ]
        )
    )

    assert metrics["model_player"] == "both"
    assert metrics["hands"] == 4
    assert metrics["hands_per_model_player"] == 2
    assert metrics["paired_deals"] == 2
    assert metrics["paired_seats"]
    assert metrics["jobs"] == 2
    assert metrics["shard_hands"] == [1, 1]
    assert metrics["player0_checkpoint"] == str(policy_checkpoint)
    assert metrics["player1_checkpoint"] == str(blend_checkpoint)
    assert metrics["blend_after_opponent_aggressions"] == 1
    assert metrics["blend_facing_bet_only"]
    assert metrics["blend_players"] == [1]
    assert len(metrics["seat_metrics"]) == 2
    assert sum(metrics["model_action_counts"].values()) == sum(
        sum(seat["model_action_counts"].values()) for seat in metrics["seat_metrics"]
    )


def test_holdem_model_eval_skips_inactive_player_blend_checkpoint(tmp_path) -> None:
    player0_checkpoint = tmp_path / "player0.pt"
    player1_checkpoint = tmp_path / "player1.pt"
    blend_checkpoint = tmp_path / "blend.pt"
    write_biased_policy_checkpoint(
        player0_checkpoint,
        "bet",
        input_dim=HOLDEM_FEATURE_DIM + HOLDEM_ACTION_HISTORY_FEATURE_DIM,
        action_history_features=True,
    )
    write_biased_policy_checkpoint(player1_checkpoint, "bet")
    write_biased_policy_checkpoint(blend_checkpoint, "raise")

    metrics = run(
        build_parser().parse_args(
            [
                "--checkpoint",
                str(player1_checkpoint),
                "--player0-checkpoint",
                str(player0_checkpoint),
                "--player1-checkpoint",
                str(player1_checkpoint),
                "--blend-checkpoint",
                str(blend_checkpoint),
                "--blend-facing-bet-only",
                "--blend-player",
                "1",
                "--hands",
                "1",
                "--model-player",
                "both",
                "--paired-seats",
                "--opponent-policy",
                "random",
            ]
        )
    )

    assert metrics["paired_seats"]
    assert metrics["player0_checkpoint"] == str(player0_checkpoint)
    assert metrics["player1_checkpoint"] == str(player1_checkpoint)
    assert metrics["blend_checkpoint"] == str(blend_checkpoint)
    assert metrics["blend_players"] == [1]


def test_holdem_model_eval_uses_player_specific_blend_checkpoints(tmp_path) -> None:
    player0_checkpoint = tmp_path / "player0.pt"
    player1_checkpoint = tmp_path / "player1.pt"
    player0_blend_checkpoint = tmp_path / "player0_blend.pt"
    player1_blend_checkpoint = tmp_path / "player1_blend.pt"
    write_biased_policy_checkpoint(
        player0_checkpoint,
        "bet",
        input_dim=HOLDEM_FEATURE_DIM + HOLDEM_ACTION_HISTORY_FEATURE_DIM,
        action_history_features=True,
    )
    write_biased_policy_checkpoint(player1_checkpoint, "bet")
    write_biased_policy_checkpoint(
        player0_blend_checkpoint,
        "raise",
        input_dim=HOLDEM_FEATURE_DIM + HOLDEM_ACTION_HISTORY_FEATURE_DIM,
        action_history_features=True,
    )
    write_biased_policy_checkpoint(player1_blend_checkpoint, "raise")

    metrics = run(
        build_parser().parse_args(
            [
                "--checkpoint",
                str(player1_checkpoint),
                "--player0-checkpoint",
                str(player0_checkpoint),
                "--player1-checkpoint",
                str(player1_checkpoint),
                "--player0-blend-checkpoint",
                str(player0_blend_checkpoint),
                "--player1-blend-checkpoint",
                str(player1_blend_checkpoint),
                "--blend-facing-bet-only",
                "--hands",
                "1",
                "--model-player",
                "both",
                "--opponent-policy",
                "random",
            ]
        )
    )

    assert metrics["player0_blend_checkpoint"] == str(player0_blend_checkpoint)
    assert metrics["player1_blend_checkpoint"] == str(player1_blend_checkpoint)
    assert metrics["blend_checkpoint"] is None
    active_by_player = {
        seat["model_player"]: seat["active_blend_checkpoint"]
        for seat in metrics["seat_metrics"]
    }
    assert active_by_player == {
        0: str(player0_blend_checkpoint),
        1: str(player1_blend_checkpoint),
    }


def test_holdem_model_eval_paired_seats_requires_both(tmp_path) -> None:
    policy_checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM).state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM,
        },
        policy_checkpoint,
    )

    with pytest.raises(ValueError, match="paired-seats"):
        run(
            build_parser().parse_args(
                [
                    "--checkpoint",
                    str(policy_checkpoint),
                    "--hands",
                    "1",
                    "--paired-seats",
                ]
            )
        )


def test_holdem_model_eval_run_parallel_smoke(tmp_path) -> None:
    policy_checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM).state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM,
        },
        policy_checkpoint,
    )

    metrics = run(
        build_parser().parse_args(
            [
                "--checkpoint",
                str(policy_checkpoint),
                "--hands",
                "2",
                "--opponent-policy",
                "random",
                "--jobs",
                "2",
            ]
        )
    )

    assert metrics["hands"] == 2
    assert metrics["jobs"] == 2
    assert metrics["shard_hands"] == [1, 1]
    assert metrics["rollout_margin"] == 1.0
    assert not metrics["paired_seats"]
