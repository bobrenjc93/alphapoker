import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.evaluate_holdem_model import (  # noqa: E402
    build_parser,
    evaluation_process_context,
    make_opponent_policy,
    model_policy_from_checkpoint,
    parse_model_players,
    run,
    split_hands,
)
from alphapoker.holdem import deal_fixed_limit_holdem  # noqa: E402
from alphapoker.holdem_features import HOLDEM_CANONICAL_ACTIONS, HOLDEM_FEATURE_DIM  # noqa: E402
from alphapoker.holdem_model import HoldemEquityNet, HoldemPolicyNet  # noqa: E402


def write_biased_policy_checkpoint(path, action: str, *, input_dim: int = HOLDEM_FEATURE_DIM) -> None:
    model = HoldemPolicyNet(input_dim=input_dim)
    for parameter in model.parameters():
        parameter.data.zero_()
    bias = model.net[-1].bias
    assert bias is not None
    bias.data[HOLDEM_CANONICAL_ACTIONS.index(action)] = 10.0
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
        },
        path,
    )


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
            "--blend-weight",
            "0.25",
            "--model-player",
            "1",
        ]
    )

    assert args.model_player == (1,)
    assert str(args.blend_checkpoint) == "blend.pt"
    assert args.blend_weight == 0.25


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
                "--blend-checkpoint",
                str(blend_checkpoint),
                "--blend-weight",
                "0.25",
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
    assert metrics["blend_checkpoint"] == str(blend_checkpoint)
    assert metrics["blend_weight"] == 0.25
    assert not metrics["paired_seats"]
    assert sum(metrics["model_action_counts"].values()) > 0
    assert sum(metrics["opponent_action_counts"].values()) > 0


def test_holdem_model_eval_run_paired_seats_smoke(tmp_path) -> None:
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
    assert len(metrics["seat_metrics"]) == 2
    assert sum(metrics["model_action_counts"].values()) == sum(
        sum(seat["model_action_counts"].values()) for seat in metrics["seat_metrics"]
    )


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
