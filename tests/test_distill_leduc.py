import json

import pytest

from alphapoker.distill_leduc import build_parser, run


def test_distill_leduc_rejects_non_leduc_strategy(tmp_path) -> None:
    strategy_json = tmp_path / "strategy.json"
    strategy_json.write_text(json.dumps({"game": "kuhn_poker", "strategy": {}}))

    args = build_parser().parse_args(
        [
            "--strategy-json",
            str(strategy_json),
            "--out",
            str(tmp_path / "out"),
            "--epochs",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="Leduc"):
        run(args)
