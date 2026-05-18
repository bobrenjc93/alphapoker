import json

import pytest

from alphapoker.evaluate_leduc_model import load_leduc_strategy


def test_load_leduc_strategy_rejects_other_games(tmp_path) -> None:
    strategy_json = tmp_path / "strategy.json"
    strategy_json.write_text(json.dumps({"game": "kuhn_poker", "strategy": {}}))

    with pytest.raises(ValueError, match="Leduc"):
        load_leduc_strategy(strategy_json)
