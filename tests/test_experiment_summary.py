import json

from alphapoker.experiment_summary import load_metrics, markdown_table


def test_experiment_summary_loads_metrics(tmp_path) -> None:
    exp_dir = tmp_path / "experiments"
    (exp_dir / "run_a").mkdir(parents=True)
    (exp_dir / "run_a" / "metrics.json").write_text(
        json.dumps(
            {
                "iterations": 10,
                "game_value_p0": -0.1,
                "exploitability": 0.2,
                "infosets": 12,
            }
        )
    )

    metrics = load_metrics(exp_dir)
    table = markdown_table(metrics)

    assert len(metrics) == 1
    assert metrics[0].name == "run_a"
    assert "| run_a | 10 | -0.100000 | 0.200000 | 12 |" in table
