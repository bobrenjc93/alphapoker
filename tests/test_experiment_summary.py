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


def test_experiment_summary_loads_model_eval(tmp_path) -> None:
    exp_dir = tmp_path / "experiments"
    (exp_dir / "model_run").mkdir(parents=True)
    (exp_dir / "model_run" / "model_eval.json").write_text(
        json.dumps(
            {
                "source_metrics": {"iterations": 20},
                "game_value_p0": -0.2,
                "exploitability": 0.3,
                "infosets": 288,
            }
        )
    )

    table = markdown_table(load_metrics(exp_dir))

    assert "| model_run | 20 | -0.200000 | 0.300000 | 288 |" in table
