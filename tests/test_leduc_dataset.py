from alphapoker.leduc_cfr import LeducCFRTrainer
from alphapoker.leduc_dataset import leduc_strategy_examples


def test_leduc_strategy_examples_have_fixed_shapes() -> None:
    trainer = LeducCFRTrainer()
    trainer.train(1)
    examples = leduc_strategy_examples(trainer.average_strategy())

    assert len(examples) == 288
    assert {len(example.features) for example in examples} == {19}
    assert {len(example.policy) for example in examples} == {5}
    assert {len(example.legal_mask) for example in examples} == {5}

