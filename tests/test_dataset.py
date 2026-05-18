from alphapoker.cfr import KuhnCFRTrainer
from alphapoker.dataset import strategy_examples


def test_strategy_examples_have_fixed_shapes() -> None:
    trainer = KuhnCFRTrainer()
    trainer.train(10)
    examples = strategy_examples(trainer.average_strategy())

    assert len(examples) == 12
    assert {len(example.features) for example in examples} == {9}
    assert {len(example.policy) for example in examples} == {4}
    assert {len(example.legal_mask) for example in examples} == {4}

