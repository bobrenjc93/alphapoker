# AlphaPoker

AlphaPoker is an AlphaGo-inspired poker research repo. The first milestone is
deliberately small and verifiable: solve Kuhn poker with self-play CFR, measure
exploitability, and optionally distill the tabular policy into a neural
policy/value model. Kuhn is not full poker, but it exercises the core issue that
Go does not have: hidden information.

This follows the useful shape of `ericjang/autogo`: rules, search/self-play,
model code, experiment scripts, and tests live in the repo so agents can iterate
on experiments without changing the foundations each time.

## Quickstart

```bash
uv run --extra dev pytest
uv run python -m alphapoker.train --iterations 50000 --out experiments/kuhn_cfr_baseline
```

The training command writes:

- `strategy.json`: average CFR strategy and exploitability report.
- `metrics.json`: compact scalar metrics for experiment tracking.

To also train the small policy/value network from the CFR strategy:

```bash
uv run --extra train python -m alphapoker.train \
  --iterations 50000 \
  --network-epochs 500 \
  --out experiments/kuhn_cfr_baseline
```

Leduc policy distillation uses the same pattern:

```bash
uv run --extra train python -m alphapoker.train_leduc \
  --iterations 5000 \
  --best-response \
  --network-epochs 500 \
  --out experiments/leduc_cfr_5k
```

To distill an already-trained strategy without rerunning CFR:

```bash
uv run --extra train python -m alphapoker.distill_leduc \
  --strategy-json experiments/leduc_cfr_20k/strategy.json \
  --epochs 500 \
  --out experiments/leduc_cfr_20k_distill
```

Evaluate the distilled model policy exactly:

```bash
uv run --extra train python -m alphapoker.evaluate_leduc_model \
  --checkpoint experiments/leduc_cfr_20k_distill/leduc_policy_value.pt \
  --strategy-json experiments/leduc_cfr_20k/strategy.json \
  --out experiments/leduc_cfr_20k_distill_eval
```

## Current Milestone

- Exact Kuhn poker environment with legal actions and zero-sum payoffs.
- Tabular CFR/CFR+ self-play trainer.
- Exact best-response exploitability by enumerating deterministic policies.
- Optional PyTorch policy/value model for distillation from the CFR average
  strategy.
- Limit Leduc poker rules for the next imperfect-information benchmark.
- Tabular Leduc CFR trainer with exact expected-value and best-response
  exploitability evaluation.
- Optional Leduc policy/value distillation model.

Current exact-evaluation bests:

```bash
uv run python -m alphapoker.experiment_summary
```

## Research Roadmap

1. Keep Kuhn as the fast correctness harness.
2. Track Leduc exploitability over longer training runs.
3. Move from full-tree Leduc CFR to external-sampling MCCFR and neural regret/policy
   approximation.
4. Add distributed self-play and GPU jobs through `gpu-dev submit`.
5. Only treat "superhuman" as satisfied for a specific poker variant after
   exploitability and match-play evaluation against strong baselines support it.
