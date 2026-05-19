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
  --strategy-json experiments/leduc_cfr_linear_20k/strategy.json \
  --epochs 2000 \
  --out experiments/leduc_cfr_linear_20k_distill_2k_best
```

Evaluate the distilled model policy exactly:

```bash
uv run --extra train python -m alphapoker.evaluate_leduc_model \
  --checkpoint experiments/leduc_cfr_linear_20k_distill_2k_best/leduc_policy_value.pt \
  --strategy-json experiments/leduc_cfr_linear_20k/strategy.json \
  --out experiments/leduc_cfr_linear_20k_distill_2k_best_eval
```

Train and evaluate the current fixed-limit Hold'em policy baseline:

```bash
uv run --extra train --extra holdem python -m alphapoker.train_holdem_policy \
  --hands 500 \
  --equity-sims 8 \
  --expert-player 0 \
  --opponent-policy random \
  --epochs 200 \
  --seed 41 \
  --out experiments/holdem_equity_p0_vs_random_distill_500

uv run --extra train --extra holdem python -m alphapoker.evaluate_holdem_model \
  --checkpoint experiments/holdem_equity_p0_vs_random_distill_500/holdem_policy.pt \
  --hands 1000 \
  --seed 22 \
  --opponent-policy random \
  --progress \
  --out experiments/holdem_policy_p0_vs_random_1k/metrics.json
```

Collect DAgger-style labels from states visited by an existing policy:

```bash
uv run --extra train --extra holdem python -m alphapoker.train_holdem_policy \
  --hands 500 \
  --equity-sims 8 \
  --expert-player 0 \
  --opponent-policy equity \
  --behavior-checkpoint experiments/holdem_equity_p0_vs_random_distill_1k/holdem_policy.pt \
  --epochs 200 \
  --seed 44 \
  --out experiments/holdem_dagger_p0_vs_equity_500
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
- Heads-up fixed-limit Texas Hold'em state transitions and hand evaluation
  built on `treys`.
- Random fixed-limit Hold'em self-play baseline for exercising the larger-game
  state machine.
- Monte Carlo equity policy baseline for fixed-limit Hold'em.
- Pot-odds-aware equity policy baseline for fixed-limit Hold'em.
- Cross-seat pot-odds parameter sweeps for stronger rule-policy gates.
- Tuned pot-odds rule-policy gate confirmed against the default pot-odds policy.
- Imperfect-information pot-odds rollout policy for one-step action-value search.
- Opponent-range-conditioned pot-odds policy that filters hidden-card samples by
  observed opponent actions under an assumed baseline policy.
- JSON metric output for Hold'em policy-vs-policy self-play baselines.
- Supervised fixed-limit Hold'em policy distillation from the equity baseline.
- Supervised fixed-limit Hold'em policy distillation from pot-odds experts.
- Supervised fixed-limit Hold'em policy distillation from rollout-search experts.
- Cacheable Hold'em policy-imitation training examples for larger expert runs.
- Held-out Hold'em policy-imitation evaluation for cloned experts.
- Optional balanced and sqrt-balanced action-class weighting for Hold'em policy
  distillation.
- REINFORCE-style Hold'em policy-gradient training against fixed opponents,
  with supervised checkpoint initialization, weighted opponent mixtures, and
  weighted seat-balanced training.
- Actor-critic Hold'em policy training with a learned value baseline and
  weighted seat-balanced training.
- Backward-compatible Hold'em hand-summary, made-hand strength, legal-action,
  and pot-odds features for neural policies.
- Optional Monte Carlo, turn/river exact, and tight range-filtered equity features
  for Hold'em policy distillation checkpoints.
- Fixed-limit Hold'em neural checkpoint evaluation against random/equity
  baselines.
- Cross-seat Hold'em neural checkpoint evaluation.
- Hold'em match-evaluation action-count diagnostics by model/opponent role and
  player seat, with optional progress reporting for long checkpoint evaluations.
- Fixed-limit Hold'em equity regression model and threshold-policy evaluation.
- Both-seat training data for fixed-limit Hold'em equity regression.
- Cacheable Hold'em equity-value training examples for longer runs.
- Tunable equity thresholds for learned Hold'em value-policy evaluation.
- Seat-aware Hold'em model evaluation and threshold-sweep tooling.
- Cross-seat threshold sweeps for Hold'em equity-value policies.

Current exact-evaluation bests:

```bash
uv run python -m alphapoker.experiment_summary
```

Current fixed-limit Hold'em gate:

- `tight-range-pot-odds` vs `tight-turn-river-exact-pot-odds` with candidate
  `equity_sims=4`, opponent `equity_sims=8`, paired seats, 2000 paired deals:
  `+0.4455 +/- 0.0633` chips/hand for the range-aware policy.
- Same-scale exact e4 control vs opponent e8:
  `-0.1073 +/- 0.0636` chips/hand.
- `tight-range` feature 1k distillation from `tight-range-pot-odds`, evaluated
  against `tight-turn-river-exact-pot-odds` e8 with paired seats and 2000 paired
  deals: `+0.5073 +/- 0.0833` chips/hand for the model.
- A larger fresh-seed evaluation of the same checkpoint against
  `tight-turn-river-exact-pot-odds` e8 over 4000 paired deals remained positive:
  `+0.4578 +/- 0.0579` chips/hand. The live action mix was aggressive relative
  to the opponent: model raises were 9.7% of model actions vs 3.8% for the
  opponent.
- Same checkpoint vs `tight-range-pot-odds` e4 with paired seats and 500 paired
  deals: `+0.1630 +/- 0.1107` chips/hand.
- Unweighted `tight-range` feature 1k distillation improved imitation accuracy
  but collapsed raises; it was weaker in match play: `+0.3795 +/- 0.0618` vs
  tight exact e8 over 2000 paired deals, and `+0.0560 +/- 0.0782` vs
  `tight-range-pot-odds` e4 over 500 paired deals.
- Balanced `tight-range` feature 2k distillation was roughly tied with the 1k
  model on the tight exact gate (`+0.4960 +/- 0.0873`) but stronger against
  `tight-range-pot-odds` e4 (`+0.2980 +/- 0.1141`) over 500 paired deals.
- `sqrt-balanced` 1k distillation improved imitation metrics but under-raised
  and was weaker in match play: `+0.3905 +/- 0.0625` vs tight exact e8 over
  2000 paired deals, and `-0.0620 +/- 0.0811` vs `tight-range-pot-odds` e4 over
  500 paired deals.

## Research Roadmap

1. Keep Kuhn as the fast correctness harness.
2. Track Leduc exploitability over longer training runs.
3. Move from full-tree Leduc CFR to external-sampling MCCFR and neural regret/policy
   approximation.
4. Add distributed self-play and GPU jobs through `gpu-dev submit`.
5. Only treat "superhuman" as satisfied for a specific poker variant after
   exploitability and match-play evaluation against strong baselines support it.
