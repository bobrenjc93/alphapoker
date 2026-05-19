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
- Optional balanced, sqrt-balanced, and custom-exponent action-class weighting
  for Hold'em policy distillation.
- Hold'em policy-distillation checkpoint initialization, optional KL anchoring,
  rollout-margin control, and shard progress for long example-generation runs.
- REINFORCE-style Hold'em policy-gradient training against fixed opponents,
  with supervised checkpoint initialization, weighted opponent mixtures, and
  weighted seat-balanced training.
- Actor-critic Hold'em policy training with a learned value baseline and
  weighted seat-balanced training.
- Hold'em RL checkpoint selection by single-opponent or multi-opponent
  evaluation gates, with weighted-mean or minimum-score aggregation and
  per-opponent equity/rollout settings.
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

## Progress Timeline

Hold'em progress is tracked as fixed-limit chip EV rather than Elo. Elo needs a
binary match-win model; these experiments directly measure `avg_utility_model`
in chips/hand with paired seats and standard errors. The timestamp column below
uses real ISO-8601 git commit times from `git log --date=iso-strict` for the
commit that first recorded the metric.

| Recorded at | Commit | Milestone | Main measured gate |
| --- | --- | --- | --- |
| 2026-05-18T08:14:14-07:00 | `83a65be` | Bootstrapped exact Kuhn CFR harness. | Exploitability/testing harness only; no Hold'em gate yet. |
| 2026-05-18T08:59:30-07:00 | `07811cd` | Improved Leduc CFR distillation checkpoint selection. | Exact small-game benchmark infrastructure before Hold'em. |
| 2026-05-18T09:30:45-07:00 | `7211b2b` | Added first fixed-limit Hold'em equity policy baseline. | First larger-game rule-policy gate. |
| 2026-05-18T12:29:44-07:00 | `995c22f` | Added Hold'em policy-gradient trainer. | RL infrastructure; later direct RL checkpoints did not beat supervised gates. |
| 2026-05-19T03:13:49-07:00 | `e228875` | Tuned exact turn/river pot-odds rule policy. | `+0.0077 +/- 0.0157` vs `tight-turn-river-exact-pot-odds` e8, 4000 paired deals. |
| 2026-05-19T04:51:12-07:00 | `c760f0a` | Added range-aware pot-odds policy. | `+0.4455 +/- 0.0633` vs tight exact e8, 2000 paired deals. |
| 2026-05-19T05:02:45-07:00 | `806fda7` | Distilled the range-aware teacher into a neural policy. | `+0.3545 +/- 0.0787` vs tight exact e8, 2000 paired deals. |
| 2026-05-19T05:52:39-07:00 | `3344aef` | Added tight-range equity features and the current best 1k balanced distillation. | `+0.5073 +/- 0.0833` vs tight exact e8, 2000 paired deals. |
| 2026-05-19T07:56:19-07:00 | `fe97bbc` | Rechecked the current best with action diagnostics on a fresh larger seed. | `+0.4578 +/- 0.0579` vs tight exact e8, 4000 paired deals. |
| 2026-05-19T08:47:22-07:00 | `b325e0b` | Confirmed the current best against the range-aware opponent. | `+0.2860 +/- 0.0843` vs `tight-range-pot-odds` e4, 1000 paired deals. |
| 2026-05-19T10:31:16-07:00 | `2861fdb` | Found a KL-anchored safe-rollout side checkpoint. | `+0.4415 +/- 0.0986` vs tight exact e8 and `+0.2400 +/- 0.3793` vs safe rollout s4, but only `+0.0470 +/- 0.0813` vs tight range e4. |
| 2026-05-19T12:09:32-07:00 | `2d027bd` | Best-batch rollout actor-critic side pilot. | Good side probes, but exact gate was only `+0.0650 +/- 0.2337` over 200 paired deals. |
| 2026-05-19T12:39:31-07:00 | `38718c9` | 25% robustness-checkpoint logit blend. | Stayed positive on small exact/range probes, but failed safe rollout s1 at `-0.7375 +/- 0.5386`; not a candidate. |

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
- The same checkpoint also beat `balanced-turn-river-exact-pot-odds` e8 over
  1000 paired deals: `+0.6905 +/- 0.1156` chips/hand. Its live raise rate stayed
  near 10%, versus 3.9% for the opponent.
- Same checkpoint vs `tight-range-pot-odds` e4 with paired seats and 1000 paired
  deals: `+0.2860 +/- 0.0843` chips/hand. The model raised on 9.8% of its
  actions vs 3.3% for the range-aware opponent.
- The same checkpoint exposed a rollout robustness gap against
  `tight-safe-rollout-pot-odds` with `rollout_sims=4`, default margin 1.0, and
  200 paired deals: `-1.3000 +/- 0.4036` chips/hand. The safe-rollout opponent
  raised on 22.6% of its actions, while the model folded on 22.3% of its
  actions. The cheaper `rollout_sims=1` probe also stayed negative at
  `-0.9750 +/- 0.6183` over 40 paired deals.
- A small DAgger-style counterexample fine-tune on 200 player-0 and 200
  player-1 hands against that safe-rollout opponent repaired the rollout probe
  to `+0.2250 +/- 0.4165`, but it damaged the main tight exact gate to
  `-0.0445 +/- 0.1324` over 1000 paired deals, so it is not the current best.
  A KL-anchored variant with weight 2.0 also failed to repair the rollout probe
  (`-0.5400 +/- 0.5295`) and over-raised in live play.
- An unweighted KL-anchored counterexample fine-tune (KL weight 1.0) kept the
  tight exact gate positive (`+0.4415 +/- 0.0986` over 1000 paired deals) and
  repaired the safe-rollout probe (`+0.2400 +/- 0.3793` over 200 paired deals),
  but regressed against `tight-range-pot-odds` e4 to `+0.0470 +/- 0.0813` over
  1000 paired deals. It is a useful robustness side checkpoint, not a new
  current best.
- A 25% logit blend from the current best toward that unweighted KL robustness
  checkpoint stayed positive but noisy on small exact and range probes
  (`+0.3950 +/- 0.4353` vs tight exact e8 and `+0.1200 +/- 0.2015` vs
  `tight-range-pot-odds` e4, both over 100 paired deals), but failed the cheap
  `tight-safe-rollout-pot-odds` `rollout_sims=1` probe (`-0.7375 +/- 0.5386`
  over 40 paired deals). It is not a candidate.
- A lower-dose unweighted KL counterexample pass with 50 player-0 and 50
  player-1 safe-rollout behavior hands was still too disruptive: the completed
  tight exact e8 probe was `-0.2700 +/- 0.3548` over 100 paired deals, so the
  slow range/safe probes were not extended.
- A mixed replay pilot combining the original 1k tight-range self-play examples
  with 200 player-0 and 200 player-1 safe-rollout counterexample hands, trained
  with sqrt-balanced weighting and KL weight 1.0, did not repair the rollout
  gap: `-0.5425 +/- 0.3559` vs `tight-safe-rollout-pot-odds` over 200 paired
  deals.
- A rollout-aware actor-critic pilot initialized from the current best and
  trained for 50 hands directly against `tight-safe-rollout-pot-odds`
  (`rollout_sims=1`, tight-range feature sims 1) produced useful runtime
  telemetry but did not become a candidate: it was positive against the cheap
  safe-rollout probe (`+0.8000 +/- 0.7393` over 100 paired deals) and a small
  tight exact probe (`+0.5125 +/- 0.2914` over 200 paired deals), but collapsed
  the `tight-range-pot-odds` gate to `+0.0050 +/- 0.2920` over 200 paired
  deals.
- A 100-hand mixed-opponent actor-critic pilot with weights 0.45
  `tight-range-pot-odds`, 0.35 `tight-turn-river-exact-pot-odds`, and 0.20
  `tight-safe-rollout-pot-odds` (`rollout_sims=1`, tight-range feature sims 1)
  trained roughly flat (`+0.0200 +/- 0.7077`) and failed the tight exact probe:
  `-0.1750 +/- 0.2737` over 200 paired deals. It is not a candidate.
- Selecting the best-batch checkpoint from that same mixed-opponent trajectory
  improved the side probes but still did not clear the main gate: tight exact
  was only `+0.0650 +/- 0.2337` over 200 paired deals, while
  `tight-range-pot-odds` was `+0.3500 +/- 0.1925` over 200 paired deals and the
  cheap safe-rollout probe was `+0.8300 +/- 0.6249` over 100 paired deals.
- A smaller 80-hand mixed-opponent actor-critic pilot with checkpoint selection
  by a paired tight-exact e8 eval selected the initial checkpoint
  (`hands_played=0`). A follow-up exact probe was positive
  (`+0.8000 +/- 0.3319` over 100 paired deals), but the range gate was flat
  (`-0.0100 +/- 0.3449` over 100 paired deals), so exact-only selection is not
  enough to produce a candidate.
- Repeating that 80-hand mixed-opponent actor-critic pilot with multi-gate
  minimum selection over paired tight-exact and tight-range probes also selected
  the initial checkpoint. The best minimum score was `-0.1750` at
  `hands_played=0`; all later checkpoints were worse, with tight exact the
  limiting component each time.
- A tiny 40-hand pilot using per-gate minimum selection over tight exact e8,
  tight range e4, and safe rollout s1 selected the 20-hand checkpoint, but the
  best minimum score was still `-2.2500`; tight exact was again the limiting
  component in every selection eval.
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
- Custom class-weight exponent `0.75` calibrated the 1k model's imitation action
  mix well (predicted raises 150 vs target 137) but did not beat full balancing:
  `+0.4010 +/- 0.0722` vs tight exact e8 over 2000 paired deals. In live play it
  raised at about the same rate as the opponent, suggesting the full-balanced
  model's extra aggression is part of its edge against this gate.

## Research Roadmap

1. Keep Kuhn as the fast correctness harness.
2. Track Leduc exploitability over longer training runs.
3. Move from full-tree Leduc CFR to external-sampling MCCFR and neural regret/policy
   approximation.
4. Add distributed self-play and GPU jobs through `gpu-dev submit`.
5. Only treat "superhuman" as satisfied for a specific poker variant after
   exploitability and match-play evaluation against strong baselines support it.
