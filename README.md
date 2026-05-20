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
- Optional soft action-probability targets from rollout action values for
  Hold'em policy distillation.
- Cacheable Hold'em policy-imitation training examples for larger expert runs.
- Held-out Hold'em policy-imitation evaluation for cloned experts.
- Optional balanced, sqrt-balanced, and custom-exponent action-class weighting
  for Hold'em policy distillation.
- Optional per-action loss-weight overrides for Hold'em policy distillation,
  useful when frequency balancing still confuses specific actions.
- Optional player-action example weighting plus per-player target/prediction
  diagnostics for Hold'em policy distillation.
- Optional facing-bet example weighting for Hold'em policy distillation, used
  to emphasize call/fold response states without changing cached example files.
- Optional Hold'em action-history policy features and first-layer input
  expansion for initializing wider feature checkpoints from older policies.
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
- State-dependent Hold'em checkpoint blending that can switch toward a
  robustness checkpoint after observed opponent aggression.
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

![Hold'em exact-gate EV over time](docs/holdem_exact_gate_progress.svg)

The graph plots the comparable tight exact gate over real commit time. Small
one-off exact spikes and failed robustness probes stay in the table rather than
the line so the visual tracks durable gate progress. The table below keeps the
broader context for range-aware and safe-rollout probes.

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
| 2026-05-19T13:16:01-07:00 | `cc55a67` | Quantified the cheap safe-rollout baseline. | Current best stayed negative vs `tight-safe-rollout-pot-odds` s1 at `-0.9750 +/- 0.6183` over 40 paired deals. |
| 2026-05-19T13:23:23-07:00 | `fa8948c` | Tried exact-selected rollout actor-critic. | Exact-only selection chose the initial checkpoint; follow-up range gate was flat at `-0.0100 +/- 0.3449` over 100 paired deals. |
| 2026-05-19T13:35:55-07:00 | `1d32683` | Tried multi-gate actor-critic selection. | Minimum selection over exact/range also chose the initial checkpoint; best minimum score was `-0.1750`, limited by exact. |
| 2026-05-19T13:45:42-07:00 | `5d88477` | Added per-gate actor-critic selection including safe rollout. | The 20-hand checkpoint was selected, but best minimum score was still `-2.2500`, again limited by exact. |
| 2026-05-19T14:05:21-07:00 | `bdec05e` | Tried unweighted KL2 safe-rollout DAgger. | Exact was noisy-positive at `+0.2950 +/- 0.4455`, but range failed at `-0.3750 +/- 0.2848` over 100 paired deals. |
| 2026-05-19T14:12:16-07:00 | `7f3d952` | Tried a 50% robustness-checkpoint logit blend. | Range stayed positive at `+0.2750 +/- 0.2105`, but safe rollout s1 stayed negative at `-0.4000 +/- 0.6534`. |
| 2026-05-19T14:47:54-07:00 | `26e80b3` | Tried a range-refresh pass from the safe-rollout side checkpoint. | Exact spiked on a small probe at `+0.7950 +/- 0.3197`, but range was only `+0.0950 +/- 0.3300` and safe rollout s1 failed at `-1.1625 +/- 0.6153`. |
| 2026-05-19T15:03:29-07:00 | `7023b00` | Tried aggression-triggered adaptive checkpoint blends. | Full response weight was strong on small exact/range probes (`+0.6800 +/- 0.2878`, `+0.5800 +/- 0.2753`) but failed safe rollout s1 at `-1.6875 +/- 0.6958`. |
| 2026-05-19T15:10:36-07:00 | `7eedeea` | Directly checked the KL1 robustness checkpoint against safe rollout s1. | The checkpoint that was positive vs safe s4 still failed the cheaper s1 probe at `-0.9375 +/- 0.5413` over 40 paired deals. |
| 2026-05-19T15:33:35-07:00 | `daab72d` | Tried an s1-specific safe-rollout DAgger pass. | Small exact spiked to `+0.9800 +/- 0.5184`, but range was only `+0.1750 +/- 0.4076` and safe rollout s1 stayed negative at `-0.6500 +/- 1.1755`. |
| 2026-05-19T15:57:39-07:00 | `fe4274e` | Tried sqrt-balanced s1 safe-rollout DAgger. | Rare raises were preserved in training, but range failed at `-0.0200 +/- 0.4721` and safe rollout s1 was `-0.8375 +/- 1.2561`; exact was `+0.5800 +/- 0.7071`. |
| 2026-05-19T16:41:33-07:00 | `abf88ee` | Labeled current-best self-play with the safe-rollout expert. | Safe rollout s1 improved to `+0.3650 +/- 0.8909` over 100 paired deals and range stayed positive at `+0.1460 +/- 0.3804`, but exact was flat at `+0.0460 +/- 0.4930`; side checkpoint only. |
| 2026-05-19T16:59:21-07:00 | `7dcbf30` | Tried static blends toward the safe-expert side checkpoint. | A 25% blend looked strong on small exact/range probes (`+1.0200 +/- 0.8197`, `+0.4200 +/- 0.5221`) but failed safe rollout s1 at `-2.5875 +/- 1.4214`; not a candidate. |
| 2026-05-19T17:21:09-07:00 | `6dd614a` | Tried an aggression-triggered switch to the safe-expert side checkpoint. | Exact/range stayed positive on larger probes (`+0.1780 +/- 0.4707`, `+0.2240 +/- 0.3381`), but safe rollout s1 failed at `-1.5550 +/- 0.8788`; not a candidate. |
| 2026-05-19T17:43:38-07:00 | `c890a0c` | Scaled safe-expert self-play DAgger to 300 hands with a stronger KL anchor. | Tight exact stayed positive on a small probe (`+0.4400 +/- 0.7888`), but range flattened to `+0.0150 +/- 0.4784` and safe rollout s1 stayed negative at `-0.3625 +/- 1.4758`. |
| 2026-05-19T18:41:56-07:00 | `2a8bc90` | Mixed original range-teacher replay with safe-expert self-play labels. | A 200-hand base replay plus 100 safe-expert hands spiked tight exact to `+0.7000 +/- 0.5822`, but range failed at `-0.1100 +/- 0.4534` and safe rollout s1 failed at `-1.0875 +/- 1.3269`. |
| 2026-05-19T19:04:07-07:00 | `81d7135` | Swept KL/class-weight replay variants from the mixed safe-expert replay set. | KL8 sqrt-balanced recovered small exact/range probes (`+0.9400 +/- 0.7225`, `+0.7050 +/- 0.3965`), but safe rollout s1 failed at `-1.5875 +/- 1.1674`; KL16 balanced was also exact/range positive but safe negative. |
| 2026-05-19T19:28:35-07:00 | `a079098` | Upweighted facing-bet response states in the mixed replay set. | KL8 sqrt-facing3 preserved small exact/range probes (`+0.8750 +/- 0.5249`, `+0.5300 +/- 0.4541`) but still failed safe rollout s1 at `-1.0500 +/- 1.0175`; not a candidate. |
| 2026-05-19T19:46:54-07:00 | `7db8e2b` | Tried explicit action-history features for safe-expert self-play labels. | First-layer-expanded KL8 sqrt-facing3 action-history pilot still failed safe rollout s1 at `-1.2750 +/- 1.0562` over 40 paired deals. |
| 2026-05-19T20:23:11-07:00 | `5574b59` | Mixed action-history range replay with safe-expert labels. | A 774-example base replay plus 472 safe labels still failed safe rollout s1 at `-1.5750 +/- 1.1819`; no exact/range extension. |
| 2026-05-19T20:39:47-07:00 | `c6c7dee` | Targeted action-history safe labels at player 1. | P1-focused replay kept small exact/range probes positive (`+0.6000 +/- 0.7372`, `+0.7800 +/- 0.7019`) and improved the cheap safe point to `-0.6250 +/- 1.1647`, but safe remained negative and P1 was still weak. |
| 2026-05-19T20:58:41-07:00 | `70e91b1` | Increased player-1 safe replay to 300 hands. | The larger P1 dose kept small exact/range probes positive (`+0.9700 +/- 0.7543`, `+0.2600 +/- 0.3663`) but cheap safe stayed negative at `-0.7125 +/- 1.3259`; P1 improved to `-0.9250` while P0 regressed. |
| 2026-05-19T21:21:42-07:00 | `13822dd` | Rebalanced 300-hand safe replay across both seats. | The 2,305-example mix kept small exact/range probes positive (`+0.4350 +/- 0.8808`, `+0.4250 +/- 0.5490`) but cheap safe stayed negative at `-0.8125 +/- 1.0932`; both safe seats were negative. |
| 2026-05-19T21:33:48-07:00 | `584196a` | Switched balanced p0+p1 safe replay to full class balancing. | Small exact/range stayed positive (`+0.4950 +/- 0.6832`, `+0.3650 +/- 0.7355`) and safe s1 smoked positive at `+0.5375 +/- 1.2551`, but a 100-paired safe confirmation was flat-negative at `-0.0850 +/- 0.9937`; side checkpoint only. |
| 2026-05-19T21:38:59-07:00 | `013dd7b` | Lowered the KL anchor on the balanced-class safe replay. | KL4 did not improve the supervised raise/fold mix and failed cheap safe rollout at `-1.2125 +/- 1.4531`; no exact/range extension. |
| 2026-05-19T21:47:45-07:00 | `2e72c2c` | Tested action-history-compatible adaptive blends. | Expanded the current best to action-history inputs and blended toward the balanced-class side checkpoint after opponent aggression; 50% failed at `-2.5500 +/- 1.7593`, while 25% still failed at `-0.5875 +/- 1.3301`. |
| 2026-05-19T21:52:00-07:00 | `b7de1aa` | Selected full-balanced safe replay by train loss. | Removing the validation split ran to epoch 200, but still collapsed raise/fold (`raise` target 440 vs predicted 297; `fold` target 390 vs predicted 533) and failed cheap safe rollout at `-0.8250 +/- 1.1044`; no exact/range extension. |
| 2026-05-19T22:02:28-07:00 | `a7f6904` | Added explicit raise/fold loss shaping. | `raise=2.0`, `fold=0.5` moved the cheap safe probe positive (`+0.8125 +/- 1.8117`) but damaged tight exact (`-0.4150 +/- 0.7566`) and flattened range (`-0.0650 +/- 0.4600`); diagnostic only. |
| 2026-05-19T22:15:43-07:00 | `562c955` | Swept milder and call-aware action shaping. | `raise=1.5`, `fold=0.75` failed safe rollout (`-1.7375 +/- 1.4825`); p1-focused heavy shaping also failed (`-0.8375 +/- 1.1339`); adding `call=1.5` kept exact/range near flat-positive but safe was too noisy (`+0.3000 +/- 2.0511`). |
| 2026-05-19T22:29:07-07:00 | `4191fc5` | Tried player-action weighting for player-1 responses. | KL8 p1 call/raise upweighting still failed cheap safe rollout (`-1.0250 +/- 1.4232`); KL2 moved p1 raises but stayed negative (`-0.4000 +/- 1.5183`), and a more call-heavy KL2 variant failed at `-1.4500 +/- 1.7017`. |
| 2026-05-19T22:56:26-07:00 | `93e7682` | Tried soft rollout action-probability targets. | A 20-hand soft safe-rollout replay raised target mass, but player 1 still under-raised and cheap safe rollout failed at `-1.8625 +/- 0.8881` over 40 paired deals. |
| 2026-05-19T23:08:24-07:00 | `145dea5` | Targeted soft safe-rollout labels at player 1. | KL2 train selection improved p1 raise imitation and made the cheap safe probe noisy-positive (`+0.5750 +/- 1.0963`), but exact and range gates failed (`-0.4000 +/- 0.3654`, `-0.3050 +/- 0.1972`). |
| 2026-05-19T23:19:47-07:00 | `99690c8` | Made the p1 soft branch blend-compatible with the current best. | A 50% aggression-triggered blend stayed near flat on exact/range/safe (`-0.0500 +/- 0.3281`, `+0.0750 +/- 0.3311`, `+0.0375 +/- 0.6366`); no current-best update. |

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
  current best. A later cheap `rollout_sims=1` check showed that this repair did
  not transfer to the faster safe-rollout setting (`-0.9375 +/- 0.5413` over 40
  paired deals).
- A direct unweighted KL1 counterexample pass against the cheap
  `tight-safe-rollout-pot-odds` setting (`rollout_sims=1`, 100 player-0 and 100
  player-1 behavior hands) improved the cheap safe-rollout probe relative to the
  current best but did not repair it: `-0.6500 +/- 1.1755` over 40 paired deals.
  Its small tight exact probe spiked to `+0.9800 +/- 0.5184`, while the
  `tight-range-pot-odds` gate was only `+0.1750 +/- 0.4076` over 100 paired
  deals, so it is a diagnostic side checkpoint rather than the current best.
- A `sqrt-balanced` variant of that cheap safe-rollout DAgger pass preserved the
  rare raise class better in training, but did not improve live robustness:
  `+0.5800 +/- 0.7071` vs tight exact e8, `-0.0200 +/- 0.4721` vs
  `tight-range-pot-odds`, and `-0.8375 +/- 1.2561` vs cheap safe rollout. Class
  weighting alone is not enough for the rollout opponent.
- Directly labeling current-best self-play states with the
  `tight-safe-rollout-pot-odds` expert and a stronger KL anchor produced the
  first positive larger cheap safe-rollout side probe: `+0.3650 +/- 0.8909` over
  100 paired deals. It also stayed positive against `tight-range-pot-odds`
  (`+0.1460 +/- 0.3804` over 250 paired deals), but the tight exact gate was
  effectively flat (`+0.0460 +/- 0.4930` over 250 paired deals), so it is a
  robustness side checkpoint rather than the current best.
- A static 25% logit blend from the current best toward that safe-expert side
  checkpoint looked promising on small tight exact and range probes (`+1.0200
  +/- 0.8197` and `+0.4200 +/- 0.5221`, both over 100 paired deals), but failed
  the cheap `tight-safe-rollout-pot-odds` `rollout_sims=1` gate badly:
  `-2.5875 +/- 1.4214` over 40 paired deals. A 50% blend was weaker on exact
  (`-0.1600 +/- 0.9439`) and only mildly positive on range (`+0.1100 +/-
  0.4157`), so static interpolation is not enough for the safe-expert branch.
- Switching fully to that safe-expert side checkpoint only after the opponent's
  first bet or raise was less damaging than a static blend on the cheap
  safe-rollout smoke test (`+0.2000 +/- 1.5660` over 40 paired deals), but the
  larger confirmation failed at `-1.5550 +/- 0.8788` over 100 paired deals.
  Tight exact and range gates stayed positive but below the current best
  (`+0.1780 +/- 0.4707` and `+0.2240 +/- 0.3381`, both over 250 paired deals).
  The safe-expert branch still needs training-time integration rather than
  runtime interpolation.
- Scaling the safe-expert self-play DAgger pass to 300 hands with a stronger
  KL anchor (`init_kl_weight=8.0`) kept the small tight exact smoke test
  positive (`+0.4400 +/- 0.7888` over 100 paired deals), but flattened the
  `tight-range-pot-odds` gate (`+0.0150 +/- 0.4784` over 100 paired deals) and
  still missed the cheap safe-rollout gate (`-0.3625 +/- 1.4758` over 40 paired
  deals). More safe-expert labels alone are not enough without replaying the
  original range-strength behavior.
- A small mixed replay fine-tune that combined 200 hands of original
  range-teacher replay with 100 hands of safe-expert self-trajectory labels
  produced a strong small tight exact probe (`+0.7000 +/- 0.5822` over 100
  paired deals), but lost the range gate (`-0.1100 +/- 0.4534`) and still failed
  cheap safe rollout (`-1.0875 +/- 1.3269`). The base replay idea remains
  plausible, but this ratio and scale underfit the range-aware opponent.
- Reusing that mixed replay dataset with stronger anchors recovered the small
  exact and range probes. The KL8 sqrt-balanced variant reached `+0.9400 +/-
  0.7225` vs tight exact e8 and `+0.7050 +/- 0.3965` vs
  `tight-range-pot-odds`, while the KL16 balanced variant reached `+0.6800 +/-
  0.5737` and `+0.4600 +/- 0.3473`. Both still failed cheap safe rollout s1
  (`-1.5875 +/- 1.1674` and `-1.0125 +/- 1.4744`), so this is not a current
  best update.
- Upweighting facing-bet examples by 3x in that same replay set preserved the
  KL8 sqrt-balanced exact/range behavior (`+0.8750 +/- 0.5249` vs tight exact
  e8 and `+0.5300 +/- 0.4541` vs `tight-range-pot-odds`), but the cheap
  safe-rollout probe remained negative at `-1.0500 +/- 1.0175`. The KL16
  balanced-facing3 variant was also negative on cheap safe rollout (`-1.3500
  +/- 1.2750`). Response-state weighting alone is not enough.
- Adding explicit action-history features and expanding the initialized first
  layer from the current best (`input_dim 141 -> 146`) did not repair the safe
  gate in a 100-hand safe-expert self-play pilot. The cheap safe-rollout probe
  was `-1.2750 +/- 1.0562`, so the issue is not solved by exposing prior
  aggression counts alone.
- Regenerating both the range-teacher replay and the safe-expert labels with
  action-history features preserved a balanced supervised action mix, but did
  not improve live robustness. The 774-example base replay plus 472 safe labels
  failed cheap safe rollout at `-1.5750 +/- 1.1819`, so history-aware replay is
  not enough at this scale and mix.
- Targeting safe-expert labels only at player 1 improved the cheap safe-rollout
  point estimate and preserved small exact/range probes (`+0.6000 +/- 0.7372`
  and `+0.7800 +/- 0.7019`), but did not clear safe rollout (`-0.6250 +/-
  1.1647` over 40 paired deals). The seat split still showed player 1 at
  `-2.7000`, so this remains a diagnostic branch rather than a candidate.
- Scaling the player-1-only safe labels to 300 hands improved the player-1 safe
  seat to `-0.9250`, but player 0 regressed to `-0.5000` and overall cheap safe
  rollout stayed negative (`-0.7125 +/- 1.3259`). Small exact/range checks were
  still positive (`+0.9700 +/- 0.7543` and `+0.2600 +/- 0.3663`), which points
  to a seat-balance problem rather than a simple need for more player-1 labels.
- Adding a matching 300-hand player-0 safe cache produced a larger balanced
  2,305-example replay set, but still failed the cheap safe rollout probe:
  `-0.8125 +/- 1.0932`, with player 0 at `-0.9000` and player 1 at `-0.7250`.
  The exact/range smoke checks remained positive but below the current best,
  so seat-balanced safe replay alone is also not enough.
- Switching that same replay set from sqrt-balanced to full balanced class
  weighting moved the 40-paired cheap safe probe positive (`+0.5375 +/-
  1.2551`) and kept small exact/range probes positive, but the 100-paired safe
  confirmation was flat-negative (`-0.0850 +/- 0.9937`) with player 1 still
  negative. This is the most useful recent direction, but not a confirmed
  repair.
- Lowering the KL anchor from 8 to 4 on the same full-balanced replay set did
  not improve the action mix: predicted raises stayed below target and folds
  stayed high. The cheap safe rollout probe regressed to `-1.2125 +/- 1.4531`,
  so the replay issue is not simply an over-strong KL anchor.
- An action-history-expanded copy of the current best enabled compatible blends
  toward the balanced-class side checkpoint, but adaptive blending after the
  opponent's first aggressive action was too disruptive. A 50% blend failed at
  `-2.5500 +/- 1.7593`, and a 25% blend still failed at `-0.5875 +/- 1.3301`.
- Selecting the full-balanced safe replay checkpoint by train loss instead of
  validation loss did not solve the raise/fold confusion. The model trained
  through epoch 200, but still predicted only 297 raises for 440 raise labels
  and 533 folds for 390 fold labels. Cheap safe rollout failed at `-0.8250 +/-
  1.1044`, with player 1 still the weak seat (`-1.7500`), so the next useful
  change is likely loss shaping or richer expert targets rather than checkpoint
  selection.
- Adding explicit action loss weights confirmed that raise/fold shaping can
  move the safe rollout gate, but the first dose was too aggressive.
  `raise=2.0` and `fold=0.5` increased predicted raises to 355 and made cheap
  safe rollout positive (`+0.8125 +/- 1.8117`), yet the small exact and range
  probes regressed to `-0.4150 +/- 0.7566` and `-0.0650 +/- 0.4600`. This points
  toward milder weights or seat-specific targets rather than a global heavy
  raise bias.
- Follow-up global action-weight sweeps did not produce a candidate. A milder
  `raise=1.5`, `fold=0.75` setting barely moved the supervised mix and failed
  cheap safe rollout (`-1.7375 +/- 1.4825`). Applying the heavy weights to the
  player-1-focused replay also failed (`-0.8375 +/- 1.1339`). Adding
  `call=1.5` to the heavy p0+p1 replay preserved small exact/range probes
  better (`+0.0350 +/- 0.7608`, `+0.2250 +/- 0.4967`) and was noisy-positive
  on cheap safe rollout (`+0.3000 +/- 2.0511`), but player 1 still rarely
  raised in live play. Global action weights are useful diagnostics, not a
  robust fix.
- Player-action weighting exposed the seat-specific failure more clearly but
  still did not repair it. KL8 p1 call/raise upweighting left p1 raises too low
  and failed cheap safe rollout (`-1.0250 +/- 1.4232`). Dropping to KL2 moved p1
  raises above target, but p1 calls collapsed and safe rollout stayed negative
  (`-0.4000 +/- 1.5183`). A more call-heavy KL2 setting balanced supervised p1
  calls/raises better, then failed live safe rollout at `-1.4500 +/- 1.7017`.
  The next useful direction is likely richer expert targets or value/margin
  labels, not more scalar hard-label weighting.
- Soft action-probability targets from safe-rollout action values improved the
  label signal but did not fix live play at small scale. A 20-hand soft replay
  produced target action mass with `25.22` raises and only `9.73` folds across
  87 examples, but the trained model still predicted only one player-1 raise on
  the held-out split. The cheap safe-rollout gate failed at `-1.8625 +/-
  0.8881` over 40 paired deals, with player 0 at `+0.2250` and player 1 at
  `-3.9500`, so the rollout-value signal needs a better seat-specific training
  setup before scaling.
- Targeting those soft safe-rollout labels at player 1 and selecting by train
  loss with a weaker KL2 anchor finally moved supervised player-1 raises in the
  desired direction (`14` predicted vs `23` target, compared with `3` predicted
  under KL6 validation selection). The live safe rollout smoke test became
  noisy-positive (`+0.5750 +/- 1.0963` over 40 paired deals), but the branch
  failed the protective gates: `-0.4000 +/- 0.3654` vs tight exact e8 and
  `-0.3050 +/- 0.1972` vs `tight-range-pot-odds` e4, both over 100 paired
  deals. It is a useful robustness diagnostic, not a candidate.
- Regenerating that p1-soft branch with feature metadata compatible with the
  action-history-expanded current best allowed aggression-triggered logit
  blends. A 25% blend still failed cheap safe rollout (`-0.4500 +/- 0.9914`).
  A 50% blend was only flat across the protective probes: `+0.0375 +/- 0.6366`
  vs cheap safe rollout over 40 paired deals, `-0.0500 +/- 0.3281` vs tight
  exact e8, and `+0.0750 +/- 0.3311` vs `tight-range-pot-odds` e4, both over
  100 paired deals. Blending soft p1 labels is not enough to recover the
  current-best exact/range edge.
- A 25% logit blend from the current best toward that unweighted KL robustness
  checkpoint stayed positive but noisy on small exact and range probes
  (`+0.3950 +/- 0.4353` vs tight exact e8 and `+0.1200 +/- 0.2015` vs
  `tight-range-pot-odds` e4, both over 100 paired deals), but failed the cheap
  `tight-safe-rollout-pot-odds` `rollout_sims=1` probe (`-0.7375 +/- 0.5386`
  over 40 paired deals). It is not a candidate.
- A 50% logit blend toward the same robustness checkpoint improved the small
  range probe (`+0.2750 +/- 0.2105` vs `tight-range-pot-odds` e4 over 100 paired
  deals) but still failed the cheap safe-rollout probe (`-0.4000 +/- 0.6534`
  over 40 paired deals). Stronger 75% blend probes timed out before writing
  complete metrics, so they are not recorded.
- A 50-hand-per-seat balanced range-refresh fine-tune from the unweighted KL1
  robustness checkpoint produced a strong small exact probe (`+0.7950 +/-
  0.3197` over 100 paired deals), but the range gate was only weakly positive
  (`+0.0950 +/- 0.3300` over 100 paired deals) and the cheap safe-rollout probe
  failed (`-1.1625 +/- 0.6153` over 40 paired deals). It is not a candidate.
- An aggression-triggered adaptive blend that switches toward the KL1 robustness
  checkpoint after an observed opponent bet/raise improved small exact and range
  probes at full response weight (`+0.6800 +/- 0.2878` vs tight exact e8 and
  `+0.5800 +/- 0.2753` vs `tight-range-pot-odds` e4, both over 100 paired
  deals), but made the cheap safe-rollout probe worse (`-1.6875 +/- 0.6958`
  over 40 paired deals). A 50% response weight was also not a candidate:
  `+0.0700 +/- 0.3146` on range and `-0.4250 +/- 0.9156` on safe rollout.
- A lower-dose unweighted KL counterexample pass with 50 player-0 and 50
  player-1 safe-rollout behavior hands was still too disruptive: the completed
  tight exact e8 probe was `-0.2700 +/- 0.3548` over 100 paired deals, so the
  slow range/safe probes were not extended.
- A more conservative unweighted KL2 counterexample pass with 100 player-0 and
  100 player-1 safe-rollout behavior hands kept the small tight exact probe
  noisy-positive (`+0.2950 +/- 0.4455` over 100 paired deals), but failed the
  `tight-range-pot-odds` gate (`-0.3750 +/- 0.2848` over 100 paired deals), so
  the safe-rollout extension was skipped.
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
