# EA-Life v3 research protocol: predictive transition learning and collective ecology

## Research question

Do bounded learned transition predictions and shared environmental traces improve performance beyond the current local heuristic plus tabular Q-learning? The two additions are designed to be independently ablated so that a larger codebase or more animated dashboard is not confused with a measured improvement.

## Hypotheses

- **H1 — predictive transition learning:** a state/action model that estimates likely next abstract states and immediate reward will reduce its own held-out-in-time reward prediction error as experience accumulates, and may improve task outcomes over the no-world-model baseline.
- **H2 — collective environmental memory:** shared decaying forage and danger traces will change spatial behavior and may improve resource acquisition or hazard avoidance relative to no-trace conditions.
- **H3 — interaction:** the combination may outperform either individual mechanism, but this is an empirical question; negative or neutral interaction effects are meaningful results too.

These are hypotheses, not established findings.

## Implemented mechanism

### Per-organism predictive world model

Each organism stores a bounded table keyed by compact observation-state plus action. A row holds a count of observed next-state keys and online mean/second central moment of immediate reward. A confidence factor, count/(count+3), scales the learned reward contribution to action scoring. Predictive state/action entries are evicted in insertion order when the configured state cap is reached, and each row keeps at most 24 next-state outcomes.

Metrics:
- `world_model_reward_mae`: mean absolute error between a reward and the estimate available before observing that reward.
- `world_model_next_state_accuracy`: accuracy of the previous most-common next-state estimate on subsequent observations for a learned state/action row.
- `world_model_surprise`: smoothed normalized log-loss proxy for next-state outcomes.
- `world_model_entries` and `world_model_predictions`: model size and count of evaluated prior predictions.

The next-state metric is an online diagnostic, not a held-out generalization result. It is affected by the policy's changing visitation distribution. Report it alongside sample counts and controlled outcomes.

### Shared environmental signals

A resource-rich visit deposits a bounded forage trace; a hazard encounter deposits a danger trace. Trace fields decay each tick. All agents read a snapshot taken at the start of the tick, so early agent iteration order cannot benefit an organism with access to a fresh same-tick deposit. Genomic traits weight attraction to forage traces and aversion to danger traces. Signals are environmental state, not direct messages.

Metrics:
- `resource_trail_mass`, `danger_trail_mass`: field mean intensity.
- `collective_signal_coverage`: fraction of grid cells with a nontrivial signal in either field, capped at one.

## Controlled ablation design

The `--ablation-suite` experiment runs four conditions with the same seed schedule:
1. `baseline`: world model disabled, collective signals disabled.
2. `predictive_only`: world model enabled, collective signals disabled.
3. `collective_only`: world model disabled, collective signals enabled.
4. `full`: both enabled.

All conditions retain the same simulator, heuristic features, and evolution setting unless separately configured. Because policy-dependent visitation changes the experience distribution, this estimates whole-system effects of enabling each mechanism, not isolated effects on an identical logged transition stream. A stricter follow-up should replay a fixed trajectory dataset to compare predictive models on exactly matched samples.

Example quick run:

```bash
python experiments.py --ablation-suite --seeds 10 --seed-start 20261015 \
  --ticks 2000 --grid-size 48 --population 8 --max-population 16 \
  --max-organisms 200 --output v3-ablation.csv
```

All scenarios:

```bash
python experiments.py --all-scenarios --ablation-suite --seeds 10 \
  --ticks 2000 --grid-size 48 --population 8 --max-population 16 \
  --max-organisms 200 --output v3-matrix.csv
```

Use matched seeds between conditions. Do not interpret a single seed or the online model metrics alone as proof of improved intelligence.

## Primary outcomes to analyze from CSV

Compare each condition and scenario using:
- survival and population stability: `alive`, `deaths`, `total_created`
- task outcomes: `goals_reached`, `mean_energy`, `resource_coverage`
- learning/prediction: `mean_prediction_error`, `world_model_reward_mae`, `world_model_next_state_accuracy`
- diversity and reproduction: `genomic_diversity`, `births`, `max_generation`
- collective field usage: `resource_trail_mass`, `danger_trail_mass`, `collective_signal_coverage`

For each primary outcome, aggregate per-seed differences using paired comparisons. Report mean difference, dispersion, and confidence interval; account for population collapse and censored runs rather than silently dropping them. Define primary outcomes before looking at the full matrix to reduce selective reporting.

## Known limitations and next experiments

- This is a tabular transition model over hand-crafted bins, not a neural latent world model; it has no multi-step imagination or causal intervention model.
- The state's compact bins omit much of the spatial environment; next-state accuracy measures coarse states only.
- Reward prediction is policy-dependent and not evaluated on a held-out replay dataset yet.
- Resource traces can reinforce paths to sites that have already been depleted; test this against alternative decay rates, trail diffusion and resource-regrowth rules.
- Genetic parameters may confound the social effect. Follow-up experiments should freeze genomes or compare matched inherited trait distributions.
- The simulator currently updates organisms sequentially. The trail signal view is simultaneous within a tick, but resource depletion and occupancy interactions are still order-dependent.
- This platform does not demonstrate consciousness, general intelligence, or biological life.

Suggested next milestone: introduce a replay buffer and fixed transition dataset to measure held-out one-step predictions; then add an optional short-horizon model-predictive controller and compare it with the current one-step reward blend using identical seeds and preregistered outcomes.
