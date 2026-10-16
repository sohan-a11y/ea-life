# Research v5: a learned local trail rule (neural cellular automaton)

## Question

EA-Life's shared trail fields are hand-coded: an organism deposits into the cell it enters (food trail: resource × 0.24 where resource ≥ 0.18; danger trail: hazard × 0.32), and at the end of each tick every cell decays on its own (× 0.985 and × 0.96). Does a learned local rule, in which every cell is updated from its 3×3 neighbourhood with shared weights (a neural cellular automaton), improve foraging?

## Design (fixed in `nca_experiment/nca_trails_prereg.json` before any run)

- Protocol: forage scenario, grid 48, 8 → 16 organisms, 600 ticks, evolution on, tabular world model on, latent model and imagination off (to isolate the trails). Primary outcome: `reward_per_agent_tick`.
- Seeds: 4 for training, 10 for validation (all choices), 20 for the single test.
- Arms: hand-coded decay (reference); no trails; 5 hand-designed CA variants (slower decay, diffusion toward the neighbour mean); a learned linear 3×3 rule (20 parameters); a learned NCA (240 parameters).
- Training: antithetic evolution strategies (16 candidates per generation, σ 0.05, Adam, learning rate 0.03, 24 generations) on the training seeds; the weights were scored on the validation seeds every 6 generations and the best kept.
- Success rule: the arm with the best validation result beats the reference on the 20 test seeds with mean paired difference > 0, exact two-sided sign-flip p < 0.05 and a relative gain of at least 2%.
- Sanity checks (passed): with no rule installed, the hooked engine reproduces the unmodified v4 engine bit for bit; so do the NumPy decay rule and both learned rules at their starting weights.

## Validation results (10 seeds)

| Arm | Mean | Difference vs reference | Seeds better |
|---|---|---|---|
| Hand-coded decay (reference) | 0.8556 | | |
| no_trails | 0.8559 | +0.0003 | 6/10 |
| ca_pcur_k0.1 | 0.8587 | +0.0031 | 5/10 |
| ca_pcur_k0.25 | 0.8558 | +0.0002 | 5/10 |
| ca_pslow_k0.0 | 0.8556 | -0.0000 | 4/10 |
| ca_pslow_k0.1 | 0.8573 | +0.0017 | 6/10 |
| ca_pslow_k0.25 | 0.8596 | +0.0040 | 6/10 |
| learned linear, after generation 5 | 0.8573 | +0.0016 | 7/10 |
| learned linear, after generation 11 | 0.8613 | +0.0057 | 5/10 |
| learned linear, after generation 17 | 0.8563 | +0.0007 | 5/10 |
| learned linear, after generation 23 | 0.8591 | +0.0035 | 6/10 |
| learned nca, after generation 5 | 0.8581 | +0.0024 | 6/10 |
| learned nca, after generation 11 | 0.8518 | -0.0039 | 4/10 |
| learned nca, after generation 17 | 0.8529 | -0.0027 | 3/10 |
| learned nca, after generation 23 | 0.8602 | +0.0046 | 7/10 |

Selected for the test (best validation difference): **learned_linear_gen11**.

## Test results (20 held-out seeds; differences are arm minus reference, paired by seed)

| Scenario | Arm | Reference mean | Arm mean | Difference | Relative | Seeds better | Sign-flip p | 95% interval |
|---|---|---|---|---|---|---|---|---|
| forage (primary) | learned_linear_gen11 | 0.8592 | 0.8595 | +0.0003 | +0.0% | 12/20 | 0.869 | [-0.0037, +0.0044] |
| forage (primary) | no trails | 0.8592 | 0.8627 | +0.0035 | +0.4% | 14/20 | 0.086 | [-0.0005, +0.0075] |
| mixed (secondary) | learned_linear_gen11 | 1.0024 | 1.0020 | -0.0004 | -0.0% | 12/20 | 0.948 | [-0.0123, +0.0115] |
| mixed (secondary) | no trails | 1.0024 | 1.0010 | -0.0014 | -0.1% | 10/20 | 0.772 | [-0.0112, +0.0084] |

## Exploratory (not pre-registered): the learned NCA on the same test seeds

The NCA ranked second on validation (learned_nca_gen23, +0.0046), so the pre-registered rule did not send it to the test. Because the NCA was the point of the question, it was also run on the test seeds; treat this as exploratory.

| Scenario | Reference mean | NCA mean | Difference | Relative | Seeds better | Sign-flip p | 95% interval |
|---|---|---|---|---|---|---|---|
| forage | 0.8592 | 0.8637 | +0.0045 | +0.5% | 13/20 | 0.055 | [-0.0001, +0.0092] |
| mixed | 1.0024 | 1.0114 | +0.0090 | +0.9% | 12/20 | 0.069 | [-0.0008, +0.0188] |

Neither difference reaches p < 0.05, and both are far below the 2% bar. In foraging, the NCA's gain (+0.5%) is about the same as switching the trails off (+0.4%). Its validation curve was also unstable (+0.0024, −0.0039, −0.0027, +0.0046 at generations 5, 11, 17 and 23), which suggests overfitting to the 4 training seeds rather than a learned use of the field. The mixed-scenario result (+0.9%, p = 0.069) is a lead for a new pre-registered test, not a finding.

## Verdict

**The pre-registered success rule is not met: no evidence that a learned local trail rule beats the hand-coded trails on foraging.**

Context: on validation, switching the trails off entirely changed the outcome by +0.0003. In this environment the organisms already sense the true food and hazard of every neighbouring cell, so the trails carry little extra information, and any trail rule has little room to help.

## What would give an NCA room to matter

1. Make information non-local: hide the true resource values of neighbouring cells, so organisms must rely on shared signals.
2. Let the field carry more than decay: train the NCA jointly with the organisms' use of it (their genomes currently weight trails with `collective_drive` and `signal_aversion`).
3. Use tasks where coordination matters (shared targets, division of labour) rather than independent foraging.
