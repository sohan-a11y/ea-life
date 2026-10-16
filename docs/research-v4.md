# EA-Life v4 research protocol: neural latent world model and imagination

## Research goal

Test whether a learned compressed state representation and a learned transition model can improve decisions when rolled forward multiple steps. EA-Life v4 adds a small neural latent world model to the v3 artificial-life simulation while keeping the earlier tabular predictor, collective signals, and the neural planner independently controllable.

This is an intentionally small research prototype implemented with the Python standard library so it can run inside the existing network-isolated life container. It is not comparable in scale or sophistication to large model-based RL systems such as Dreamer or MuZero.

## Model architecture

- **Input observations (18 values):** normalized energy, local resource and hazard, novelty, relative beacon coordinates/distance, local collective traces, local resource/hazard averages, neighborhood crowding, normalized position, and a one-hot of the organism's task (forager, explorer, navigator, avoider). The task is needed because rewards depend on it and the model is shared by all organisms.
- **Encoder:** a tanh fully connected layer maps observations to an 8-dimensional latent vector.
- **Decoder:** a sigmoid fully connected layer reconstructs the 14 normalized observation values; this reconstruction loss provides a representation-learning signal.
- **Dynamics model:** a small tanh multilayer perceptron receives the latent state concatenated with a 9-way action one-hot vector. It predicts the change in the latent vector (next = current + change) and the immediate reward. The latent rows of its output layer start at one tenth of their usual scale.
- **Training:** online supervised transition learning plus reconstruction learning and sampling from a bounded shared replay pool. The model shares parameters across organisms; it does not train an independent neural network for every genome.
- **Imagination:** after 48 real transitions, bounded beam search composes repeated latent transitions, accumulating discounted predicted rewards and selecting the first action of the best sequence. By default it plans 4 steps with beam width 3. Both values are configurable and validated (horizon 1 to 12, beam width 1 to 8).
- **Safety and resource bounds:** replay capacity is capped at 2,048 transitions; the existing Docker memory/CPU/PID limits remain in force; the model cannot execute real-world actions or read from the network.

The training code uses hand-written backpropagation. This keeps the deployment dependency-free but is slower and less numerically mature than PyTorch/JAX. Treat it as a baseline implementation for method exploration.

## Hypotheses

- **H1 representation learning:** reconstruction loss will decrease with experience for stable state distributions.
- **H2 transition learning:** the dynamics model's online latent-transition and reward errors will decrease as experience accumulates.
- **H3 planning:** multi-step imagination may improve goal completion, cumulative reward proxies, energy and survival over a v3 control and one-step planning condition.
- **H4 failure modes:** model exploitation and compounding error may make long-horizon planning worse than shorter horizons. A negative result is informative.

None of these hypotheses is assumed true.

## Latent ablation study

Run with matched seeds and fixed settings:

~~~bash
python experiments.py --latent-ablation-suite --seeds 10 --ticks 2000 --grid-size 48 --population 8 --max-population 16 --max-organisms 200 --imagination-horizon 4 --imagination-beam-width 3 --output v4-latent-ablation.csv
~~~

This produces four conditions:
1. `v3_control`: same old world-model/collective controls, neural latent model disabled.
2. `latent_model_only`: neural representation and transition model train, but imagined plans do not influence action choice. Its behaviour, and therefore every outcome, is identical to `v3_control` on each seed by construction; compare it only on model-quality metrics.
3. `one_step_planning`: latent model plus horizon-one imagined planning.
4. `long_horizon_imagination`: latent model plus the requested multi-step horizon.

Use `--all-scenarios` for a scenario-by-condition matrix. The legacy `--ablation-suite` remains dedicated to v3's tabular predictor and collective-signal pathways; the two experiment suites cannot run together.

## Individual settings

- `SIM_LATENT_MODEL_ENABLED=0`: disable neural-model training and latent planning.
- `SIM_IMAGINATION_ENABLED=0`: keep learning but prevent imagined plans from affecting policy choice.
- `SIM_IMAGINATION_HORIZON=4`: number of simulated action transitions (maximum 12).
- `SIM_IMAGINATION_BEAM_WIDTH=3`: maximum candidate imagined paths retained at each depth (maximum 8).
- CLI equivalents: `--disable-latent-model`, `--disable-imagination`, `--imagination-horizon`, `--imagination-beam-width`.

The default model activates planning after at least 48 real transitions. Before warm-up it falls back to the existing policy. The chosen imagined action receives a small scoring bonus rather than overriding the existing policy outright.

## Metrics and interpretation

- `latent_transitions_seen`, `latent_replay_size`, `latent_training_updates`: learning and memory volume.
- `latent_reconstruction_loss`: smoothed autoencoder reconstruction MSE on new transitions, measured before training on them (prequential); replayed samples are not counted.
- `latent_transition_loss`: smoothed latent transition plus reward prediction loss.
- `latent_reward_mae`: smoothed immediate reward prediction MAE in environment-reward units.
- `latent_imagination_calls`, `latent_imagination_predictions`, `latent_last_plan_horizon`, `latent_last_plan_return`: planning activity and the model's estimated return.
- `latent_plan_adoption_rate`: share of planned decisions where the plan's first action was taken. `latent_plan_promotion_rate`: share where the plan's action was outside the greedy choice band until the bonus moved it in.
- `reward_per_agent_tick`: mean reward per organism per tick lived, over all organisms; the primary outcome.
- With `--heldout-ticks N`: `heldout_reward_mae`, `heldout_constant_reward_mae`, `heldout_latent_mse` and `heldout_persistence_latent_mse`, computed on the last N ticks with the model frozen.

These online losses are affected by the policy's changing data distribution. They are not held-out measures of generalization; use `--heldout-ticks` for that, and compare against the persistence and constant baselines. Analyse paired seeds with `python analyze.py`. In the default protocol `alive` and `deaths` sit at the population cap, so use `reward_per_agent_tick` as the primary outcome. The latent representation is not guaranteed to have semantic structure, and its dimensions are not individually interpretable. A robust follow-up should freeze evaluation seeds, keep a held-out transition set, and report paired per-seed differences with uncertainty intervals.

## Limitations and follow-up

- The input is a compact feature vector, not raw pixels or a complete world description.
- The model predicts a learned latent state and scalar reward; it has no explicit terminal-state head, uncertainty ensemble, or probabilistic latent transition.
- Beam search is a small planner; its value estimates can exploit model errors, especially at longer horizons.
- The model is shared across the population rather than individual-specific, which improves sample efficiency but blurs differences among role-specific dynamics.
- Parameters use plain Python lists and hand-coded gradients. A later implementation could optionally move training to PyTorch and retain the same simulator/model API.
- The 2-D environment's updates are not a biologically faithful ecology, and the experiment does not establish consciousness, general intelligence or biological life.

Next steps: add a held-out replay evaluator; test horizon sweeps (1, 2, 4, 6, 8, 12) across paired seeds; compare a one-step reward model against a latent rollout model; measure model exploitation and long-horizon error growth; then experiment with a convolutional/spatial encoder or neural cellular automata as a separate, more compute-intensive branch.
