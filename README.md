# EA-Life v4 research branch — neural latent world model and imagination

This branch extends EA-Life v3 with a small learned neural latent representation, a latent transition/reward model, a shared bounded experience-replay pool and multi-step beam-search imagination. The model is implemented in pure Python to fit the existing network-isolated life container. The code lives on `research/v4-neural-latent-imagination`; it is intentionally separate from `main` while the research prototype is evaluated. The dashboard exposes diagnostic metrics rather than model weights or replay observations.

This is **an artificial-life simulation, not evidence of consciousness or general intelligence**. Each capability has an explicit implementation and measurable proxy; research conclusions should come from repeated controlled runs, baselines and exports, not a single visually interesting run.

## Inherited from v3

- **Four behavior roles:** forager, explorer, navigator and hazard-avoider. Run a mixed ecology or a single-task scenario.
- **Predictive world model:** each organism learns a bounded state/action-to-next-state distribution and reward estimate, using online counts and reward moments; prediction error, next-state accuracy and surprise are tracked separately from the Q-learning TD error.
- **Collective behavior:** organisms read shared environmental resource and danger traces. They deposit bounded signals, which decay over time; genomes evolve separate attraction and avoidance weights. Signals are read from a pre-tick snapshot, avoiding within-tick order advantage.
- **Ablation-ready experiments:** independently turn prediction and collective signals on/off; the four-condition suite compares baseline, predictive-only, collective-only and full systems on matched seeds.
- **Intrinsic motivation:** novelty, uncertainty, learning progress and risk contribute to a distinct curiosity signal.
- **Evolution:** mutable numerical genomes, energy budgets, lineage IDs, inherited traits, birth/death accounting and population caps.
- **Bounded memory:** compact recent transitions, bounded state-action memory and bounded semantic term counts.
- **Research protocol:** only named topic IDs enter the broker queue. The broker performs allowlisted Wikipedia HTTPS GETs, returns short snippets and never executes retrieved text.
- **Experiment observability:** resource and hazard layers, beacons, individual organisms, lineages, task completions, diversity, prediction error and population history.
- **Repeatability:** deterministic simulation seed, checkpoints, experiment summaries, configurable scenario and CI tests.

## Research milestone v3: prediction and collective behavior

This release adds two distinct mechanisms whose impact should be evaluated independently:

1. **Predictive transition model.** For a compact state key and action, each organism stores bounded counts of observed next-state keys and online reward moments. The model's reward estimate contributes to action scoring in proportion to its sample confidence. Per-organism storage is capped by `max_world_model_states`; each transition keeps at most 24 next-state outcomes. This remains a small tabular model, not a neural or general-purpose world model.
2. **Shared environmental traces.** Successful visits to resource-rich cells deposit a forage trace; encounters with hazards deposit a danger trace. All organisms read the same pre-tick view, and the traces decay each tick. This is an environmental-memory / stigmergy experiment, not direct communication or a claim of social intelligence.
3. **Evaluation, not just visualization.** The dashboard reports reward prediction MAE, next-state accuracy, surprise, model entries, trail coverage and trail mass. Run matched ablations with identical seeds to see whether the mechanisms improve task outcomes, survival, energy or goal completion. Use many seeds and confidence intervals before making research claims.

Quick four-condition ablation on the mixed scenario:

~~~bash
python experiments.py --ablation-suite --seeds 10 --ticks 2000 --grid-size 48 --population 8 --max-population 16 --max-organisms 200 --output v3-ablation.csv
~~~

Run the full scenario-by-ablation matrix:

~~~bash
python experiments.py --all-scenarios --ablation-suite --seeds 10 --ticks 2000 --grid-size 48 --population 8 --max-population 16 --max-organisms 200 --output v3-matrix.csv
~~~

Individual controls for custom comparisons are `--disable-world-model` and `--disable-collective`. Each CSV row records which pathways were enabled. Details and hypotheses are in [the v3 research protocol](docs/research-v3.md).

## Research milestone v4: neural latent model and imagination

The neural latent model learns a compact representation from 18 normalized features (14 local observations plus a one-hot of the organism's task) using an encoder/decoder, then learns latent transition and reward prediction with a small neural network. New transitions enter a bounded pool shared across organisms. After 48 transitions, a beam search can roll predicted latent states forward and score multi-step action sequences. The default horizon is four; supported values are 1 to 12, with beam width 1 to 8.

~~~bash
python experiments.py --latent-ablation-suite --seeds 10 --ticks 2000 --grid-size 48 --population 8 --max-population 16 --max-organisms 200 --imagination-horizon 4 --imagination-beam-width 3 --output v4-latent-ablation.csv
~~~

The four matched-seed conditions compare the v3 control, latent learning without planning, horizon-one planning and long-horizon imagination. This is a research prototype with manually implemented backpropagation and tabular-sized networks, not a production model-based reinforcement learning system. Online losses are not held-out evidence that the model generalizes. It is expected to run more slowly than v3; first test small populations and grids, then scale up. See [the v4 protocol](docs/research-v4.md).

### Review notes (16 Oct 2026): measured before drawing conclusions

These were measured on this branch; the scripts are in the review package.
- The hand-written gradients are correct (finite-difference check). Runs are deterministic for a given seed.
- **The task is now part of the latent observation.** Rewards depend on the task, and the model is shared by all organisms. Without the task, its reward error in the mixed scenario could not fall below what a linear model reaches; a linear model that knows the task had 61% lower held-out error (3 seeds).
- **Reported online errors are now prequential.** They count only new transitions, measured before the model trains on them. Previously, half of each average came from replayed samples the model had already trained on.
- **The dynamics model now predicts the change in the latent state** (next = current + change), and its outputs start small. Measured on 3 seeds (900 ticks of online training with imagination off, then the frozen model scored on the next 300 ticks):
  - Held-out latent error, relative to simply predicting "next latent = current", went from 1.21 to 0.40 (mixed) and from 2.26 to 0.51 (forage). The original model was worse than that trivial baseline; adding the task alone did not change this.
  - Held-out reward error fell from 0.271 to 0.160 (mixed) and from 0.108 to 0.078 (forage).
  - A linear model on the same inputs still predicts reward better (0.096 and 0.044), so the reward head remains the weak part.
- **`latent_model_only` behaves exactly like `v3_control`.** The model trains but never influences a decision, so its outcomes are identical on every seed. Use that arm for model-quality metrics only.
- **In the default protocol (grid 48, 8 → 16 organisms, 2000 ticks), the population stays at its cap.** So `alive` and `deaths` cannot show differences. Food regrows only up to 0.09 and coverage counts cells above 0.15, so eaten rich patches never come back: food coverage fell from 0.57 to below 0.01 between ticks 1500 and 1900 (5 seeds, documented settings). Hazards fade from 0.75 to 0.25, so the tasks change during a run. The primary outcome is `reward_per_agent_tick`, the quantity the planner maximises.
- **In a 4-seed pilot (mixed, 1000 ticks), imagination lowered reward per agent-tick in every seed, before and after the fixes.** Control: 0.784. One-step planning: −0.035 with the original model, −0.044 with the fixed model. Four-step imagination: −0.038 and −0.037. Beacon goals fell from 23 per run to about 8–9. Plans changed about 37% of decisions.
  - A likely reason: the default policy already scores each neighbouring cell with its true resource, hazard and novelty. That is an exact one-step look-ahead, so a learned one-step prediction can only add error. The 4-step plan has no value estimate at its end, so it cannot see gains beyond 4 steps.
  - To test imagination fairly, compare against a policy that lacks this look-ahead, or add a leaf value (for example from the Q-table). With 4 seeds the smallest possible p is 0.125, so treat this as a direction to test, not a result.
- **The curiosity score, learning progress and the Wikipedia research pathway** (terms, topics, fact counts) are recorded and displayed, but they never enter action choice, reward or energy. They only decide when a research request is sent.
- **The dashboard port could not be reached from the host.** The dashboard's only network was `internal: true`, and Docker does not publish ports for such containers. The first container test in CI (commit f3342d7: `curl` from the runner to 127.0.0.1:8000) failed, and so did the next one, which read the runtime port mapping. The test that passes now calls the API from inside the dashboard container, so it cannot catch this. Fix: the network is an ordinary bridge with IP masquerading off, so the published port works and the dashboard still has no working route to the internet. The port is published on 127.0.0.1 only. CI now tests the host path and checks that the dashboard cannot open an outbound connection.
- **New tools.**
  - `--heldout-ticks N` freezes the latent model for the last N ticks and reports held-out error next to two baselines: "next latent = current" and "constant reward".
  - `--imagination-bonus` sets how strongly the plan steers choices.
  - `latent_plan_adoption_rate` and `latent_plan_promotion_rate` measure how often plans are followed, and how often the bonus is what made the plan's action a top choice.
  - `python analyze.py results.csv` gives paired differences, 95% intervals, sign-flip p-values and Holm correction.

~~~bash
python experiments.py --latent-ablation-suite --seeds 10 --ticks 2000 --heldout-ticks 200 --grid-size 48 --population 8 --max-population 16 --max-organisms 200 --output v4-latent-ablation.csv
python analyze.py v4-latent-ablation.csv
python -m unittest discover -s tests -t . -v
~~~

## Run locally

Requires Docker Engine and Docker Compose v2.

~~~bash
docker compose up --build
~~~

Open http://localhost:8000. Keep the forwarded/private port private because the dashboard can pause, resume or reset the experiment. The initial run is bounded by SIM_TICKS. Use Ctrl+C to stop the Compose stack.

To start with a clean experiment, stop the stack and remove its data volumes. This is destructive.

~~~bash
docker compose down -v
~~~

## Repeatable offline experiment suite

Run matched seeds across scenarios and compare evolution-enabled runs with a no-reproduction baseline:

~~~bash
python experiments.py --all-scenarios --seeds 3 --ticks 600 --grid-size 48 --output experiment-results.csv
python experiments.py --scenario mixed --disable-evolution --seeds 3 --ticks 600 --grid-size 48 --output no-evolution.csv
~~~

The runner prints one summary row per scenario/seed and writes CSV files for analysis. It uses the same simulation engine with the external research broker disabled, so benchmark runs are offline and reproducible from the recorded seed. This is a small experiment harness, not a statistical claim: use more seeds and confidence intervals before concluding that a behavior or genome is superior.

## Run in Codespaces

1. Create a Codespace from this repository's default branch.
2. Wait for the dev container to initialize. Docker-in-Docker is configured and the dashboard port is forwarded as private.
3. If services did not start automatically, run:

   ~~~bash
   docker compose up --build -d
   ~~~

4. Open the forwarded port 8000 named EA-Life dashboard.

Codespaces is a development session, not a permanent host. The dashboard runs only while the Codespace and containers are running.

## Experiment settings

The default scenario is mixed, with twelve initial organisms and a maximum live population of forty.

| Environment variable | Default | Purpose |
|---|---:|---|
| SIM_GRID | 96 | Grid width and height |
| SIM_POPULATION | 12 | Initial population |
| SIM_MAX_POPULATION | 40 | Maximum living population |
| SIM_MAX_ORGANISMS | 500 | Total lifetime creation cap |
| SIM_SCENARIO | mixed | mixed, forage, explore, navigate or avoid_hazards |
| SIM_TICKS | 100000 | Maximum experiment ticks |
| TICK_SLEEP | 0.04 | Delay per tick; lower is faster |
| SAVE_EVERY | 10 | Checkpoint interval |
| MAX_EVENT_LOG_BYTES | 2000000 | Active JSONL event-log size cap; the previous log is rotated |
| BROKER_MAX_REQUESTS | 200 | Maximum number of broker queue entries processed before research access is marked exhausted |
| SIM_SEED | 20261015 | Python RNG seed |
| SIM_WORLD_MODEL_ENABLED | 1 | Set to 0 to disable the predictive transition model |
| SIM_WORLD_MODEL_STATES | 256 | Maximum state/action entries per organism |
| SIM_COLLECTIVE_ENABLED | 1 | Set to 0 to disable shared environmental trails |
| SIM_LATENT_MODEL_ENABLED | 1 | Train the neural latent model and replay pool |
| SIM_IMAGINATION_ENABLED | 1 | Allow imagined action plans to influence action choice |
| SIM_IMAGINATION_HORIZON | 4 | Imagined transitions per plan, 1 to 12 |
| SIM_IMAGINATION_BEAM_WIDTH | 3 | Candidate sequences retained per depth, 1 to 8 |
| SIM_IMAGINATION_BONUS | 0.45 | Score bonus for the imagined plan's first action, 0 to 10 (0 = no influence) |
| RESUME | 0 | Set to 1 to load the checkpoint volume |

For a controlled comparison, change one factor at a time, record the seed and scenario, and compare multiple seeds. A single run is not evidence that a mutation or behavior is generally superior.

## Architecture

- **ea_life/engine.py:** world state, v3 tabular model, collective signals, and integration with latent planning.
- **ea_life/latent_world_model.py:** neural encoder/decoder, latent transition/reward model, bounded shared replay, multi-step imagination and model diagnostics.
- **runner.py:** simulation lifecycle, checkpointing, event persistence and pause/resume/reset flags.
- **dashboard.py:** standard-library HTTP server with narrow same-origin control actions.
- **broker.py:** fixed-topic, read-only reference lookup.
- **web/:** interactive dashboard assets.
- **compose.yaml:** separate hardened containers and volume boundaries.
- **tests/:** deterministic unit and protocol tests.

## Security boundaries

The dashboard binds to loopback by default (`127.0.0.1:8000`). Change this only if you have configured appropriate authentication, TLS and network restrictions.

The life container uses network mode none, a non-root UID, read-only root filesystem, dropped capabilities, no-new-privileges, and CPU, memory and process limits. The dashboard does not mount the request queue or inspect genomes. The broker has no listener or published port and only accepts fixed topic identifiers.

The broker's destination restriction is implemented in application code; Compose alone is **not** a host-level outbound firewall. For strong containment, run on a dedicated Linux VM and enforce broker egress with firewall rules or a restrictive proxy. Never mount the Docker socket, host filesystem, cloud credentials or personal files into the life container.

## Current scientific limitations

The v4 neural latent model is a compact pure-Python research prototype. It learns from a hand-engineered local feature vector, not pixels; shares parameters across organisms; uses a smoothed latent state rather than a stochastic latent distribution; and may exploit inaccuracies during imagined rollouts. The hand-coded gradients are slower and less mature than a framework such as PyTorch. Evaluate on held-out transitions and paired seeds before making performance claims. This is not evidence of consciousness, general intelligence or biological life.
