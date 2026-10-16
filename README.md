# EA-Life v3 — predictive and collective artificial-life research platform

EA-Life v3 extends the grid experiment with per-organism predictive transition learning and shared environmental signals. It is a configurable research platform for embodied agents, intrinsic motivation, bounded memory, evolutionary variation, emergent collective behavior, and repeatable controlled experiments. It is split into a network-isolated simulator, a narrow read-only research broker, and a dashboard that exposes observations rather than private genome or memory state.

This is **an artificial-life simulation, not evidence of consciousness or general intelligence**. Each capability has an explicit implementation and measurable proxy; research conclusions should come from repeated controlled runs, baselines and exports, not a single visually interesting run.

## What is new in v3

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
| RESUME | 0 | Set to 1 to load the checkpoint volume |

For a controlled comparison, change one factor at a time, record the seed and scenario, and compare multiple seeds. A single run is not evidence that a mutation or behavior is generally superior.

## Architecture

- **ea_life/engine.py:** world state, bounded predictive transition models, shared trail dynamics, agent policies, learning, memory, curiosity, reproduction and metrics.
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

The action-value model and predictive transition model are small tabular systems; semantic memory is a bounded count of words from short snippets, and roles are simple task labels. The world model does not perform latent-state learning or long-horizon imagination, and collective traces do not imply centralized planning. This is not a language-capable agent, a general world model or a neural cellular automaton. Next research work should investigate multi-step predictive rollouts, richer latent representations, independent resource dynamics, spatial neural cellular automata, and preregistered multi-seed studies. Until those are measured, treat the current modules as testable prototypes.
