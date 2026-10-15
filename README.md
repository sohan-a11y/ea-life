# EA-Life v2 — artificial-life research platform

EA-Life v2 advances the original grid experiment into a configurable research platform for embodied agents, intrinsic motivation, online prediction, bounded memory, evolutionary variation and repeatable task experiments. It is split into a network-isolated simulator, a narrow read-only research broker, and a dashboard that exposes observations rather than private genome or memory state.

This is **an artificial-life simulation, not evidence of consciousness or general intelligence**. Each capability has an explicit implementation and measurable proxy; research conclusions should come from repeated controlled runs, baselines and exports, not a single visually interesting run.

## What is new in v2

- **Four behavior roles:** forager, explorer, navigator and hazard-avoider. Run a mixed ecology or a single-task scenario.
- **Online prediction and value learning:** bounded state-action Q tables, per-action reward models, temporal-difference updates and prediction-error tracking.
- **Intrinsic motivation:** novelty, uncertainty, learning progress and risk contribute to a distinct curiosity signal.
- **Evolution:** mutable numerical genomes, energy budgets, lineage IDs, inherited traits, birth/death accounting and population caps.
- **Bounded memory:** compact recent transitions, bounded state-action memory and bounded semantic term counts.
- **Research protocol:** only named topic IDs enter the broker queue. The broker performs allowlisted Wikipedia HTTPS GETs, returns short snippets and never executes retrieved text.
- **Experiment observability:** resource and hazard layers, beacons, individual organisms, lineages, task completions, diversity, prediction error and population history.
- **Repeatability:** deterministic simulation seed, checkpoints, experiment summaries, configurable scenario and CI tests.

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

4. Open the forwarded port 8000 named EA-Life v2 dashboard.

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
| SIM_SEED | 20261015 | Python RNG seed |
| RESUME | 0 | Set to 1 to load the checkpoint volume |

For a controlled comparison, change one factor at a time, record the seed and scenario, and compare multiple seeds. A single run is not evidence that a mutation or behavior is generally superior.

## Architecture

- **ea_life/engine.py:** world state, agents, policy scoring, learning, memory, curiosity, reproduction and metrics.
- **runner.py:** simulation lifecycle, checkpointing, event persistence and pause/resume/reset flags.
- **dashboard.py:** standard-library HTTP server with narrow same-origin control actions.
- **broker.py:** fixed-topic, read-only reference lookup.
- **web/:** interactive dashboard assets.
- **compose.yaml:** separate hardened containers and volume boundaries.
- **tests/:** deterministic unit and protocol tests.

## Security boundaries

The life container uses network mode none, a non-root UID, read-only root filesystem, dropped capabilities, no-new-privileges, and CPU, memory and process limits. The dashboard does not mount the request queue or inspect genomes. The broker has no listener or published port and only accepts fixed topic identifiers.

The broker's destination restriction is implemented in application code; Compose alone is **not** a host-level outbound firewall. For strong containment, run on a dedicated Linux VM and enforce broker egress with firewall rules or a restrictive proxy. Never mount the Docker socket, host filesystem, cloud credentials or personal files into the life container.

## Current scientific limitations

The action-value model is small and handcrafted, the semantic memory is a bounded count of words from short snippets, and roles are simple task labels. This is not a language-capable agent, a general world model or a neural cellular automaton. Future research milestones should be separate, measured modules: richer predictive state representations, multi-cell collective behaviors, ablations, multi-seed benchmarks, and optionally a locally hosted language model behind a constrained planner interface.
