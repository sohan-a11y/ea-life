# EA-Life — artificial-life research sandbox

EA-Life is a small artificial-life experiment. Four digital organisms move through a 128 × 128 resource world, estimate action outcomes, explore unfamiliar places, request selected reference topics, and reproduce with mutated inherited traits. A browser dashboard displays the world and experiment metrics.

This is a research simulation, not a claim of consciousness or general intelligence. Curiosity is an engineered intrinsic-motivation signal; learning is a lightweight action-value predictor; reproduction uses explicit simulation rules.

## Architecture and security

- **Life process:** runs with Docker network mode set to none. It cannot open a network connection.
- **Dashboard:** serves the local visual interface and a narrow pause/resume API. It reads public snapshots, not genomes or detailed episodic memory.
- **Research broker:** the only process intended to access the internet. It accepts fixed topic IDs and makes bounded HTTPS GET requests to the English Wikipedia API. It has no listener or published port; it accepts no arbitrary URL and uses no POST/PUT/DELETE method.
- **Local queues:** the simulation submits topic IDs through a shared request volume. Retrieved text is treated as untrusted data, never as executable instructions.

The broker’s destination allowlist is enforced in application code, not by a host firewall. For stronger guarantees, enforce broker egress with host firewall rules or a restrictive outbound proxy on a dedicated Linux VM. Never mount the Docker socket or host filesystem into the life container.

## Run in GitHub Codespaces

1. Open this repository and select Code → Codespaces → Create codespace.
2. The dev container includes Docker-in-Docker and forwards port 8000 as Private.
3. In the Codespaces terminal, run:

   ~~~bash
   docker compose up --build
   ~~~

4. Open forwarded port 8000 labelled EA-Life visual dashboard. Keep it private because it includes simulation controls.

Codespaces is a development session, not permanent hosting. The dashboard works while the Codespace and services are running. For continuous hosting, deploy a separately secured front end and keep the life process isolated on a host you control.

## Run locally

Requirements: Docker Engine and Docker Compose v2.

~~~bash
docker compose up --build
~~~

Open http://localhost:8000. Use Pause and Resume to control the experiment. Press Ctrl+C to stop the stack. To remove containers and all saved experiment data:

~~~bash
docker compose down -v
~~~

The first build needs internet to pull base images. During the experiment, only the broker has network access; the life process has no network interface.

## Included features

- 128 × 128 resource world and four starting organisms
- Evolvable motivational weights, novelty tracking, prediction-error estimates and learning progress
- Energy costs, explicit reproduction, genome mutation and lineage IDs
- Bounded topic requests and sanitized research snippets
- Live visual grid, organism registry, population history, recent events and pause/resume controls
- Docker hardening, Codespaces configuration and unit tests

## Known limitations

This is an educational prototype. It does not yet accept arbitrary English tasks, use a general-purpose language model, interpret native machine code, or demonstrate autonomous general intelligence. Web research is limited to fixed topics and simple term-count retention. These are separate future research milestones that should be added behind tests and explicit permission boundaries.

## Data handling

Experiment state uses the Docker named volume life_state. Request and result queues use separate volumes. The browser UI makes same-origin API calls only. The dashboard API accepts pause/resume commands only; it does not expose arbitrary shell execution.
