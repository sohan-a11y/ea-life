# EA-Life v2 threat model

## Boundaries

- The life simulator runs with network mode disabled and has no route to external networks.
- The dashboard is published on loopback only (`127.0.0.1:8000`). Its network is an ordinary bridge with IP masquerading off, so it has no working route to the internet. (An `internal: true` network would block the published port as well.)
- The dashboard mounts only the public snapshot volume and a narrow control volume. It does not mount the request queue, raw checkpoint, genomes, Q tables or full episodic memory.
- The broker accepts only fixed topic identifiers, uses HTTPS GET against a constant Wikipedia API endpoint, rate-limits requests, bounds response sizes and writes short snippets. It has no listening socket or published port.
- Containers run as a dedicated unprivileged UID, with read-only root filesystem, all Linux capabilities dropped, no-new-privileges and resource limits.

## Important limitation

The broker's destination restriction is an application-level allowlist, not a host-enforced egress policy. A compromised broker may attempt other outbound connections if the host network permits it. On a dedicated Linux VM, pair this design with host firewall rules or an outbound proxy that permits only the Wikipedia API hostname. Test egress policy rather than assuming application validation is equivalent.

## Dashboard

Keep port 8000 private. The API exposes state metrics and only pause, resume and reset commands. It rejects unsupported actions, validates body size and checks the Origin header where available. Do not put the dashboard directly on the public internet without authentication, TLS termination, rate limiting and stronger CSRF protections.

## Do not add

Do not mount the Docker socket, host filesystem, user home directory, cloud credentials, SSH keys, repository credentials or personal data into the life container. Organisms do not generate executable files or issue shell commands. Research content is untrusted data and must never be evaluated or run.
