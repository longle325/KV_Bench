# KV Bench Physical Nodes

KV Bench Physical Nodes is a small benchmark runner for evaluating KV-cache
reuse and distributed vLLM serving across two physical machines.

It focuses on two practical questions:

- Can independent vLLM replicas reuse prefix KV through LMCache and Redis?
- Can one vLLM service span two machines through Ray pipeline or tensor
  parallelism?

The repository intentionally keeps operational secrets and machine-specific
values out of git. Configure hosts, users, ports, interfaces, and model choices
in a local `.env` file created from the example template.

## Layout

```text
.env.example                 # template only; copy to .env locally
requirements.txt             # Python runtime packages
configs/                     # reference LMCache configs
scripts/
  start_track_a.sh           # one-command Track A runner
  start_track_b.sh           # one-command Track B runner
kvbench_physical/
  clients/                   # OpenAI-compatible benchmark client
  services/                  # Redis, Ray, vLLM, runtime helpers
  workflows/                 # Track A, Track B, preflight, smoke tests
```

Generated `logs/`, `results/`, and `reports/` directories are local artifacts
and are ignored by git.

## Tracks

Track A starts two independent vLLM replicas, one per machine, and compares:

- no cache
- LMCache CPU cache
- LMCache Redis backend

Track B starts a Ray cluster and launches one distributed vLLM service. The
second machine joins as a Ray worker; it does not expose a separate public vLLM
API.

## Setup

Create a local environment file:

```bash
cp .env.example .env
```

Edit `.env` and fill:

- `A_IP`, `B_IP`
- `A_IFACE`, `B_IFACE`
- `REMOTE_USER`, `REMOTE_HOST`, `SSH_PORT`, `REMOTE_ROOT`
- `MODEL`
- Track-specific request, concurrency, prefix, and output-token knobs

Use plain `KEY="value"` assignments. Do not commit `.env`.

Install Python packages in your own environment, or allow the runner to create
a user-local virtual environment:

```bash
python -m pip install -r requirements.txt
```

For platform-specific wheels, set `PIP_FIND_LINKS` or `WHEEL_FIND_LINKS`
locally instead of committing private wheel indexes.

## Run

From the repository root:

```bash
bash scripts/start_track_a.sh
bash scripts/start_track_b.sh
```

For long runs, start the same commands inside `screen` or another process
manager you normally use.

Optional runtime install is gated:

```bash
ALLOW_RUNTIME_INSTALL=1 bash scripts/start_track_b.sh
```

## Useful Commands

Run the Python CLI directly:

```bash
PYTHONPATH=. python -m kvbench_physical --help
```

Sync the runnable bundle to the second machine:

```bash
PYTHONPATH=. python -m kvbench_physical sync-to-b
```

Run Ray/GPU smoke tests:

```bash
PYTHONPATH=. python -m kvbench_physical run-ray-smoke-tests
```

## Artifact Policy

Commit source, configs, scripts, and the `.env.example` template.

Do not commit:

- `.env`
- `logs/`
- `results/`
- `reports/`
- Python cache files
- private hostnames, usernames, IPs, ports, or wheel indexes

## Notes

Ray coordinates distributed workers, but it does not merge two GPUs into one
shared VRAM pool. vLLM performs the actual model partitioning. Pipeline
parallelism usually requires less cross-machine communication than tensor
parallelism on ordinary Ethernet.
