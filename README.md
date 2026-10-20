# CoSec Code Supplement

This package contains the execution and verification code for CoSec, an
executable benchmark of privacy and authorization enforcement in persistent
LLM agent systems. The full scenario dataset and raw trial outputs are
distributed separately and are not included in the code archive.

## Requirements

- Python 3.12
- Docker
- Harbor 0.20 or 0.21 for full experiments
- The native CLI for each evaluated harness

The reported environment used Python 3.12.14 and Harbor 0.21.0. Runtime images
contained OpenAI Python 3.3.1 and MCP 2.0.0. The evaluated native harnesses were
OpenClaw 2026.7.1-2, Hermes Agent 0.20.4, Codex CLI 0.148.0, and Claude Code
2.1.237 where applicable.

Install the package from this directory with

```bash
python -m pip install --upgrade "pip>=24" setuptools wheel
python -m pip install -e '.[dev]'
```

The archive directory may have any name after installation. Imports and
commands use the Python package name `dynacomsec` for compatibility with the
retained experiment artifacts.

Build the base runtime image from the archive root with

```bash
docker build -f docker/Dockerfile -t dynacomsec:base .
```

## Dataset Layout

Place the separately released scenario files under a local directory and pass
an explicit scenario path to every run. A scenario must follow
`dynacomsec.scenario.v2` and should retain its release manifest and SHA-256
digest. Do not combine results from same-ID scenarios whose hashes differ.

## Configuration

`configs/model_registry.example.toml` documents the 13 harness and model
combinations in the paper with official provider identifiers and endpoints.
It is an illustrative template rather than an active default or a record of
every routing endpoint used in the reported runs. Create a separate
`configs/model_registry.local.toml` if execution is needed, and never commit
credentials with the supplement.

The profiles reference `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `ZHIPU_API_KEY`,
`DASHSCOPE_API_KEY`, `MINIMAX_API_KEY`, and `OPENAI_API_KEY` as applicable.

Run a scenario with

```bash
python -m dynacomsec.harness.runner \
  --scenario /path/to/scenario.json \
  --agent openclaw \
  --model-profile openclaw.deepseek_4_1_flash \
  --model-registry configs/model_registry.local.toml \
  --judge
```

The runner creates an isolated Harbor task and stores the trajectory, tool
events, artifacts, verifier evidence, scorecard, and reward in the selected
trial directory. The verifier marks missing, truncated, provider-error, or
unverifiable trajectories inconclusive rather than safe.

## Release Scope

The code archive should include the Python packages, Docker files, two scenario
examples in `dataset_examples/`, example configuration, and documentation.
Exclude the full dataset, `results/`, `audit/`, local model registries,
environment files, caches, and editor metadata. A license file must
be added before public release using the license selected by the authors.
