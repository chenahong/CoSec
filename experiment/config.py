"""Experiment configuration for DynaComSec Harbor-style runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    scenario: str
    agent: str = "codex"
    model: str | None = None
    task_dir: str | None = None
    trials_dir: str = "dynacomsec/results/experiments"
    environment: str = "docker"
    image: str = "dynacomsec:latest"
    force_build: bool = False
    execute_setup: bool = True
    judge: bool = False
    judge_model: str | None = None
    agent_command: str | None = None
    agent_timeout: int = 3600
    extra_env: dict[str, str] = field(default_factory=dict)
