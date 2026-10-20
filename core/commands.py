"""Command execution helpers for setup, cleanup, and physical snapshots."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    skipped: bool = False


def run_command(command: str, timeout: int = 30, dry_run: bool = True) -> CommandResult:
    """Run a shell command or record it as skipped when dry_run is enabled."""

    if dry_run:
        return CommandResult(command=command, returncode=0, stdout="", stderr="", skipped=True)
    proc = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return CommandResult(
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout.strip(),
        stderr=proc.stderr.strip(),
        skipped=False,
    )


def run_commands(commands: list[str], timeout: int = 30, dry_run: bool = True) -> list[dict]:
    results = []
    for command in commands:
        result = run_command(command, timeout=timeout, dry_run=dry_run)
        results.append(result.__dict__)
    return results
