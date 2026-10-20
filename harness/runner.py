"""Primary DynaComSec runner.

This entrypoint now uses the chi-bench-style Harbor/Docker execution path.
The previous lightweight per-turn runner has been superseded by
``dynacomsec.experiment.runner`` and this module forwards to it so existing
commands using ``python -m dynacomsec.harness.runner`` keep working.
"""

from __future__ import annotations

from dynacomsec.experiment.runner import (
    build_harbor_command,
    main,
    run_experiment,
)


__all__ = ["build_harbor_command", "run_experiment", "main"]


if __name__ == "__main__":
    main()
