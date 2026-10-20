"""Local Docker Harbor environment for DynaComSec trials."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import threading
import urllib.parse
from collections.abc import Mapping
from pathlib import Path

try:
    from harbor.environments.base import BaseEnvironment, ExecResult
    from harbor.environments.capabilities import EnvironmentCapabilities
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.paths import EnvironmentPaths
except ModuleNotFoundError:  # pragma: no cover - Harbor is optional until experiment runtime.
    BaseEnvironment = object  # type: ignore[assignment,misc]
    ExecResult = None  # type: ignore[assignment]
    EnvironmentCapabilities = None  # type: ignore[assignment]
    EnvironmentType = None  # type: ignore[assignment]
    EnvironmentPaths = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "dynacomsec:latest"

FORWARDED_ENV_KEYS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "GPTOSS_ZZ_API_KEY",
    "HERMES_GPTOSS_ZZ_API_KEY",
    "CODEX_GPTOSS_ZZ_API_KEY",
    "OPENCLAW_LLAMA4_ZZ_API_KEY",
    "HERMES_LLAMA4_ZZ_API_KEY",
    "OPENCLAW_OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENROUTER_API_KEY",
    "NVIDIA_API_KEY",
    "NVIDIA_KIMI_API_KEY",
    "ALIYUN_MINIMAX_API_KEY",
    "ALIYUN_GLM52_API_KEY",
    "DEEPROUTER_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "DEEPSEEK_API_KEY",
    "TOGETHER_API_KEY",
    "DYNACOMSEC_SPEAKER_API_KEY",
    "DYNACOMSEC_SPEAKER_BASE_URL",
    "DYNACOMSEC_SPEAKER_MODEL",
    "DYNACOMSEC_NORMAL_API_KEY",
    "DYNACOMSEC_NORMAL_BASE_URL",
    "DYNACOMSEC_NORMAL_MODEL",
    "DYNACOMSEC_ATTACKER_API_KEY",
    "DYNACOMSEC_ATTACKER_BASE_URL",
    "DYNACOMSEC_ATTACKER_MODEL",
    "DYNACOMSEC_JUDGE",
    "DYNACOMSEC_JUDGE_API_KEY",
    "DYNACOMSEC_JUDGE_BASE_URL",
    "DYNACOMSEC_JUDGE_MODEL",
    "DYNACOMSEC_ENABLE_MCP",
)

PROFILE_HOME_MOUNTS: tuple[tuple[str, str], ...] = (
    ("DYNACOMSEC_OPENCLAW_HOME_HOST", "/root/.openclaw"),
    ("DYNACOMSEC_CODEX_HOME_HOST", "/root/.codex"),
    ("DYNACOMSEC_CLAUDE_HOME_HOST", "/root/.claude"),
    ("DYNACOMSEC_HERMES_HOME_HOST", "/root/.hermes"),
)

_CONTAINER_NAME_INVALID = re.compile(r"[^a-zA-Z0-9_.-]")
_ACTIVE_CONTAINERS: list[str] = []
_ACTIVE_CONTAINERS_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def sanitize_container_name(name: str) -> str:
    safe = _CONTAINER_NAME_INVALID.sub("_", name)
    if not safe or not safe[0].isalnum():
        safe = "d_" + safe
    return safe[:64]


def _resolve_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "dynacomsec" / "docker" / "Dockerfile").is_file():
            return candidate
    return None


def _parse_unset_env(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in stripped.split(",")]
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("unset_env must be a JSON list or comma-separated string")
    return [str(item) for item in parsed if str(item)]


def _docker_proxy_url(value: str) -> str:
    """Make a host-local proxy reachable from a Docker container."""

    parsed = urllib.parse.urlsplit(value)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return value
    hostname = "host.docker.internal"
    netloc = hostname
    if parsed.port:
        netloc += f":{parsed.port}"
    if parsed.username or parsed.password:
        credentials = parsed.username or ""
        if parsed.password is not None:
            credentials += f":{parsed.password}"
        netloc = f"{credentials}@{netloc}"
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def build_docker_run_argv(
    *,
    image: str,
    container_name: str,
    task_id: str,
    host_env: Mapping[str, str],
    extra_env: Mapping[str, str] | None = None,
    profile_home_hosts: Mapping[str, str] | None = None,
    unset_env: list[str] | tuple[str, ...] | None = None,
    allow_internet: bool = True,
    detached: bool = True,
) -> list[str]:
    unset = set(unset_env or ())
    argv = ["docker", "run"]
    if detached:
        argv.append("-d")
    argv += ["--name", container_name]
    argv += ["--entrypoint", "/usr/bin/tini"]
    if not allow_internet:
        argv += ["--network", "none"]
    argv += ["-e", f"DYNACOMSEC_TASK_ID={task_id}"]
    task_host_dir = (extra_env or {}).get("DYNACOMSEC_TASK_HOST_DIR")
    if task_host_dir:
        argv += ["-v", f"{task_host_dir}:/opt/dynacomsec/tasks/{task_id}:ro"]
    for env_key, container_path in PROFILE_HOME_MOUNTS:
        if env_key in unset:
            continue
        host_path = (
            (profile_home_hosts or {}).get(env_key)
            or host_env.get(env_key)
            or (extra_env or {}).get(env_key)
        )
        if host_path:
            argv += ["-v", f"{Path(host_path).expanduser().resolve()}:{container_path}"]
    for key in FORWARDED_ENV_KEYS:
        if key in unset:
            continue
        value = host_env.get(key)
        if value:
            if key in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}:
                value = _docker_proxy_url(value)
            argv += ["-e", f"{key}={value}"]
    if extra_env:
        for key, value in extra_env.items():
            if key in unset:
                continue
            if value is not None:
                argv += ["-e", f"{key}={value}"]
    argv += [image, "--", "/usr/local/bin/dynacomsec-entrypoint.sh", "sleep", "infinity"]
    return argv


def build_docker_exec_argv(
    *,
    container_name: str,
    command: str,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    user: str | int | None = None,
) -> list[str]:
    argv = ["docker", "exec"]
    if cwd:
        argv += ["-w", cwd]
    if env:
        for key, value in env.items():
            argv += ["-e", f"{key}={value}"]
    if user is not None:
        argv += ["-u", str(user)]
    argv += [container_name, "bash", "-c", command]
    return argv


if BaseEnvironment is object:
    class DynaComSecDockerEnvironment:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "Harbor is required for DynaComSecDockerEnvironment. "
                "Install Harbor or run through an environment where harbor is available."
            )
else:
    class DynaComSecDockerEnvironment(BaseEnvironment):  # type: ignore[misc,no-redef]
        """Single-image Docker environment for DynaComSec Harbor trials."""

        def __init__(
            self,
            *args,
            image: str = DEFAULT_IMAGE,
            force_build: bool | None = None,
            host_env: Mapping[str, str] | None = None,
            openclaw_home_host: str | None = None,
            codex_home_host: str | None = None,
            claude_home_host: str | None = None,
            hermes_home_host: str | None = None,
            unset_env: str | list[str] | tuple[str, ...] | None = None,
            **kwargs,
        ) -> None:
            self._image = image
            self._force_build_default = bool(force_build) if force_build is not None else False
            self._host_env = dict(host_env) if host_env is not None else dict(os.environ)
            self._profile_home_hosts = {
                env_key: host_path
                for env_key, host_path in (
                    ("DYNACOMSEC_OPENCLAW_HOME_HOST", openclaw_home_host),
                    ("DYNACOMSEC_CODEX_HOME_HOST", codex_home_host),
                    ("DYNACOMSEC_CLAUDE_HOME_HOST", claude_home_host),
                    ("DYNACOMSEC_HERMES_HOME_HOST", hermes_home_host),
                )
                if host_path
            }
            self._unset_env = _parse_unset_env(unset_env)
            self._container_name: str | None = None
            super().__init__(*args, **kwargs)

        @staticmethod
        def type() -> str:
            return EnvironmentType.DOCKER

        @property
        def capabilities(self):
            return EnvironmentCapabilities(gpus=False, disable_internet=False, windows=False, mounted=False)

        @property
        def env_paths(self):
            return EnvironmentPaths.for_os(self.task_env_config.os)

        def _validate_definition(self) -> None:
            return

        @classmethod
        def preflight(cls) -> None:
            if shutil.which("docker") is None:
                raise SystemExit("docker CLI not found on PATH.")

        @property
        def task_id(self) -> str:
            return self.environment_name

        @property
        def container_name(self) -> str:
            if self._container_name is None:
                self._container_name = sanitize_container_name(self.session_id)
            return self._container_name

        async def start(self, force_build: bool) -> None:
            if force_build or self._force_build_default:
                await self._build_image()
            argv = build_docker_run_argv(
                image=self._image,
                container_name=self.container_name,
                task_id=self.task_id,
                host_env=self._host_env,
                extra_env=self._persistent_env,
                profile_home_hosts=self._profile_home_hosts,
                unset_env=self._unset_env,
                allow_internet=self._network_is_public,
            )
            result = await _run_subprocess(argv)
            if result.return_code != 0:
                raise RuntimeError(f"docker run failed: {result.stderr or result.stdout}")
            _register_container(self.container_name)
            await self.exec("mkdir -p /logs/agent /logs/verifier /logs/artifacts && chmod -R 777 /logs", user="root")
            await self._wait_for_ready()

        async def stop(self, delete: bool) -> None:
            if self._container_name is None:
                return
            await _run_subprocess(["docker", "rm", "-f", self._container_name])
            _unregister_container(self._container_name)

        async def exec(
            self,
            command: str,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ):
            argv = build_docker_exec_argv(
                container_name=self.container_name,
                command=command,
                cwd=cwd or self.task_env_config.workdir,
                env=self._merge_env(env),
                user=self._resolve_user(user),
            )
            return await _run_subprocess(argv, timeout_sec=timeout_sec)

        async def upload_file(self, source_path: Path | str, target_path: str) -> None:
            result = await _run_subprocess(["docker", "cp", str(source_path), f"{self.container_name}:{target_path}"])
            if result.return_code != 0:
                raise RuntimeError(f"docker cp upload failed: {result.stderr or result.stdout}")

        async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
            await self.exec(f"mkdir -p {shlex.quote(target_dir)}", user="root")
            src = str(source_dir).rstrip("/") + "/."
            result = await _run_subprocess(["docker", "cp", src, f"{self.container_name}:{target_dir}"])
            if result.return_code != 0:
                raise RuntimeError(f"docker cp upload_dir failed: {result.stderr or result.stdout}")

        async def download_file(self, source_path: str, target_path: Path | str) -> None:
            target = Path(target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            result = await _run_subprocess(["docker", "cp", f"{self.container_name}:{source_path}", str(target)])
            if result.return_code != 0:
                raise RuntimeError(f"docker cp download failed: {result.stderr or result.stdout}")

        async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            result = await _run_subprocess(["docker", "cp", f"{self.container_name}:{source_dir.rstrip('/')}/.", str(target)])
            if result.return_code != 0:
                raise RuntimeError(f"docker cp download_dir failed: {result.stderr or result.stdout}")

        async def _build_image(self) -> None:
            repo_root = _resolve_repo_root()
            if repo_root is None:
                raise RuntimeError("Could not find dynacomsec/docker/Dockerfile above package.")
            dockerfile = repo_root / "dynacomsec" / "docker" / "Dockerfile"
            result = await _run_subprocess(["docker", "build", "-f", str(dockerfile), "-t", self._image, str(repo_root)])
            if result.return_code != 0:
                raise RuntimeError(f"docker build failed: {result.stderr or result.stdout}")

        async def _wait_for_ready(self) -> None:
            probe = "from pathlib import Path; assert Path('/tmp/dynacomsec-ready').exists()"
            for _ in range(90):
                result = await self.exec(f"python -c {shlex.quote(probe)}", user="root", timeout_sec=10)
                if result.return_code == 0:
                    return
                await asyncio.sleep(1)
            logs = await _run_subprocess(["docker", "logs", "--tail", "120", self.container_name])
            diagnostics = await self._ready_failure_diagnostics()
            detail = "\n".join(part for part in (logs.stderr or logs.stdout, diagnostics) if part)
            if detail:
                raise RuntimeError(
                    f"DynaComSec container {self.container_name} did not become ready.\n"
                    f"Startup diagnostics:\n{detail}"
                )
            raise RuntimeError(f"DynaComSec container {self.container_name} did not become ready.")

        async def _ready_failure_diagnostics(self) -> str:
            command = r"""
set +e
echo '--- ps ---'
ps -ef | sed -n '1,80p'
echo '--- task dir ---'
ls -la "/opt/dynacomsec/tasks/${DYNACOMSEC_TASK_ID}" 2>&1
echo '--- openclaw home ---'
ls -la /root/.openclaw 2>&1
echo '--- setup.log ---'
tail -n 120 /logs/artifacts/setup.log 2>&1
echo '--- openclaw-workspace.log ---'
tail -n 80 /logs/artifacts/openclaw-workspace.log 2>&1
echo '--- artifacts dir ---'
find /logs/artifacts -maxdepth 2 -type f -print 2>&1
"""
            result = await self.exec(command, user="root", timeout_sec=10)
            return result.stdout or result.stderr


def _register_container(name: str) -> None:
    global _ATEXIT_REGISTERED
    with _ACTIVE_CONTAINERS_LOCK:
        if name not in _ACTIVE_CONTAINERS:
            _ACTIVE_CONTAINERS.append(name)
        if not _ATEXIT_REGISTERED:
            atexit.register(_terminate_tracked_containers)
            _ATEXIT_REGISTERED = True


def _unregister_container(name: str) -> None:
    with _ACTIVE_CONTAINERS_LOCK:
        if name in _ACTIVE_CONTAINERS:
            _ACTIVE_CONTAINERS.remove(name)


def _terminate_tracked_containers() -> None:
    with _ACTIVE_CONTAINERS_LOCK:
        names = list(_ACTIVE_CONTAINERS)
        _ACTIVE_CONTAINERS.clear()
    for name in names:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


async def _run_subprocess(argv: list[str], *, timeout_sec: int | None = None):
    if ExecResult is None:
        raise RuntimeError("Harbor ExecResult is unavailable because harbor is not installed.")
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        if timeout_sec is None:
            stdout_b, stderr_b = await proc.communicate()
        else:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return ExecResult(stdout="", stderr=f"timeout after {timeout_sec}s", return_code=124)
    return ExecResult(
        stdout=stdout_b.decode("utf-8", errors="replace") if stdout_b else "",
        stderr=stderr_b.decode("utf-8", errors="replace") if stderr_b else "",
        return_code=proc.returncode if proc.returncode is not None else -1,
    )
