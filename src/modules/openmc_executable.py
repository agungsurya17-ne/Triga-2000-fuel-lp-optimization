"""Resolve and validate an OpenMC executable before campaign execution.

resolve_openmc_executable accepts an explicit path or searches PATH. Helpers
check file existence and execution permissions during preflight and assembly."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


class OpenMCExecutableResolutionError(RuntimeError):
    """Report an unavailable or unusable OpenMC executable during preflight."""


def _is_explicit_path(value: str | Path) -> bool:
    """Return whether ``value`` denotes a path rather than a PATH command name.

    An explicit path is absolute or contains a path separator. This keeps a
    simple ``"openmc"`` (including ``Path("openmc")``) as a command to search
    through ``PATH`` and reserves project-relative handling for real paths."""
    value_text = str(value)
    return Path(value_text).is_absolute() or "/" in value_text or "\\" in value_text


def _resolution_error(requested: str | Path, detail: str) -> OpenMCExecutableResolutionError:
    """Build one actionable resolution error for ``requested`` executable input."""
    return OpenMCExecutableResolutionError(
        f"Executable OpenMC tidak dapat digunakan untuk {requested!r}: {detail}. "
        f"Python interpreter aktif: {sys.executable}. PATH proses: {os.environ.get('PATH', '')!r}. "
        "Gunakan kernel Jupyter openmc_env atau berikan openmc_executable_override=Path('/path/ke/openmc') yang absolut."
    )


def _validate_executable(path: Path, requested: str | Path) -> Path:
    """Validate ``path`` as an executable file and return its resolved absolute path."""
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise _resolution_error(requested, f"file tidak ditemukan: {resolved_path}")
    if not os.access(resolved_path, os.X_OK):
        raise _resolution_error(requested, f"file tidak memiliki izin executable: {resolved_path}")
    return resolved_path


def resolve_openmc_executable(
    openmc_executable_override: str | Path | None = None,
    configured_openmc_executable: str | None = None,
    *,
    project_root: str | Path,
) -> Path:
    """Resolve the OpenMC binary from override, config, then the process PATH.

    ``openmc_executable_override`` has highest priority and may be a command
    name or an explicit ``str``/``Path`` path. ``configured_openmc_executable``
    is the optional ``[paths]`` value. Explicit relative paths are resolved
    against ``project_root``; a plain command name is resolved only with
    :func:`shutil.which`. The return value is an absolute executable path."""
    if openmc_executable_override is not None and not isinstance(
        openmc_executable_override, (str, Path)
    ):
        raise TypeError("openmc_executable_override harus berupa str, pathlib.Path, atau None.")
    if configured_openmc_executable is not None and not isinstance(
        configured_openmc_executable, str
    ):
        raise TypeError("configured_openmc_executable harus berupa str atau None.")
    if not isinstance(project_root, (str, Path)):
        raise TypeError("project_root harus berupa str atau pathlib.Path.")

    requested = (
        openmc_executable_override
        if openmc_executable_override is not None
        else configured_openmc_executable
    )
    if requested is None:
        requested = "openmc"
    if isinstance(requested, str) and not requested.strip():
        raise _resolution_error(requested, "nama executable kosong")

    if _is_explicit_path(requested):
        configured_path = Path(requested).expanduser()
        candidate_path = (
            configured_path
            if configured_path.is_absolute()
            else Path(project_root) / configured_path
        )
        return _validate_executable(candidate_path, requested)

    discovered_path = shutil.which(requested)
    if discovered_path is None:
        raise _resolution_error(requested, "command tidak ditemukan melalui PATH")
    return _validate_executable(Path(discovered_path), requested)
