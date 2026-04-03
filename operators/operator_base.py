"""Minimal protocol and result type for Ego-X operators.

Every operator adapter implements the ``Operator`` protocol so that the
pipeline can orchestrate them uniformly.  Existing standalone scripts are
**not** modified -- operator implementations live alongside them (``op_impl.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class OperatorResult:
    """Uniform return type for all operators."""

    status: str  # "ok" | "error" | "skipped"
    operator: str  # operator name, e.g. "privacy_blur"
    output_files: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@runtime_checkable
class Operator(Protocol):
    """Protocol that every operator adapter must satisfy."""

    name: str

    def run(self, episode_dir: Path, **kwargs: Any) -> OperatorResult:
        """Process one episode directory. Returns OperatorResult."""
        ...
