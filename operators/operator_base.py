"""Shared protocol and result type for Ego-X pipeline operators.

Each operator exposes a lightweight adapter in ``op_impl.py`` so the
pipeline can orchestrate heterogeneous modules through one interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class OperatorResult:
    """Normalized result payload returned by every operator."""

    status: str  # "ok" | "error" | "skipped" | "pending"
    operator: str  # operator name, e.g. "privacy_blur"
    output_files: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@runtime_checkable
class Operator(Protocol):
    """Interface that every pipeline operator adapter must satisfy."""

    name: str

    def run(self, episode_dir: Path, **kwargs: Any) -> OperatorResult:
        """Process one work directory and return an ``OperatorResult``."""
        ...
