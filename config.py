"""Runtime configuration loaded from flags and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b"
    workspace: Path = Path("workspace")
    state_dir: Path = Path(".omni")
    max_steps: int = 8
    timeout_seconds: float = 90.0
    api_key: str | None = field(default=None, repr=False)
    brave_search_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        model: str | None = None,
        workspace: str | None = None,
        state_dir: str | None = None,
        max_steps: int | None = None,
    ) -> "Settings":
        env_steps = os.environ.get("OMNI_MAX_STEPS", "8")
        try:
            configured_steps = int(env_steps) if max_steps is None else max_steps
        except ValueError as exc:
            raise ValueError("OMNI_MAX_STEPS must be an integer") from exc
        if not 1 <= configured_steps <= 12:
            raise ValueError("max_steps must be between 1 and 12")
        try:
            timeout_seconds = float(os.environ.get("OMNI_TIMEOUT_SECONDS", "90"))
        except ValueError as exc:
            raise ValueError("OMNI_TIMEOUT_SECONDS must be a number") from exc
        if not 1 <= timeout_seconds <= 600:
            raise ValueError("OMNI_TIMEOUT_SECONDS must be between 1 and 600")

        return cls(
            base_url=(base_url or os.environ.get("OMNI_BASE_URL") or "http://localhost:11434/v1").strip(),
            model=(model or os.environ.get("OMNI_MODEL") or "qwen2.5:7b").strip(),
            workspace=Path(workspace or os.environ.get("OMNI_WORKSPACE") or "workspace"),
            state_dir=Path(state_dir or os.environ.get("OMNI_STATE_DIR") or ".omni"),
            max_steps=configured_steps,
            timeout_seconds=timeout_seconds,
            api_key=os.environ.get("OMNI_API_KEY") or None,
            brave_search_api_key=os.environ.get("BRAVE_SEARCH_API_KEY") or None,
        )
