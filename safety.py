"""Human approval gates for local side effects."""

from __future__ import annotations

import sys
from collections.abc import Callable


class WriteApproval:
    """Approval policy for writes inside the configured workspace.

    Writes are denied by default in non-interactive use. ``auto_approve`` is an
    explicit opt-in intended for disposable workspaces and automation.
    """

    def __init__(
        self,
        *,
        auto_approve: bool = False,
        confirm_fn: Callable[[str, bool], bool] | None = None,
    ) -> None:
        self.auto_approve = auto_approve
        self.confirm_fn = confirm_fn

    def approve_write(self, relative_path: str, overwriting: bool) -> bool:
        if self.auto_approve:
            return True
        if self.confirm_fn is not None:
            return bool(self.confirm_fn(relative_path, overwriting))
        if not sys.stdin.isatty():
            return False

        action = "overwrite" if overwriting else "create"
        answer = input(
            f"Allow the agent to {action} workspace file '{relative_path}'? [y/N] "
        )
        return answer.strip().casefold() in {"y", "yes"}
