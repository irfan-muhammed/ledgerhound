"""
ledgerhound.compute.router
=======================

ComputeRouter -- single source of truth for "which backend should we use"
once `ledgerhound doctor` has decided.

The router reads `compute_profile.json` (per-workspace or per-machine)
exactly once and answers two questions for the rest of the codebase:

  * `preferred_dataframe_backend()` -> "pandas" | "cudf"
  * `preferred_vector_backend()`    -> "numpy"  | "faiss"  | "cuvs"

If no profile exists yet, the router returns conservative defaults
("pandas", "numpy") without raising. Callers should treat the router
as advisory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from ledgerhound.doctor.profile import ComputeProfile, default_profile_path, load_profile

log = logging.getLogger(__name__)


_CONSERVATIVE_DF = "pandas"
_CONSERVATIVE_VEC = "numpy"


class ComputeRouter:
    """
    Resolve a compute_profile.json and answer backend questions.

    Parameters
    ----------
    workspace_root : Optional[Path]
        If given, look for compute_profile.json inside this workspace.
        Otherwise the global ~/.ledgerhound/compute_profile.json is used.
    profile        : Optional[ComputeProfile]
        Pre-loaded profile. Useful for tests; overrides workspace lookup.
    """

    def __init__(
        self,
        *,
        workspace_root: Optional[Union[str, Path]] = None,
        profile: Optional[ComputeProfile] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root) if workspace_root is not None else None
        if profile is not None:
            self._profile: Optional[ComputeProfile] = profile
        else:
            path = default_profile_path(self.workspace_root)
            self._profile = load_profile(path)

    @property
    def profile(self) -> Optional[ComputeProfile]:
        return self._profile

    @property
    def has_profile(self) -> bool:
        return self._profile is not None

    def preferred_dataframe_backend(self) -> str:
        if self._profile is None:
            return _CONSERVATIVE_DF
        return self._profile.preferred_dataframe or _CONSERVATIVE_DF

    def preferred_vector_backend(self) -> str:
        if self._profile is None:
            return _CONSERVATIVE_VEC
        return self._profile.preferred_vector or _CONSERVATIVE_VEC

    def explain(self) -> str:
        """One-line description of the active profile, for op_log/notes."""
        if self._profile is None:
            return (
                f"no compute_profile.json found -- defaults "
                f"(df={_CONSERVATIVE_DF}, vec={_CONSERVATIVE_VEC}). "
                f"Run `ledgerhound doctor` to benchmark this machine."
            )
        return (
            f"compute_profile: df={self._profile.preferred_dataframe}, "
            f"vec={self._profile.preferred_vector} "
            f"(generated_at={self._profile.generated_at:.0f})"
        )


# -- Process-level cache ----------------------------------------------------
#
# Most code will use the global router via `default_router()`. Tests and
# isolated tools may construct their own ComputeRouter instances.

_DEFAULT_ROUTER: Optional[ComputeRouter] = None


def default_router(
    *,
    workspace_root: Optional[Union[str, Path]] = None,
    refresh: bool = False,
) -> ComputeRouter:
    """Return the process-wide router. Pass refresh=True to re-read disk."""
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None or refresh:
        _DEFAULT_ROUTER = ComputeRouter(workspace_root=workspace_root)
    return _DEFAULT_ROUTER


def reset_default_router() -> None:
    """Test helper."""
    global _DEFAULT_ROUTER
    _DEFAULT_ROUTER = None
