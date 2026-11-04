"""
ledgerhound.auto.patch
===================

Patch-candidate management.

When the agent diagnoses a real bug in a tracked source file, it calls
``PatchStore.propose()``.  This writes a ``.diff`` + ``.context.md`` +
``.test.py`` to ``ledgerhound_workspace/patch_candidates/``.

The human reviews and applies with:
    ledgerhound auto review-patches
    ledgerhound auto apply-patch <id>
    ledgerhound auto reject-patch <id>
"""

from __future__ import annotations

import difflib, hashlib, json, textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class PatchCandidate:
    id: str
    created_at: str
    target_file: str
    original_code: str
    patched_code: str
    reason: str
    repro_test: str = ""
    status: str = "pending"  # pending | applied | rejected

    def diff_text(self) -> str:
        orig_lines = self.original_code.splitlines(keepends=True)
        patched_lines = self.patched_code.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                orig_lines, patched_lines,
                fromfile=f"a/{self.target_file}",
                tofile=f"b/{self.target_file}",
            )
        )


class PatchStore:
    """Read/write patch candidates from a directory."""

    def __init__(self, patch_dir: Path):
        self._dir = patch_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def propose(
        self,
        target_file: str | Path,
        original_code: str,
        patched_code: str,
        reason: str,
        repro_test: str = "",
    ) -> PatchCandidate:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        stem = Path(target_file).stem
        uid = hashlib.sha1(f"{ts}{stem}{reason}".encode(),
                           usedforsecurity=False).hexdigest()[:8]
        pid = f"{ts}_{stem}_{uid}"

        candidate = PatchCandidate(
            id=pid,
            created_at=ts,
            target_file=str(target_file),
            original_code=original_code,
            patched_code=patched_code,
            reason=reason,
            repro_test=repro_test,
        )
        self._write(candidate)
        return candidate

    def list(self) -> list[PatchCandidate]:
        candidates = []
        for meta_file in sorted(self._dir.glob("*.meta.json")):
            candidates.append(self._load(meta_file))
        return candidates

    def get(self, patch_id: str) -> PatchCandidate | None:
        meta = self._dir / f"{patch_id}.meta.json"
        if not meta.exists():
            return None
        return self._load(meta)

    def apply(self, patch_id: str) -> bool:
        """Write patched content to the target file and mark as applied."""
        c = self.get(patch_id)
        if c is None:
            return False
        target = Path(c.target_file)
        if not target.exists():
            raise FileNotFoundError(f"Target file not found: {target}")
        target.write_text(c.patched_code, encoding="utf-8")
        c.status = "applied"
        self._write(c)
        return True

    def reject(self, patch_id: str, reason: str = "") -> bool:
        c = self.get(patch_id)
        if c is None:
            return False
        c.status = f"rejected: {reason}" if reason else "rejected"
        self._write(c)
        return True

    # ------------------------------------------------------------------ #
    def _write(self, c: PatchCandidate) -> None:
        base = self._dir / c.id
        # .diff
        (self._dir / f"{c.id}.diff").write_text(c.diff_text(), encoding="utf-8")
        # .context.md
        ctx = textwrap.dedent(f"""\
            # Patch {c.id}

            **Target:** `{c.target_file}`
            **Created:** {c.created_at}
            **Status:** {c.status}

            ## Reason
            {c.reason}

            ## Repro test
            ```python
            {c.repro_test or "# (none provided)"}
            ```
        """)
        (self._dir / f"{c.id}.context.md").write_text(ctx, encoding="utf-8")
        # .meta.json (includes full code for apply)
        meta = {
            "id": c.id,
            "created_at": c.created_at,
            "target_file": c.target_file,
            "original_code": c.original_code,
            "patched_code": c.patched_code,
            "reason": c.reason,
            "repro_test": c.repro_test,
            "status": c.status,
        }
        (self._dir / f"{c.id}.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def _load(self, meta_file: Path) -> PatchCandidate:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        return PatchCandidate(**data)
