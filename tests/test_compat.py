"""Setups made under the old name (Sentinel) keep working after the rename."""

from __future__ import annotations

from ledgerhound import _compat
from ledgerhound.project.project import Project
from ledgerhound.project.store import ProjectStore


def test_legacy_env_vars_are_adopted():
    env = {"SENTINEL_TRINO_HOST": "trino.old", "SENTINEL_USER": "irfan",
           "LEDGERHOUND_USER": "new-wins", "PATH": "/bin"}
    adopted = _compat.adopt_legacy_env(env)
    assert env["LEDGERHOUND_TRINO_HOST"] == "trino.old"
    assert env["LEDGERHOUND_USER"] == "new-wins"          # never overwritten
    assert adopted == ["SENTINEL_TRINO_HOST"]


def test_legacy_project_folder_is_migrated_on_open(tmp_path):
    old = tmp_path / "proj" / ".sentinel"
    old.mkdir(parents=True)
    (old / "project.json").write_text('{"name": "proj"}', encoding="utf-8")
    (old / "SENTINEL.md").write_text("team notes", encoding="utf-8")
    p = Project(tmp_path / "proj")
    assert p.exists() and p.config().name == "proj"
    assert p.memory_path.read_text(encoding="utf-8") == "team notes"
    assert not old.exists()


def test_legacy_projects_are_listed(tmp_path):
    old = tmp_path / "a" / ".sentinel"
    old.mkdir(parents=True)
    (old / "project.json").write_text('{"name": "a"}', encoding="utf-8")
    assert [p.root.name for p in ProjectStore(tmp_path).list()] == ["a"]


def test_migration_never_overwrites(tmp_path):
    (tmp_path / ".sentinel").mkdir()
    (tmp_path / ".sentinel" / "x").write_text("old", encoding="utf-8")
    (tmp_path / ".ledgerhound").mkdir()
    (tmp_path / ".ledgerhound" / "x").write_text("new", encoding="utf-8")
    _compat.migrate_dotdir(tmp_path)
    assert (tmp_path / ".ledgerhound" / "x").read_text(encoding="utf-8") == "new"
    assert (tmp_path / ".sentinel" / "x").exists()


def test_cwd_workspace_is_migrated(tmp_path):        # cwd is tmp_path (conftest)
    (tmp_path / "sentinel_workspace" / "reports").mkdir(parents=True)
    _compat.migrate_cwd()
    assert (tmp_path / "ledgerhound_workspace" / "reports").is_dir()
