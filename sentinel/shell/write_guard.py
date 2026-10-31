"""
sentinel.shell.write_guard
============================

V3-063: the sandbox write boundary, for real this time.

V3-049's fix (bracket exec() with a chdir into scratch) only protects a
BARE RELATIVE write -- `df.to_csv("out.csv")`. It does nothing for an
ABSOLUTE path, and does nothing for the extremely common case of code
that builds a path from the loaded data file's own directory, e.g.::

    out = os.path.join(os.path.dirname(data_path), "features_v2.csv")
    df.to_csv(out)

which is exactly what got reported in production: feature tables landing
in the ORIGINAL DATA FOLDER instead of the workspace.

This module patches the actual write METHODS so every write is checked
regardless of how the destination path was built. A write outside the
allowed roots is not blocked (a crashed turn helps no one) -- it is
TRANSPARENTLY REDIRECTED into the features directory, with a printed
notice, so the model can see where its file went.

2026-09-19 (feature saves "not landing in the right folder"), probed in a
real kernel:

* `df.to_parquet("features/acct.parquet")` -- the most natural thing a
  model writes -- crashed: relative paths resolve under scratch, where no
  `features/` folder exists. A relative path starting with `features/` now
  means the features folder.
* `features_dir / "account" / "x.parquet"` crashed: pandas does not create
  folders. Every guarded write now creates its parent folder.
* `pyarrow.parquet.write_table(...)` and `Path.write_text(...)` were not
  patched at all and wrote straight into the raw-data folder. They are now
  guarded, as are pyarrow's write_to_dataset/write_csv and dask's
  to_parquet/to_csv.
* Two different outside paths with the same file name
  (`/data/a/feat.parquet`, `/data/b/feat.parquet`) were both redirected to
  `features/feat.parquet` -- the second silently overwrote the first. A
  redirect now never lands on a file it did not itself create for that
  same original path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

# Relative paths that start with one of these mean "the features folder".
_FEATURE_PREFIXES = ("features", "features_dir")

# original path -> redirected path, per redirect folder. Module level so a
# later cell writing the same original path again (an update) lands on the
# same redirected file, even though exec mode re-installs the guard per cell.
_REDIRECTS: dict[str, dict[str, str]] = {}


def _is_inside(path: Path, roots: list) -> bool:
    try:
        rp = path.resolve()
    except Exception:
        return False
    for root in roots:
        try:
            rp.relative_to(Path(root).resolve())
            return True
        except ValueError:
            continue
    return False


def _redirect_target(original: Path, redirect_dir: Path) -> Path:
    """Where an outside write goes. Stable per original path; never an
    existing file that some other path produced."""
    key = str(redirect_dir.resolve())
    seen = _REDIRECTS.setdefault(key, {})
    orig = str(original)
    if orig in seen:
        return Path(seen[orig])
    taken = set(seen.values())
    name = original.name or "output"
    candidates = [redirect_dir / name]
    parent = original.parent.name
    if parent:
        candidates.append(redirect_dir / f"{parent}__{name}")
    stem, suffix = Path(name).stem, Path(name).suffix
    candidates += [redirect_dir / f"{stem}_{i}{suffix}" for i in range(2, 1000)]
    for c in candidates:
        if str(c) not in taken and not c.exists():
            seen[orig] = str(c)
            return c
    seen[orig] = str(candidates[-1])
    return candidates[-1]


def resolve_write_path(target, allowed_roots: list, redirect_dir,
                       notice: Callable[[str, str], None] | None = None):
    """The path a write should really go to, with its parent folder created.

    Returns `target` unchanged when it is not a filesystem path (a buffer,
    an s3:// URL, ...). Used by every patched writer and by the kernel's
    open() override, so all writes follow one rule."""
    if not isinstance(target, (str, Path)):
        return target
    s = str(target)
    if "://" in s:                      # s3://, gs://, http:// -- not ours to judge
        return target
    p = Path(s)
    redirect_dir = Path(redirect_dir)
    if not p.is_absolute() and p.parts and p.parts[0] in _FEATURE_PREFIXES:
        p = redirect_dir.joinpath(*p.parts[1:]) if len(p.parts) > 1 else redirect_dir
    elif not _is_inside(p, allowed_roots):
        redirected = _redirect_target(p, redirect_dir)
        if notice is not None:
            notice(s, str(redirected))
        p = redirected
    try:
        # A partitioned write (to_parquet(dir, partition_cols=...)) names a
        # folder; creating the parent is right for both files and folders.
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p if isinstance(target, Path) or p != Path(s) else target


def link_features_into(scratch_dir, features_dir) -> bool:
    """Make `<scratch>/features` point at the real features folder, so a
    relative `features/...` READ finds what a relative write saved (code
    runs with scratch as its cwd). Symlink on Linux; directory junction on
    Windows, which needs no special privilege. Best-effort: returns False
    if neither works, and writes still map correctly via
    resolve_write_path."""
    scratch_dir, features_dir = Path(scratch_dir), Path(features_dir)
    link = scratch_dir / "features"
    try:
        if link.resolve() == features_dir.resolve():
            return True                          # already linked, or the same folder
        if link.exists() or link.is_symlink():
            return False                         # a real folder is there -- leave it
        scratch_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(features_dir.resolve(), target_is_directory=True)
            return True
        except OSError:
            import os
            if os.name != "nt":
                return False
            import _winapi  # type: ignore
            _winapi.CreateJunction(str(features_dir.resolve()), str(link))
            return True
    except Exception:  # noqa: BLE001
        return False


_PATCHED_ATTR = "_sentinel_write_guard_original"
_PATH_KWARGS = ("path_or_buf", "path", "fname", "buf", "where", "root_path",
                "output_file", "file", "excel_writer", "filename")


def _guard_call(orig: Callable, arg_index: int, roots_getter, redirect_getter,
                notice) -> Callable:
    """Wrap a writer whose destination is positional argument `arg_index`
    (counting `self` for methods) or one of the usual keyword names."""

    def guarded(*args, **kwargs):
        args = list(args)
        if len(args) > arg_index:
            args[arg_index] = resolve_write_path(
                args[arg_index], roots_getter(), redirect_getter(), notice)
        else:
            for name in _PATH_KWARGS:
                if name in kwargs:
                    kwargs[name] = resolve_write_path(
                        kwargs[name], roots_getter(), redirect_getter(), notice)
                    break
        return orig(*args, **kwargs)

    setattr(guarded, _PATCHED_ATTR, True)
    return guarded


def install(allowed_roots_getter: Callable[[], list],
            redirect_dir_getter: Callable[[], "Path | str"],
            notice: Callable[[str, str], None]) -> Callable[[], None]:
    """Patch pandas/cuDF/pyarrow/dask/pathlib/matplotlib writers. Returns an
    `uninstall()` callable. Best-effort throughout -- a missing or
    incompatible library must never break sandbox startup."""
    patched: list[tuple[object, str, Callable]] = []

    def _patch(owner, name: str, arg_index: int) -> None:
        orig = getattr(owner, name, None)
        if orig is None or getattr(orig, _PATCHED_ATTR, False):
            return
        wrapped = _guard_call(orig, arg_index, allowed_roots_getter,
                              redirect_dir_getter, notice)
        try:
            setattr(owner, name, wrapped)
            patched.append((owner, name, orig))
        except Exception:  # noqa: BLE001 -- some proxy classes are read-only
            pass

    # Methods: destination is the first argument after `self`.
    try:
        import pandas as pd
        for m in ("to_csv", "to_parquet", "to_excel", "to_json", "to_pickle",
                  "to_feather"):
            _patch(pd.DataFrame, m, 1)
        for m in ("to_csv", "to_parquet", "to_json", "to_pickle"):
            _patch(pd.Series, m, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import cudf  # type: ignore
        for m in ("to_csv", "to_parquet", "to_json"):
            _patch(cudf.DataFrame, m, 1)
    except Exception:  # noqa: BLE001 -- cuDF not installed / not on a GPU box
        pass
    try:
        import dask.dataframe as dd  # type: ignore
        for m in ("to_parquet", "to_csv"):
            _patch(dd.DataFrame, m, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import pathlib
        _patch(pathlib.Path, "write_text", 0)    # the Path itself is the target
        _patch(pathlib.Path, "write_bytes", 0)
    except Exception:  # noqa: BLE001
        pass

    # Functions: destination is the second argument (data first).
    try:
        import pyarrow.parquet as pq  # type: ignore
        _patch(pq, "write_table", 1)
        _patch(pq, "write_to_dataset", 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import pyarrow.csv as pacsv  # type: ignore
        _patch(pacsv, "write_csv", 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import pyarrow.feather as pafeather  # type: ignore
        _patch(pafeather, "write_feather", 1)
    except Exception:  # noqa: BLE001
        pass

    # plt.savefig(fname, ...) and Figure.savefig(self, fname, ...).
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        _patch(plt, "savefig", 0)
        _patch(Figure, "savefig", 1)
    except Exception:  # noqa: BLE001
        pass

    def uninstall() -> None:
        for owner, name, orig in reversed(patched):
            try:
                setattr(owner, name, orig)
            except Exception:  # noqa: BLE001
                pass

    return uninstall
