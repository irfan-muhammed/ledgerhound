"""
sentinel.session.librarian (deprecated shim)
=============================================

This module was renamed to ``sentinel.session.file_index`` and the class
``FileLibrarian`` became ``FileIndex``, to remove the name collision with the
memory curator ``sentinel.memory.librarian.Librarian`` (the moat gateway).

Importing from here still works but is deprecated. Update imports to::

    from sentinel.session.file_index import FileIndex
"""

from __future__ import annotations

from sentinel.session.file_index import (  # noqa: F401
    Category,
    FileEntry,
    FileIndex,
)

# Backward-compatible alias. Prefer FileIndex in new code.
FileLibrarian = FileIndex

__all__ = ["Category", "FileEntry", "FileIndex", "FileLibrarian"]
