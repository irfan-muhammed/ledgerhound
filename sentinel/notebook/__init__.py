"""
sentinel.notebook
=================

Notebook-as-journal writer.

A read-only ``.ipynb`` artifact rendered from an EDA agent trace.  The
journal is the *primary reviewable output* of an interactive session: it
preserves the chain of reasoning (thinking -> KB lookup -> code -> finding)
in a form that opens cleanly in Jupyter, JupyterLab, Colab, or VSCode.

Why a notebook (not Markdown)?
------------------------------
* Code cells display the exact Python the agent ran, with monospace
  formatting and syntax highlighting.
* Image outputs (charts) render inline.
* The format is industry-standard -- a senior reviewer can scroll through
  on any platform without installing Sentinel.
* Cells are static -- no live ipywidgets needed (Phase-1 environment lock).

Why not live ipywidgets?
------------------------
The Kubeflow Arcon environment doesn't render widget outputs reliably.
We separated the human-in-the-loop interaction (CLI) from the audit
artifact (notebook) so each can be optimised for its surface.
"""

from .journal import JournalWriter, write_journal

__all__ = ["JournalWriter", "write_journal"]
