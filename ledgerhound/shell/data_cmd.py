"""
ledgerhound.shell.data_cmd
=========================

`/local --path <path>` and `/s3 --path <s3://bucket/key>`: load data with
the source stated explicitly. Both call the same loader the model's
load_data tool uses (ConversationalAnalyst.load_from_command), so the
shell and the model never disagree about what is loaded or where outputs
go. `/paths` shows the current run's path record.

    /local --path /data/upi/july.parquet
    /s3 --path s3://sentinel-data/data/july/            (a folder of Parquet parts)
    /s3 --path data/july/txn.parquet                   (default bucket)
    /s3 --path s3://sentinel-data/data/july/ --as-one     (plain files that ARE one dataset)
    /local --path july.parquet --name trino:hive.payments.txn_sample
                                   (a Trino export: share the table's team knowledge)
"""

from __future__ import annotations

import os
import shlex

USAGE = {
    "local": "/local --path <file or folder>  [--as-one] [--name <dataset name>]",
    "s3": "/s3 --path s3://<bucket>/<key or folder/>  [--as-one] [--name <dataset name>]",
}


def parse_load_args(text: str) -> tuple[str, bool, str]:
    """(path, as_one, name). Accepts `--path X`, `--path=X` or a bare X,
    plus `--as-one` and `--name N` (the dataset's shared name, e.g.
    `trino:hive.payments.txn_sample`). Raises ValueError with a usage hint."""
    try:
        tokens = shlex.split(text or "", posix=(os.name != "nt"))
    except ValueError as e:
        raise ValueError(f"could not read the arguments ({e}).") from e
    tokens = [t.strip('"').strip("'") for t in tokens]
    path, as_one, name, i = "", False, "", 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("--as-one", "--as_one", "--one"):
            as_one = True
        elif t == "--path":
            if i + 1 >= len(tokens):
                raise ValueError("--path needs a value.")
            path = tokens[i + 1]
            i += 1
        elif t.startswith("--path="):
            path = t.split("=", 1)[1]
        elif t == "--name":
            if i + 1 >= len(tokens):
                raise ValueError("--name needs a value.")
            name = tokens[i + 1]
            i += 1
        elif t.startswith("--name="):
            name = t.split("=", 1)[1]
        elif t.startswith("--"):
            raise ValueError(f"unknown option {t}.")
        elif not path:
            path = t
        else:
            raise ValueError(f"more than one path given ({path!r}, {t!r}) -- quote a path with spaces.")
        i += 1
    if not path:
        raise ValueError("no path given.")
    return path, as_one, name


def model_note(kind: str, path: str, loader_output: str) -> str:
    """What the model is told after a command-driven load."""
    return (f"[I loaded data myself with /{kind} --path {path} -- it is ALREADY "
            f"loaded; do not call load_data for it again.]\n"
            f"Loader output:\n{loader_output}\n\n"
            "Acknowledge in one or two lines what is loaded and where outputs "
            "will be saved. If we already agreed a goal, continue it on this data; "
            "otherwise ask what I want to find.")
