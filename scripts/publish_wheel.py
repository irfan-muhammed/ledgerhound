#!/usr/bin/env python
"""
scripts/publish_wheel.py
========================

Build the ledgerhound wheel and publish it to a MinIO-hosted PyPI-style
``simple/`` index, so pods can ``ledgerhound upgrade`` against it.

Layout produced in the wheels bucket (PEP 503 "simple repository"):

    simple/index.html                  -> links to each project
    simple/ledgerhound/index.html      -> links to every published wheel
    simple/ledgerhound/<pkg>-<ver>-*.whl

Point pip / `ledgerhound upgrade` at it via:

    export LEDGERHOUND_INDEX_URL="https://<minio-host>/<wheels-bucket>/simple"
    ledgerhound upgrade

(That requires the wheels bucket / simple prefix to be HTTP-readable by pods --
either an anonymous-download policy on the prefix or a fronting proxy. The
publish step here only writes the objects; serving is an infra setting.)

Usage:
    python scripts/publish_wheel.py --minio-config configs/minio.yaml
    python scripts/publish_wheel.py --no-build      # publish existing dist/*.whl

Credentials come from env (MINIO_ACCESS_KEY / MINIO_SECRET_KEY); never commit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from html import escape
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledgerhound.io import MinIOStore  # noqa: E402

PROJECT = "ledgerhound"
SIMPLE_PREFIX = "simple"
PROJ_PREFIX = f"{SIMPLE_PREFIX}/{PROJECT}"


def build_wheel(dist_dir: Path) -> None:
    """Build a wheel into dist/ via `python -m build --wheel`."""
    try:
        import build  # noqa: F401
    except ImportError:
        print("ERROR: the 'build' package is not installed.\n"
              "  pip install build   (or rerun with --no-build to publish an "
              "existing dist/*.whl)", file=sys.stderr)
        raise SystemExit(2)
    print("Building wheel...")
    subprocess.check_call([sys.executable, "-m", "build", "--wheel",
                           "--outdir", str(dist_dir)])


def find_wheels(dist_dir: Path) -> list[Path]:
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        print(f"ERROR: no wheels found in {dist_dir}", file=sys.stderr)
        raise SystemExit(1)
    return wheels


def _simple_index_html(title: str, links: list[tuple[str, str]]) -> bytes:
    """Render a minimal PEP 503 index page. links = [(href, text)]."""
    rows = "\n".join(f'    <a href="{escape(h)}">{escape(t)}</a><br/>'
                     for h, t in links)
    html = ("<!DOCTYPE html>\n<html><head><title>"
            f"{escape(title)}</title></head><body>\n{rows}\n"
            "</body></html>\n")
    return html.encode("utf-8")


def publish(store: MinIOStore, wheels: list[Path]) -> str:
    bucket_stream = "wheels"
    store.ensure_bucket(bucket_stream)

    # 1. upload each wheel under the project prefix
    for whl in wheels:
        key = f"{PROJ_PREFIX}/{whl.name}"
        store.put_file(bucket_stream, key, whl,
                       content_type="application/octet-stream")
        print(f"  uploaded {key}")

    # 2. rebuild the project index from ALL wheels now in the prefix
    all_keys = store.list_keys(bucket_stream, prefix=f"{PROJ_PREFIX}/")
    wheel_names = sorted(k.rsplit("/", 1)[-1] for k in all_keys
                         if k.endswith(".whl"))
    proj_html = _simple_index_html(
        f"Links for {PROJECT}",
        [(name, name) for name in wheel_names],
    )
    store.put_bytes(bucket_stream, f"{PROJ_PREFIX}/index.html", proj_html,
                    content_type="text/html")

    # 3. ensure the root index links to the project
    root_html = _simple_index_html("Simple index", [(f"{PROJECT}/", PROJECT)])
    store.put_bytes(bucket_stream, f"{SIMPLE_PREFIX}/index.html", root_html,
                    content_type="text/html")

    return f"{store.config.bucket(bucket_stream)}/{SIMPLE_PREFIX}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minio-config", default="configs/minio.yaml")
    ap.add_argument("--dist", default="dist", help="wheel output directory")
    ap.add_argument("--no-build", action="store_true",
                    help="skip build; publish existing dist/*.whl")
    args = ap.parse_args()

    dist_dir = Path(args.dist)
    if not args.no_build:
        dist_dir.mkdir(parents=True, exist_ok=True)
        build_wheel(dist_dir)
    wheels = find_wheels(dist_dir)

    store = MinIOStore.from_yaml(args.minio_config)
    if not store.config.endpoint:
        print("ERROR: MinIO endpoint not set in "
              f"{args.minio_config}", file=sys.stderr)
        raise SystemExit(1)

    index_path = publish(store, wheels)
    print("\nPublished. To consume from a pod:")
    print(f'  export LEDGERHOUND_INDEX_URL="https://<minio-host>/{index_path}"')
    print("  ledgerhound upgrade")


if __name__ == "__main__":
    main()
