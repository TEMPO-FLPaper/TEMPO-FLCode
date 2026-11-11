"""Environment and source identities; no terminal presentation imports."""

from __future__ import annotations
import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess
import sys


def source_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    for path in sorted((root / "config").rglob("*.json")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment():
    import tensorflow as tf

    packages = {}
    for p in (
        "tensorflow",
        "tensorflow-intel",
        "keras",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "Pillow",
        "matplotlib",
        "openpyxl",
        "rich",
    ):
        try:
            packages[p] = version(p)
        except PackageNotFoundError:
            pass
    try:
        git = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.TimeoutExpired):
        git = None
    refs = Path(__file__).resolve().parents[1] / "config/reference_sources.json"
    import json

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "devices": [
            {"name": d.name, "type": d.device_type} for d in tf.config.list_physical_devices()
        ],
        "git_commit": git,
        "implementation_sha256": source_identity(),
        "references": json.loads(refs.read_text(encoding="utf-8")),
        "determinism": "Reference seeds and deterministic tf.data. Cross-platform bitwise equality is not guaranteed.",
    }
