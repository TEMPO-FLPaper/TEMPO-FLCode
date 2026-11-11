"""Bounded, restartable download of the reference TrashNet archive."""

from __future__ import annotations
import os
from pathlib import Path
import zipfile


def acquire(cache: Path, url: str, emit, stopped) -> Path:
    import requests

    expected = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
    cache.mkdir(parents=True, exist_ok=True)

    def valid(p):
        return p.is_dir() and expected.issubset({x.name.lower() for x in p.iterdir() if x.is_dir()})

    for p in [cache / "dataset-resized"] + list(cache.rglob("*")):
        if valid(p):
            return p
    archive = cache / "dataset-resized.zip"
    if archive.exists():
        try:
            with zipfile.ZipFile(archive) as z:
                bad = z.testzip() is not None
        except zipfile.BadZipFile:
            bad = True
        if bad:
            archive.unlink()
    if not archive.exists():
        emit({"type": "stage", "stage": "Downloading TrashNet", "total": None})
        temporary = archive.with_suffix(".download")
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or None
            current = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if stopped():
                        raise InterruptedError(
                            "Stopped during download. Restart will retry the archive."
                        )
                    if chunk:
                        handle.write(chunk)
                        current += len(chunk)
                        emit({"type": "download", "current": current, "total": total})
        os.replace(temporary, archive)
    emit({"type": "stage", "stage": "Extracting TrashNet", "total": None})
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if stopped():
                raise InterruptedError("Stopped during extraction. Restart will retry.")
            if not (cache / member.filename).resolve().is_relative_to(cache.resolve()):
                raise ValueError("Unsafe archive path.")
            z.extract(member, cache)
    for p in [cache] + list(cache.rglob("*")):
        if valid(p):
            return p
    raise FileNotFoundError(
        "Archive did not contain the six TrashNet class folders. Select a local copy."
    )
