"""Cache the same public Keras ImageNet object through official GCS transports."""

from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


def ensure_imagenet_weights(settings, emit, stopped):
    size = settings["image_size"]
    rows = size if size in {96, 128, 160, 192, 224} else 224
    name = f"mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{float(settings['mobilenet_alpha'])}_{rows}_no_top.h5"
    home = Path(os.environ.get("KERAS_HOME", str(Path.home() / ".keras")))
    directory = home / "models"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    object_name = "keras-applications/mobilenet_v2/" + name
    primary = "https://storage.googleapis.com/tensorflow/" + object_name
    api = (
        "https://www.googleapis.com/download/storage/v1/b/tensorflow/o/"
        + urllib.parse.quote(object_name, safe="")
        + "?alt=media"
    )
    if not target.is_file():
        emit({"type": "stage", "stage": "Downloading ImageNet MobileNetV2 weights", "total": None})
        url = primary
        try:
            response = urllib.request.urlopen(primary, timeout=180)
        except urllib.error.HTTPError as exc:
            if exc.code != 403:
                raise
            # Same bucket and object: no different pretrained model is substituted.
            url = api
            response = urllib.request.urlopen(api, timeout=180)
            emit(
                {
                    "type": "log",
                    "message": "Primary weight transport returned 403; using the official GCS download API for the identical bucket/object.",
                }
            )
        temporary = target.with_suffix(".download")
        md5 = hashlib.md5()
        sha = hashlib.sha256()
        current = 0
        with response, temporary.open("wb") as handle:
            expected = int(response.headers.get("content-length", 0)) or None
            hash_header = response.headers.get("x-goog-hash", "")
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                if stopped():
                    raise InterruptedError(
                        "Stopped during ImageNet weight acquisition; restart will retry."
                    )
                handle.write(chunk)
                md5.update(chunk)
                sha.update(chunk)
                current += len(chunk)
                emit({"type": "download", "current": current, "total": expected})
        if expected is not None and current != expected:
            raise IOError("ImageNet weight download was incomplete.")
        expected_md5 = next(
            (p.strip()[4:] for p in hash_header.split(",") if p.strip().startswith("md5=")), None
        )
        if expected_md5 and base64.b64encode(md5.digest()).decode() != expected_md5:
            raise IOError("ImageNet object MD5 integrity check failed.")
        with temporary.open("rb") as handle:
            if handle.read(8) != b"\x89HDF\r\n\x1a\n":
                raise IOError("ImageNet download is not the expected HDF5 file.")
        os.replace(temporary, target)
        target.with_suffix(".provenance.json").write_text(
            json.dumps(
                {
                    "bucket": "tensorflow",
                    "object": object_name,
                    "transport": url,
                    "sha256": sha.hexdigest(),
                    "verified_gcs_md5": expected_md5,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "initialization": "imagenet",
        "bucket": "tensorflow",
        "object": object_name,
        "file": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
