"""Build a wheel with project-local temporary files, including managed Windows hosts.

The host used for verification rejects tempfile's 0700 directories. Inherited
ACL creation is confined to this development process; production is unaffected.
"""

from pathlib import Path
import os
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".verification/dist"
TEMP = ROOT / ".verification/packaging"
OUTPUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(TEMP)
tempfile.tempdir = str(TEMP)


def inherited_acl_temp(suffix=None, prefix=None, dir=None):
    path = Path(dir or TEMP) / f"{prefix or 'tmp'}{uuid.uuid4().hex}{suffix or ''}"
    path.mkdir()
    return str(path)


tempfile.mkdtemp = inherited_acl_temp

if __name__ == "__main__":
    os.chdir(ROOT)
    from setuptools.build_meta import build_wheel

    print(OUTPUT / build_wheel(str(OUTPUT)))
