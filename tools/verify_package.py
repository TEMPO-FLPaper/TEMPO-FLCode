"""Install the built wheel into an isolated tree and exercise it from another cwd."""

from pathlib import Path
import json
import os
import subprocess
import sys
import uuid
import zipfile

# Development-only ACL adaptation from the audited wheel builder.
import build_release

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".verification"
wheel = max((OUTPUT / "dist").glob("tempo_fl-*.whl"), key=lambda p: p.stat().st_mtime)
target = OUTPUT / ("isolated_install_" + uuid.uuid4().hex[:8])
os.environ["PIP_CACHE_DIR"] = str(ROOT / ".cache/pip")
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    assert not any("/gui/" in n or "PySide" in n for n in names)
    assert sum(n.endswith("/TEMPO_FL.pdf") for n in names) == 1
    assert "implementation/config/README.md" in names
    for method in ("tempo_fl", "ditto", "hfl", "flat"):
        for preset in ("reference", "verification"):
            assert f"implementation/config/{method}/{preset}.json" in names
from pip._internal.cli.main import main as pip_main

code = pip_main(
    ["install", "--no-index", "--no-deps", "--no-compile", "--target", str(target), str(wheel)]
)
assert code == 0
env = dict(os.environ, PYTHONPATH=str(target), TEMPO_FL_HOME=str(target))
commands = [
    ["--plain", "methods"],
    ["doctor"],
    ["config", "help", "ditto_lambda", "--method", "ditto"],
]
records = []
for args in commands:
    p = subprocess.run(
        [sys.executable, "-m", "implementation", *args],
        cwd=OUTPUT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert p.returncode == 0, p.stderr
    records.append({"args": args, "returncode": p.returncode, "stdout": p.stdout})
code = """
import sys, pathlib
import implementation
from implementation.config.schema import RunConfig, project_root
from implementation.cli.app import open_paper, verification_config
import os
target = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(implementation.__file__).is_relative_to(target)
assert project_root() == target
opened = []
if os.name == 'nt':
    os.startfile = opened.append
    assert open_paper().is_relative_to(target)
    assert pathlib.Path(opened[0]).is_file()
for method in ('tempo','ditto','hfl','flat'):
    c = RunConfig.defaults(method); c.validate()
    assert pathlib.Path(c.output_root).is_relative_to(target)
    v = verification_config(method); v.validate()
    assert v.protocol_label == 'verification'
assert 'tensorflow' not in sys.modules
assert not (target / 'references').exists()
print('Isolated wheel resources, four presets, lightweight imports and local paths verified.')
"""
p = subprocess.run(
    [sys.executable, "-c", code, str(target)],
    cwd=OUTPUT,
    env=env,
    capture_output=True,
    text=True,
    encoding="utf-8",
    timeout=30,
)
assert p.returncode == 0, p.stderr
records.append({"resource_check": p.stdout, "returncode": p.returncode})
(OUTPUT / "package_check.json").write_text(
    json.dumps({"wheel": str(wheel), "target": str(target), "records": records}, indent=2)
)
print(p.stdout)
