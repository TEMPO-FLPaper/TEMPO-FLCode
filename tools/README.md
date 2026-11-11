# Maintainer tools

Run these commands from the repository root using the project Python environment. The extraction audit and pytest suite require the local `tests/` and `references/` directories, which Git ignores; they are not available in a fresh clone. The reduced experiment, wheel and export checks do not require those directories. Install development dependencies with `python -m pip install -e ".[dev]"`.

| Tool | Purpose | Command |
| --- | --- | --- |
| `audit_extraction.py` | Compares normalized production function bodies with original notebook definitions using `tests/reference_index.json`. Does not run notebook training. | `python tools/audit_extraction.py` |
| `verification_runs.py` | Runs real methods with reduced presets, stops after round 1, resumes committed state and verifies completion, results and archive integrity. Downloads original data/weights if absent. | `python tools/verification_runs.py --method all` |
| `build_release.py` | Builds a wheel with project-local temporary files. Its inherited-ACL directory adaptation is confined to this development process on managed Windows hosts. | `python tools/build_release.py` |
| `verify_package.py` | Installs the latest built wheel into an isolated directory and checks commands, the single paper resource, presets and lightweight imports from another working directory. Run the builder first. | `python tools/verify_package.py` |
| `verify_export.py` | Exercises the existing Flat FedAvg TFLite export using a trusted completed checkpoint, invokes the interpreter and checks output tensors. This is not experiment resume. | `python tools/verify_export.py --run-dir "<COMPLETED_FLAT_RUN>"` |

`verification_runs.py --method tempo` selects one method; other keys are `ditto`, `hfl`, `flat`. `--no-stop-resume` checks continuous execution instead. The default verifies cooperative stop/resume.

Outputs, reduced runs, temporary installs, exports and wheels stay under `.verification/`, which Git ignores. These disposable outputs can consume substantial disk space. Keep a trusted completed run until desired export checks finish; checkpoint deserialization is intended for locally generated research state.

```powershell
python -m pytest -q
python -m ruff check implementation tests tools
```

Passing an audit or reduced run does not establish full 100-round reproduction. Tools do not rewrite Reference presets or algorithms. User instructions, scientific context and limitations are in the [main README](../README.md).
