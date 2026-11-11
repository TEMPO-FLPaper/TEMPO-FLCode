"""Reduced real-algorithm runs; never changes Reference presets.

Uses the original TrashNet data and ImageNet MobileNetV2, with smaller configurable
resolution/width, epochs and steps. Stops after a complete round then resumes.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from implementation.config.schema import RunConfig


def reduced(method):
    from implementation.cli.app import verification_config
    c = verification_config(method)
    c.output_root = str(ROOT / ".verification/runs")
    return c


def run(method, stop_resume=True):
    root = ROOT / ".verification"
    root.mkdir(exist_ok=True)
    c = reduced(method)
    config = root / f"{method}.verification.json"
    c.save(config)
    destination = Path(c.output_root) / f"{time.strftime('%Y%m%d_%H%M%S')}_{method}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    events = []

    def launch(resume=False):
        args = [
            sys.executable,
            "-u",
            "-m",
            "implementation.execution.worker",
            "--config",
            str(config),
            "--run-dir",
            str(destination),
        ] + (["--resume"] if resume else [])
        err = (root / f"{method}.stderr.log").open("a", encoding="utf-8")
        p = subprocess.Popen(
            args,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        for line in p.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            if event.get("type") in {"stage", "checkpoint", "error", "completed", "interrupted"}:
                print(
                    method,
                    event.get("type"),
                    event.get("stage", event.get("message", event.get("round", ""))),
                    flush=True,
                )
            if (
                stop_resume
                and not resume
                and event.get("type") == "checkpoint"
                and event.get("round") == 1
            ):
                (destination / "stop.request").write_text(
                    "Verification: stop at round boundary.", encoding="utf-8"
                )
        code = p.wait()
        err.close()
        if code not in (0, 130):
            raise RuntimeError(f"{method} worker failed ({code}); see {destination}/failure.txt")

    launch()
    state = json.loads((destination / "status.json").read_text())
    if stop_resume and state["state"] == "interrupted":
        launch(True)
    state = json.loads((destination / "status.json").read_text())
    assert state["state"] == "completed"
    results = json.loads((destination / "results.json").read_text())
    assert results["current_round"] == 2
    import zipfile

    archive_path = next(e["path"] for e in reversed(events) if e.get("kind") == "complete_run_archive")
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert "resolved_config.json" in archive.namelist()
        assert "03_checkpoints/latest.json" in archive.namelist()
    summary = {
        "method": method,
        "run_dir": str(destination),
        "status": state,
        "stop_resume": any(e.get("type") == "restored" for e in events),
        "event_types": sorted(set(e["type"] for e in events)),
        "general_rows": len(results["tables"]["general_accuracy_rows"]),
        "owned_rows": len(results["tables"]["owned_accuracy_rows"]),
        "archive": archive_path,
        "archive_integrity_passed": True,
    }
    (root / f"{method}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tempo", "ditto", "hfl", "flat", "all"], default="all")
    parser.add_argument("--no-stop-resume", action="store_true")
    args = parser.parse_args()
    for method in ["tempo", "ditto", "hfl", "flat"] if args.method == "all" else [args.method]:
        run(method, not args.no_stop_resume)
