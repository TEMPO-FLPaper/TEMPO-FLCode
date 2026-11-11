"""Worker entry point. Standard output is a structured JSON-lines event stream."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import signal
import traceback
import time
from implementation.config.schema import RunConfig
from implementation.persistence.checkpoints import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="TEMPO-FL experiment worker, normally launched by the terminal application."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = RunConfig.load(args.config)
    root = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else Path(config.output_root)
        / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%f}_{config.method}"
    )
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        parser.error("Resume directory does not exist.")
    event_file = (root / "events.jsonl").open("a", encoding="utf-8", buffering=1)

    def emit(event):
        line = json.dumps(
            {"time": datetime.now(timezone.utc).isoformat(), **event},
            ensure_ascii=True,
            allow_nan=False,
        )
        event_file.write(line + "\n")
        print(line, flush=True)

    lock = root / "worker.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        emit(
            {
                "type": "error",
                "message": "Worker lock exists. Confirm the prior process exited before removing worker.lock.",
            }
        )
        return 2
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    experiment = None
    active_start = time.monotonic()
    elapsed_before = 0.0
    if args.resume and (root / "status.json").is_file():
        elapsed_before = float(json.loads((root / "status.json").read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
    from implementation.config.schema import project_root
    from implementation.execution.lease import WorkerLease
    lease = WorkerLease(project_root() / ".cache/active_worker.json", root)

    def safe_signal(signum, frame):
        (root / "stop.request").write_text("Cooperative stop requested by signal.\n", encoding="utf-8")

    signal.signal(signal.SIGINT, safe_signal)
    signal.signal(signal.SIGTERM, safe_signal)
    try:
        lease.acquire()
        if args.resume:
            original = RunConfig.load(root / "resolved_config.json")
            if original.fingerprint() != config.fingerprint():
                raise ValueError("Resume settings differ from the saved run.")
        else:
            config.save(root / "resolved_config.json")
        (root / "stop.request").unlink(missing_ok=True)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ["PYTHONHASHSEED"] = str(config.settings["seed"])
        os.environ.setdefault("KERAS_HOME", str(project_root() / ".cache/keras"))
        emit(
            {
                "type": "started",
                "run_dir": str(root),
                "method": config.method,
                "protocol_label": config.protocol_label,
                "elapsed_before": elapsed_before,
            }
        )
        atomic_json(
            root / "status.json",
            {
                "state": "running",
                "pid": os.getpid(),
                "method": config.method,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        emit({"type": "stage", "stage": "Initializing TensorFlow in worker process", "total": None})
        from implementation.execution.provenance import environment

        env = environment()
        atomic_json(root / ("resume_environment.json" if args.resume else "environment.json"), env)
        module = importlib.import_module("implementation.methods." + ("tempo_fl" if config.method == "tempo" else config.method))
        experiment = module.Experiment(config, root, emit, env)
        completed = experiment.run(resume=args.resume)
        status = {
            "state": "completed",
            "round": completed,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_before + time.monotonic() - active_start,
        }
        atomic_json(root / "status.json", status)
        emit({"type": "completed", **status})
        return 0
    except InterruptedError as exc:
        completed = getattr(experiment, "STATE", {}).get("current_round", 0) if experiment else 0
        status = {
            "state": "interrupted",
            "round": completed,
            "message": str(exc),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_before + time.monotonic() - active_start,
        }
        atomic_json(root / "status.json", status)
        emit({"type": "interrupted", **status})
        return 130
    except Exception as exc:
        detail = traceback.format_exc()
        (root / "failure.txt").write_text(detail, encoding="utf-8")
        status = {
            "state": "failed",
            "message": f"{type(exc).__name__}: {exc}",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_before + time.monotonic() - active_start,
        }
        atomic_json(root / "status.json", status)
        emit({"type": "error", **status, "traceback": detail})
        return 1
    finally:
        if experiment:
            experiment.log_file.close()
        lock.unlink(missing_ok=True)
        lease.release()
        event_file.close()


if __name__ == "__main__":
    sys.exit(main())
