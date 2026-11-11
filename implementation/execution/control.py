"""Process-isolated execution and bounded structured-event monitoring."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from implementation.config.schema import RunConfig, project_root


def read_status(run_dir: Path) -> dict:
    path = run_dir / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"state": "starting"}


def request_stop(run_dir: Path) -> None:
    if not run_dir.is_dir():
        raise ValueError("Run directory does not exist.")
    if read_status(run_dir).get("state") in {"completed", "interrupted", "failed"}:
        raise ValueError("This run is not active.")
    (run_dir / "stop.request").write_text(
        "Cooperative stop requested by terminal.\n", encoding="utf-8"
    )


class RunController:
    """One independently running worker; navigation never imports TensorFlow."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.run_dir: Path | None = None
        self.event_start_offset = 0

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, config: RunConfig, resume_dir: Path | None = None) -> Path:
        if self.running:
            raise ValueError(
                "An experiment is already running. Monitor or stop it before starting another."
            )
        config.validate()
        output = Path(config.output_root).expanduser().resolve()
        if resume_dir:
            self.run_dir = resume_dir.resolve()
            config_path = self.run_dir / "resolved_config.json"
            if not (self.run_dir / "03_checkpoints/latest.json").is_file():
                raise ValueError(
                    "No committed round checkpoint. Partial initialization must restart in a new run."
                )
            config = RunConfig.load(config_path)
        else:
            self.run_dir = output / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%f}_{config.method}"
            pending = output / ".pending"
            pending.mkdir(parents=True, exist_ok=True)
            config_path = pending / f"{self.run_dir.name}.json"
            config.save(config_path)
        events_path = self.run_dir / "events.jsonl"
        self.event_start_offset = events_path.stat().st_size if events_path.exists() else 0
        args = [
            sys.executable,
            "-u",
            "-m",
            "implementation.execution.worker",
            "--config",
            str(config_path),
            "--run-dir",
            str(self.run_dir),
        ]
        if resume_dir:
            args.append("--resume")
        # Complete subprocess output goes to disk. State comes only from events/status.
        transport = self.run_dir.parent / f".{self.run_dir.name}.transport.log"
        transport.parent.mkdir(parents=True, exist_ok=True)
        with transport.open("a", encoding="utf-8") as handle:
            self.process = subprocess.Popen(
                args,
                cwd=project_root(),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
                if os.name == "nt"
                else 0,
                start_new_session=os.name != "nt",
            )
        return self.run_dir


class RunMonitor:
    """Tail persisted typed events with fixed-size buffers and bounded reads."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.offset = 0
        self.partial = b""
        self.logs: deque[str] = deque(maxlen=8)
        self.metrics: deque[dict] = deque(maxlen=12)
        self.stage = "Waiting for worker"
        self.progress = "Total unknown"
        self.round = 0
        self.checkpoint: int | None = None
        self.state = "starting"
        self.method = "pending"
        self.protocol = "pending"
        self.started_at: str | None = None
        self.elapsed_before = 0.0
        self.ended_at: str | None = None
        self.seen = 0

    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.ended_at) if self.ended_at else datetime.now(timezone.utc)
        return self.elapsed_before + max(0.0, (end - start).total_seconds())

    def poll(self) -> list[dict]:
        path = self.run_dir / "events.jsonl"
        if not path.is_file():
            return []
        if path.stat().st_size < self.offset:
            self.offset, self.partial = 0, b""
        with path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read(256 * 1024)
            self.offset = handle.tell()
        lines = (self.partial + chunk).split(b"\n")
        self.partial = lines.pop()[-1024 * 1024 :]
        events = []
        for line in lines:
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                self.logs.append("Malformed event record; inspect events.jsonl.")
                continue
            self.consume(event)
            events.append(event)
        return events

    def consume(self, event: dict) -> None:
        self.seen += 1
        kind = event.get("type")
        if kind == "started":
            self.state = "running"
            self.method = event.get("method", "unknown")
            self.protocol = event.get("protocol_label", "unknown")
            self.started_at = event.get("time")
            self.elapsed_before = event.get("elapsed_before", 0.0)
            self.ended_at = None
        elif kind == "stage":
            self.stage = event["stage"]
            self.progress = "Total unknown"
        elif kind in {"progress", "download", "epoch", "round"}:
            if kind == "round":
                self.round = event["current"]
            if kind == "epoch":
                self.progress = f"Epoch {event['epoch']} completed"
                self.metrics.append({"source": event.get("stage"), **event.get("metrics", {})})
            else:
                current, total = event.get("current"), event.get("total")
                self.progress = (
                    f"{kind}: {current} / {total}" if total else f"{kind}: {current}; total unknown"
                )
        elif kind == "metrics":
            self.metrics.extend(event.get("rows", []))
        elif kind in {"checkpoint", "restored"}:
            self.checkpoint = event["round"]
            self.logs.append(f"{kind.capitalize()}: valid complete round {self.checkpoint}")
        elif kind in {"completed", "interrupted", "error"}:
            self.state = {
                "completed": "completed",
                "interrupted": "interrupted",
                "error": "failed",
            }[kind]
            self.ended_at = event.get("time")
            self.logs.append(event.get("message", self.state.capitalize()))
        elif kind == "log":
            self.logs.append(str(event.get("message", ""))[-2000:])


def wait_for_start(controller: RunController, timeout: float = 5.0) -> None:
    """Wait only for filesystem readiness, never for scientific initialization."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and controller.running:
        if (controller.run_dir / "events.jsonl").exists():
            return
        time.sleep(0.05)
