"""Throttled terminal monitoring with coordinated live logs and stop controls."""

from __future__ import annotations
import json
import os
from pathlib import Path
import select
import sys
import time
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from implementation.execution.control import RunMonitor, request_stop

TERMINAL_STATES = {"completed", "interrupted", "failed"}


def render_monitor(monitor: RunMonitor, width: int = 100, height: int = 40):
    elapsed = int(monitor.elapsed())
    summary = Text(
        f"Method: {monitor.method} | Protocol: {monitor.protocol}"
        + (" (reduced; not paper reproduction)" if monitor.protocol == "verification" else "")
        + "\n"
        f"State: {monitor.state.upper()}    Elapsed: {elapsed // 60:02d}:{elapsed % 60:02d}\n"
        f"Stage: {monitor.stage}\n{monitor.progress}\n"
        f"Round: {monitor.round} | Last valid checkpoint: {monitor.checkpoint if monitor.checkpoint is not None else 'none'}\n"
        f"Run: {monitor.run_dir}"
    )
    metrics = Table("Metric / scope", "Latest measured value", box=None, expand=True)
    if monitor.metrics:
        row = monitor.metrics[-1]
        for key, value in row.items():
            if isinstance(value, (int, float)) and any(
                term in key for term in ("loss", "accuracy", "f1", "rate", "score")
            ):
                metrics.add_row(key, f"{value:.6g}")
                if metrics.row_count >= (3 if height < 32 else 5):
                    break
        if not metrics.row_count:
            metrics.add_row(str(row.get("table", "Metrics")), "Saved in results.json")
    else:
        metrics.add_row("Metrics", "No measured metrics yet")
    logs = Text()
    log_count = max(1, min(8, height - 24))
    for line in list(monitor.logs)[-log_count:]:
        # Bounded one-line log summaries keep controls visible in a short terminal.
        text = " ".join(line.split())
        logs.append(text[: max(20, width - 8)] + ("..." if len(text) > width - 8 else "") + "\n")
    if not monitor.logs:
        logs.append("Waiting for structured worker events. Full output is saved to disk.")
    return Group(
        Panel(summary, title="TEMPO-FL / Run monitoring"),
        Panel(metrics, title="Measured metrics (scope retained)"),
        Panel(logs, title="Recent logs / complete logs on disk"),
        Text(
            "s: request safe stop | b: return to menu (worker continues) | Ctrl+C: safe stop",

        ),
    )


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    if os.name == "nt":
        import msvcrt

        return msvcrt.getwch().lower() if msvcrt.kbhit() else None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.readline().strip().lower() if ready else None


def describe_event(event: dict) -> str:
    """Readable redirected output from typed fields, including unknown totals."""
    kind = event.get("type")
    if kind in {"round", "progress", "download"}:
        total = event.get("total")
        return f"{event.get('stage', kind)}: {event.get('current')} / {total if total else 'unknown total'}"
    if kind == "epoch":
        return f"{event.get('stage', 'Training')} epoch {event['epoch']}: {json.dumps(event.get('metrics', {}))}"
    if kind == "artifact":
        return f"{event.get('kind')}: {event.get('path')}"
    if kind == "started":
        return f"{event.get('method')} | {event.get('protocol_label')} | {event.get('run_dir')}"
    if kind == "metrics":
        return (
            f"round {event.get('round')}; {len(event.get('rows', []))} measured scoped rows saved"
        )
    return str(event.get("stage", event.get("message", event.get("round", event.get("state", "")))))


def monitor_run(console, run_dir: Path, controller=None, follow=True, interactive=False) -> int:
    monitor = RunMonitor(run_dir)
    if controller is not None and controller.running:
        monitor.offset = controller.event_start_offset
    if not run_dir.is_dir() and controller is None:
        raise ValueError("Run directory does not exist.")
    dynamic = console.is_terminal and not console.no_color and follow
    live = Live(console=console, auto_refresh=False, transient=False) if dynamic else None
    if live:
        live.start()
    stop_requested = False
    last_status = time.monotonic()
    if follow and not dynamic:
        controls = (
            "s: safe stop | b: back | Ctrl+C: safe stop" if interactive else "Ctrl+C: safe stop"
        )
        console.print(Text(f"Run monitoring | {controls}"))
    try:
        while True:
            try:
                events = monitor.poll()
                if (
                    controller is not None
                    and controller.process.poll() is not None
                    and monitor.state not in TERMINAL_STATES
                ):
                    monitor.poll()
                    event_path = run_dir / "events.jsonl"
                    caught_up = (
                        not event_path.exists() or monitor.offset >= event_path.stat().st_size
                    )
                    if monitor.state not in TERMINAL_STATES and caught_up:
                        monitor.state = "failed"
                        monitor.logs.append(
                            f"Worker exited ({controller.process.returncode}) without a terminal event. Inspect transport log."
                        )
                if live:
                    live.update(
                        render_monitor(monitor, console.width, console.height), refresh=True
                    )
                else:
                    for event in events:
                        kind = event.get("type")
                        if follow and kind != "log":
                            console.print(
                                Text(f"[{event.get('time', '')}] {kind}: {describe_event(event)}")
                            )
                    if follow and time.monotonic() - last_status >= 5:
                        elapsed = int(monitor.elapsed())
                        checkpoint = (
                            monitor.checkpoint if monitor.checkpoint is not None else "none"
                        )
                        console.print(
                            Text(
                                f"Status: {monitor.state.upper()} | Elapsed: {elapsed // 60:02d}:{elapsed % 60:02d} | "
                                f"Stage: {monitor.stage} | Round: {monitor.round} | Checkpoint: {checkpoint}"
                            )
                        )
                        last_status = time.monotonic()
                if not follow or monitor.state in TERMINAL_STATES:
                    # Finish consuming the file before declaring a terminal outcome.
                    if events and (run_dir / "events.jsonl").stat().st_size > monitor.offset:
                        continue
                    break
                key = read_key() if interactive else None
                if key == "b":
                    return 0
                if key == "s" and not stop_requested:
                    request_stop(run_dir)
                    stop_requested = True
                    monitor.logs.append(
                        "Stop requested. Waiting for a safe boundary; no partial checkpoint will be committed."
                    )
                time.sleep(0.25)
            except KeyboardInterrupt:
                if run_dir.exists():
                    request_stop(run_dir)
                    stop_requested = True
                    message = "Stop requested. Waiting for the current safe boundary; last complete checkpoint is retained."
                    monitor.logs.append(message)
                    if not live:
                        console.print(message)
                else:
                    console.print(
                        "Worker is starting; request stop again once its run directory exists."
                    )
    finally:
        if live:
            live.stop()
    if not dynamic:
        console.print(render_monitor(monitor, console.width))
    if monitor.state == "interrupted":
        console.print("Interrupted. Resume is available from a committed complete round only.")
        return 130
    return 1 if monitor.state == "failed" else 0
