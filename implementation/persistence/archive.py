"""Stream the notebooks' final complete-run ZIP into an atomic unique artifact."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import zipfile


def archive_experiment(experiment) -> Path:
    """Preserve the optional final archive cell without overwriting prior runs.

    Science outputs are copied byte-for-byte. Transient worker controls are
    excluded, and archive status is a completion snapshot rather than a live PID.
    """
    root = experiment.ROOT.resolve()
    target = root.parent / (root.name + "_COMPLETE.zip")
    number = 1
    while target.exists():
        number += 1
        target = root.parent / f"{root.name}_COMPLETE_{number}.zip"
    temporary = target.with_suffix(".zip.incomplete")
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name not in {"worker.lock", "stop.request", "status.json"}
        and not p.name.endswith((".tmp", ".incomplete"))
    )
    experiment.stage("Creating complete experiment archive")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for index, path in enumerate(files, 1):
            experiment.check_stop()
            relative = path.relative_to(root).as_posix()
            with (
                path.open("rb") as source,
                archive.open(relative, "w", force_zip64=True) as destination,
            ):
                while chunk := source.read(1024 * 1024):
                    experiment.check_stop()
                    destination.write(chunk)
            experiment.emit(
                {
                    "type": "progress",
                    "stage": "Complete experiment archive",
                    "current": index,
                    "total": len(files),
                }
            )
        archive.writestr(
            "status.json",
            json.dumps(
                {
                    "state": "completed",
                    "round": experiment.STATE["current_round"],
                    "method": experiment.run_config.method,
                    "archive_snapshot": True,
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "note": "Completed scientific outputs. No worker PID/lock is active inside this archive. "
                    "Dataset caches are external to run directories; manifests identify their contents.",
                },
                indent=2,
            ),
        )
    experiment.check_stop()
    os.replace(temporary, target)
    experiment.emit({"type": "artifact", "kind": "complete_run_archive", "path": str(target)})
    experiment.log("Complete experiment archive:", target)
    return target
