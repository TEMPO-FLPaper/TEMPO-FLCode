"""Exclusive project worker lease; stale records require a verified dead process."""

from __future__ import annotations
import json
import os
from pathlib import Path


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # Access denied is not evidence that a process is dead.
            return ctypes.get_last_error() != 87
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class WorkerLease:
    def __init__(self, path: Path, run_dir: Path) -> None:
        self.path, self.run_dir = path, run_dir
        self.owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    previous = json.loads(self.path.read_text(encoding="utf-8"))
                    alive = process_alive(int(previous["pid"]))
                except (ValueError, KeyError, OSError):
                    raise RuntimeError(
                        f"Worker lease cannot be verified: {self.path}. Inspect it before starting another worker."
                    )
                if alive:
                    raise RuntimeError(
                        f"An experiment worker is already active (PID {previous['pid']}): {previous.get('run_dir')}"
                    )
                self.path.unlink()
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump({"pid": os.getpid(), "run_dir": str(self.run_dir)}, handle)
                self.owned = True
                return
        raise RuntimeError("Unable to acquire the exclusive worker lease.")

    def release(self) -> None:
        if self.owned:
            self.path.unlink(missing_ok=True)
            self.owned = False
