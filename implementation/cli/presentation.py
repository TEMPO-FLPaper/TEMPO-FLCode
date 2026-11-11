"""Width-aware presentation shared by menus and reproducible commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from rich import box
from rich.console import Console, ConsoleDimensions
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from implementation.config.schema import METHODS, metadata
from implementation.cli.terminal import display_text, safe_output


class TerminalConsole(Console):
    def render_str(self, text, **kwargs):
        return super().render_str(display_text(text), **kwargs)

    def render(self, renderable, options=None):
        if isinstance(renderable, Text):
            renderable = Text(
                display_text(renderable.plain),
                justify=renderable.justify,
                overflow=renderable.overflow,
                no_wrap=renderable.no_wrap,
                end=display_text(renderable.end),
                tab_size=renderable.tab_size,
            )
        elif isinstance(renderable, str):
            renderable = display_text(renderable)
        yield from super().render(renderable, options)

    @property
    def size(self) -> ConsoleDimensions:
        dimensions = super().size
        if os.name == "nt" and getattr(self.file, "isatty", lambda: False)():
            # Classic conhost can have a wider scroll buffer than its viewport.
            import ctypes
            from ctypes import wintypes

            class Coord(ctypes.Structure):
                _fields_ = [("x", ctypes.c_short), ("y", ctypes.c_short)]

            class Rect(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_short),
                    ("top", ctypes.c_short),
                    ("right", ctypes.c_short),
                    ("bottom", ctypes.c_short),
                ]

            class Info(ctypes.Structure):
                _fields_ = [
                    ("buffer", Coord),
                    ("cursor", Coord),
                    ("attributes", wintypes.WORD),
                    ("window", Rect),
                    ("maximum", Coord),
                ]

            kernel = ctypes.WinDLL("kernel32")
            kernel.GetStdHandle.restype = wintypes.HANDLE
            kernel.GetConsoleScreenBufferInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Info)]
            info = Info()
            if kernel.GetConsoleScreenBufferInfo(kernel.GetStdHandle(-11), ctypes.byref(info)):
                width = info.window.right - info.window.left + 1
                height = info.window.bottom - info.window.top + 1
                if width > 0 and height > 0:
                    return ConsoleDimensions(width, height)
        return dimensions


def make_console(plain: bool = False, *, file=None, width=None) -> Console:
    """One stable profile, independent of shell, font and FORCE_COLOR settings.

    ``plain`` remains accepted for command-line compatibility; safe ASCII is
    always the default so callers cannot accidentally re-enable ANSI output.
    """
    return TerminalConsole(
        file=safe_output(sys.stdout if file is None else file),
        width=width,
        no_color=True,
        color_system=None,
        force_terminal=False,
        force_interactive=False,
        legacy_windows=False,
        highlight=False,
        emoji=False,
        markup=False,
    )


def table(console: Console, headers: list[str], rows, title: str = "") -> None:
    grid = Table(
        title=title or None,
        box=box.SIMPLE_HEAD,
        expand=True,
        show_lines=False,
        safe_box=True,
        padding=(0, 1),
    )
    for heading in headers:
        grid.add_column(heading, overflow="fold")
    for row in rows:
        grid.add_row(*(str(value) for value in row))
    console.print(grid)


def banner(console: Console, subtitle: str = "Reference research workbench") -> None:
    console.print(
        Panel(
            Text(f"TEMPO-FL  /  {subtitle}"),

            padding=(1, 2),
        )
    )


def show_config(console: Console, config, page: str | None = None) -> None:
    fields = [p for p in metadata(config.method) if page is None or p.page == page]
    console.print(
        f"{METHODS[config.method]} | {config.preset} | protocol: {config.protocol_label}"
    )
    table(
        console,
        ["Parameter", "Selected value", "State"],
        [
            (
                p.key,
                json.dumps(config.settings[p.key]),
                "Fixed"
                if not p.editable
                else "Changed"
                if config.settings[p.key] != p.default
                else "Reference",
            )
            for p in fields
        ],
        page or "Resolved scientific configuration",
    )
    if not fields:
        console.print("This section does not apply to the selected method.")
    console.print(Text(f"Output: {config.output_root}\nTrashNet cache: {config.dataset_cache}"))


def review(console: Console, config) -> None:
    banner(console, "Experiment review")
    table(
        console,
        ["Item", "Selection"],
        [
            ("Method", METHODS[config.method]),
            ("Preset", config.preset),
            ("Protocol", config.protocol_label),
            ("Seed", config.settings["seed"]),
            ("Rounds", config.settings["num_rounds"]),
            ("Dataset", config.settings["dataset_mode"]),
            (
                "Input / width",
                f"{config.settings['image_size']} px / alpha {config.settings['mobilenet_alpha']}",
            ),
            ("Scientific departures", len(config.departures())),
            ("Output", config.output_root),
            ("Dataset cache", config.dataset_cache),
        ],
    )
    if config.departures():
        table(
            console,
            ["Departure", "Reference", "Selected"],
            [
                (k, json.dumps(v["reference"]), json.dumps(v["selected"]))
                for k, v in config.departures().items()
            ],
        )
    if config.protocol_label == "verification":
        console.print(
            "Reduced verification run. This is not paper reproduction."
        )
    console.print(
        "Stops commit complete rounds. Partial initialization must restart; weights alone are not a full resume."
    )


def show_results(console: Console, run_dir: Path, name: str | None = None, limit: int = 15) -> dict:
    if not run_dir.is_dir():
        raise ValueError("Run directory does not exist.")
    path = run_dir / "results.json"
    if not path.is_file():
        console.print(
            "No committed results yet. Inspect status and logs; initialization may still be running."
        )
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    banner(console, "Measured results")
    console.print(
        Text(
            f"{METHODS[data['method']]} | {data['preset']} | {data['protocol_label']}\n"
            f"Committed round {data['current_round']} / {data['requested_rounds']}\n{run_dir}"
        )
    )
    console.print(
        "General test, owned test, specialist-only, cascade and accepted-only results have different scopes.\n"
        "Coverage and Unknown must accompany accepted-only accuracy. Stream events can repeat images."
    )
    if data["method"] == "tempo":
        console.print(Text(data.get("metric_warning", "")))
    tables = data["tables"]
    if name is None:
        table(
            console,
            ["Saved table", "Rows / entries"],
            [(key, len(value)) for key, value in tables.items()],
        )
        console.print(
            "Use results RUN --table TABLE to inspect a measured table; workbooks retain full precision and support counts."
        )
    else:
        if name not in tables:
            raise ValueError(f"Unknown result table {name!r}. Available: {', '.join(tables)}")
        values = tables[name]
        if isinstance(values, dict):
            console.print_json(data=values)
        elif values:
            # Vertical records preserve every field at narrow terminal widths.
            for row in values[-limit:]:
                table(
                    console,
                    ["Field", "Value"],
                    [
                        (k, json.dumps(v) if isinstance(v, (dict, list)) else str(v))
                        for k, v in row.items()
                    ],
                    name,
                )
        else:
            console.print("This table contains no events for this run.")
    return data
