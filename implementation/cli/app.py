"""Interactive menus and argparse automation through the same experiment core."""

from __future__ import annotations

import argparse
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys

from rich.text import Text
from implementation.config.schema import RunConfig, METHODS, PAGES, metadata, preset_path, project_root
from implementation.cli.explanations import explain
from implementation.cli.presentation import banner, make_console, review, show_config, show_results, table
from implementation.cli.monitor import monitor_run
from implementation.execution.control import RunController, request_stop, wait_for_start
from implementation.cli.terminal import configure_terminal


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def open_paper() -> Path:
    path = project_root() / "TEMPO_FL.pdf"
    if not path.is_file():
        candidates = [
            Path(__file__).resolve().parents[2] / "share/tempo-fl/TEMPO_FL.pdf",
            Path(sys.prefix) / "share/tempo-fl/TEMPO_FL.pdf",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(
            "The packaged TEMPO_FL.pdf is missing. Reinstall the project or restore its copied paper."
        )
    if os.name == "nt":
        os.startfile(str(path))
    else:
        subprocess.Popen(
            ["open" if sys.platform == "darwin" else "xdg-open", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return path


def parse_value(text: str, default):
    if isinstance(default, str):
        return text.strip().strip('"')
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Use a JSON value: integer, decimal, true/false, or a list such as [0.1, 0.5]."
        ) from exc
    if type(default) is float and type(value) is int:
        value = float(value)
    return value


def set_parameter(config: RunConfig, key: str, text: str) -> RunConfig:
    fields = {p.key: p for p in metadata(config.method)}
    if key not in fields or not fields[key].editable:
        raise ValueError(f"{key}: unknown or fixed executable-reference setting.")
    candidate = deepcopy(config)
    candidate.settings[key] = parse_value(text, fields[key].default)
    candidate.validate()
    return candidate


def verification_config(method: str) -> RunConfig:
    """Explicit reduced real-algorithm preset; the phase calendar is unchanged."""
    return RunConfig.load(preset_path(method, "verification"))


def save_config(config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config.save(path)


class Workbench:
    def __init__(self, console) -> None:
        self.console = console
        self.drafts = {method: RunConfig.defaults(method) for method in METHODS}
        self.method = "tempo"
        self.controller = RunController()

    @property
    def config(self) -> RunConfig:
        return self.drafts[self.method]

    def ask(self, prompt="Select", default="") -> str:
        value = self.console.input(f"{prompt} > ").strip()
        return value or default

    def pause(self) -> None:
        self.ask("Enter to return")

    def select_method(self, method: str) -> None:
        self.method = method

    def edit_category(self, page: str) -> None:
        fields = [p for p in metadata(self.method) if p.page == page]
        if not fields:
            self.console.print("This section does not apply to the selected method.")
            self.pause()
            return
        page_index = 0
        page_size = 10
        while True:
            banner(self.console, page)
            if page == PAGES[3] and self.method in {"flat", "ditto"}:
                self.console.print(
                    "Context topology is data metadata for this method; there is no hierarchical aggregation."
                )
            visible = list(enumerate(fields, 1))[
                page_index * page_size : (page_index + 1) * page_size
            ]
            table(
                self.console,
                ["#", "Parameter", "Value", "State"],
                [
                    (
                        i,
                        p.key,
                        json.dumps(self.config.settings[p.key]),
                        "Fixed"
                        if not p.editable
                        else "Changed"
                        if self.config.settings[p.key] != p.default
                        else "Reference",
                    )
                    for i, p in visible
                ],
            )
            self.console.print(
                f"Page {page_index + 1} / {(len(fields) + page_size - 1) // page_size}. n next | p previous"
            )
            answer = self.ask("Number to edit | ?NUMBER for help | b back")
            if answer.lower() == "b":
                return
            if answer.lower() == "n":
                page_index = min(page_index + 1, (len(fields) - 1) // page_size)
                continue
            if answer.lower() == "p":
                page_index = max(0, page_index - 1)
                continue
            help_only = answer.startswith("?")
            index = answer.lstrip("?")
            if not index.isdigit() or not 1 <= int(index) <= len(fields):
                self.console.print("Choose a displayed number, ?NUMBER, or b.")
                continue
            p = fields[int(index) - 1]
            if help_only or not p.editable:
                self.console.print(Text(p.help()))
                self.pause()
                continue
            while True:
                value = self.ask(f"{p.key}: JSON value | ? help | b cancel")
                if value == "?":
                    self.console.print(Text(p.help()))
                    continue
                if value.lower() == "b":
                    break
                try:
                    self.drafts[self.method] = set_parameter(self.config, p.key, value)
                    self.console.print("Value saved in this method's draft.")
                    break
                except ValueError as exc:
                    self.console.print(Text(str(exc)))

    def configure(self) -> None:
        while True:
            banner(self.console, "Configuration categories")
            table(self.console, ["#", "Category"], list(enumerate(PAGES, 1)))
            self.console.print(
                "p  Data and output locations\ns  Save configuration\nl  Load configuration\n"
                "r  Restore Reference\nv  Load reduced verification preset\nb  Back"
            )
            answer = self.ask()
            if answer == "b":
                return
            if answer.isdigit() and 1 <= int(answer) <= len(PAGES):
                self.edit_category(PAGES[int(answer) - 1])
            elif answer == "p":
                self.locations()
            elif answer in {"s", "l"}:
                path = resolve_path(self.ask("JSON path (relative paths use project root)"))
                if answer == "s":
                    if path.exists() and self.ask("File exists. Replace? y/n", "n") != "y":
                        continue
                    save_config(self.config, path)
                    self.console.print(Text(f"Saved: {path}"))
                else:
                    c = RunConfig.load(path)
                    self.drafts[c.method] = c
                    self.method = c.method
                    self.console.print("Configuration loaded and validated.")
            elif answer in {"r", "v"}:
                previous = self.config
                c = (
                    RunConfig.defaults(self.method)
                    if answer == "r"
                    else verification_config(self.method)
                )
                c.output_root, c.dataset_cache = previous.output_root, previous.dataset_cache
                self.drafts[self.method] = c
                self.console.print(f"Loaded {c.preset}; local output/cache locations retained.")

    def locations(self) -> None:
        candidate = deepcopy(self.config)
        self.console.print(
            "Paths may contain spaces. Relative paths resolve against the project root. Blank keeps the current value."
        )
        for key, help_text in [
            (
                "output_root",
                "Run directories and complete logs. Default: project/runs. Any existing directory is allowed.",
            ),
            (
                "dataset_cache",
                "TrashNet download/extraction cache. Default: project/datasets. Uses only the reference dataset.",
            ),
        ]:
            while True:
                self.console.print(Text(f"{key}: {getattr(candidate, key)}"))
                value = self.ask("New directory | ? help | Enter keep")
                if value == "?":
                    self.console.print(help_text)
                    continue
                if value:
                    setattr(candidate, key, str(resolve_path(value)))
                break
        candidate.validate()
        self.drafts[self.method] = candidate
        self.console.print(
            "Locations saved. Dataset mode/local-folder selection is in Dataset and partitions."
        )

    def choose_run(self) -> Path:
        root = Path(self.config.output_root)
        runs = sorted((p.parent for p in root.glob("*/resolved_config.json")), reverse=True)[:15]
        table(self.console, ["#", "Run"], list(enumerate((p.name for p in runs), 1)))
        value = self.ask("Run number or path")
        if value.isdigit() and 1 <= int(value) <= len(runs):
            return runs[int(value) - 1]
        return resolve_path(value)

    def start(self, resume: Path | None = None) -> None:
        c = RunConfig.load(resume / "resolved_config.json") if resume else self.config
        c.validate()
        review(self.console, c)
        if self.ask("Start this resolved experiment? y/n", "n") != "y":
            return
        run_dir = self.controller.start(c, resume)
        wait_for_start(self.controller)
        self.console.print(Text(f"Worker started: {run_dir}"))
        monitor_run(self.console, run_dir, self.controller, interactive=True)

    def results(self) -> None:
        path = self.choose_run()
        data = show_results(self.console, path)
        if not data:
            self.pause()
            return
        while True:
            name = self.ask("Table name | b back")
            if name == "b":
                return
            show_results(self.console, path, name, 3)

    def run(self) -> int:
        while True:
            try:
                banner(self.console)
                c = self.config
                self.console.print(
                    f"{METHODS[self.method]} | {c.preset} | {len(c.departures())} scientific departures"
                )
                table(
                    self.console,
                    ["Key", "Workspace"],
                    [
                        ("1", "Overview and experiment review"),
                        ("2", "Select method"),
                        ("3", "Categorized configuration / save / load"),
                        ("4", "Data and output locations"),
                        ("5", "Start experiment"),
                        ("6", "Run monitoring"),
                        ("7", "Request safe stop"),
                        ("8", "Resume committed run"),
                        ("9", "Inspect measured results"),
                        ("10", "How the methods work"),
                        ("11", "Open paper"),
                        ("12", "Project information"),
                        ("q", "Exit terminal"),
                    ],
                )
                if self.controller.running:
                    self.console.print(
                        "Worker active. Navigation remains available; use 6 to monitor or 7 to stop."
                    )
                answer = self.ask()
                if answer == "q":
                    if self.controller.running:
                        action = self.ask(
                            "Worker active: s stop and wait | d detach and exit | b back", "b"
                        )
                        if action == "s":
                            request_stop(self.controller.run_dir)
                            monitor_run(self.console, self.controller.run_dir, self.controller)
                        elif action != "d":
                            continue
                        self.console.print(Text(f"Run: {self.controller.run_dir}"))
                    return 0
                if answer == "1":
                    review(self.console, c)
                    self.pause()
                elif answer == "2":
                    table(self.console, ["Key", "Method"], METHODS.items())
                    value = self.ask("Method key | b back")
                    if value in METHODS:
                        self.select_method(value)
                elif answer == "3":
                    self.configure()
                elif answer == "4":
                    self.locations()
                elif answer == "5":
                    self.start()
                elif answer == "6":
                    path = self.controller.run_dir or self.choose_run()
                    monitor_run(
                        self.console,
                        path,
                        self.controller if self.controller.run_dir == path else None,
                        interactive=True,
                    )
                    self.pause()
                elif answer == "7":
                    path = self.controller.run_dir or self.choose_run()
                    request_stop(path)
                    self.console.print(
                        "Safe stop requested. Monitor the run for the last committed checkpoint."
                    )
                elif answer == "8":
                    self.start(self.choose_run())
                elif answer == "9":
                    self.results()
                elif answer == "10":
                    value = self.ask("Method key (tempo/ditto/hfl/flat) or all", "all")
                    if value != "all" and value not in METHODS:
                        raise ValueError("Unknown method key.")
                    explain(self.console, None if value == "all" else value)
                    self.pause()
                elif answer == "11":
                    self.console.print(Text(f"Opened paper: {open_paper()}"))
                elif answer == "12":
                    self.console.print(
                        Text(
                            f"TEMPO-FL: Temporary Model Escalation and Selective Promotion for Adaptive Hierarchical Federated Learning\n"
                            f"Project: {project_root()}\nFour immutable reference notebooks; no runtime notebook execution.\n"
                            "Reference caveats: README.md\nVerification evidence: README.md\n"
                            "No full reference experiment reproduction is implied by reduced verification runs."
                        )
                    )
                    self.pause()
                else:
                    self.console.print("Choose a displayed key.")
            except (ValueError, OSError, KeyError) as exc:
                self.console.print(Text(str(exc)))
            except KeyboardInterrupt:
                self.console.print(
                    "Cancelled menu action. Use q to exit or 7 to stop an active worker safely."
                )
            except EOFError:
                self.console.print(
                    "Terminal input closed. Any detached worker continues; use the stop command to request a safe stop."
                )
                return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TEMPO-FL terminal research application. No command opens interactive menus."
    )
    p.add_argument(
        "--plain", action="store_true", help="Stable ASCII display (already the default)."
    )
    commands = p.add_subparsers(dest="command")
    commands.add_parser(
        "interactive", help="Open categorized interactive menus (requires a terminal)."
    )
    commands.add_parser("methods", help="List the four executable-reference methods.")
    ex = commands.add_parser(
        "explain", help="Show implementation-based ASCII diagrams and comparison."
    )
    ex.add_argument("--method", choices=METHODS)
    config = commands.add_parser(
        "config", help="Create, inspect, validate, edit, and explain configurations."
    )
    sub = config.add_subparsers(dest="action", required=True)
    for name in ("reference", "verification"):
        cp = sub.add_parser(
            name,
            help=f"Save a {'reduced real-algorithm verification' if name == 'verification' else 'Reference'} preset.",
        )
        cp.add_argument("--method", choices=METHODS, default="tempo")
        cp.add_argument("--output", required=True)
    for name in ("show", "validate", "set"):
        cp = sub.add_parser(name)
        cp.add_argument("--config", required=True)
        if name == "show":
            cp.add_argument("--category", choices=PAGES)
        if name == "set":
            cp.add_argument(
                "--set", dest="changes", action="append", required=True, metavar="KEY=JSON"
            )
            cp.add_argument("--output", required=True)
    hp = sub.add_parser(
        "help", help="Context, type, range, units, dependencies and source for a parameter."
    )
    hp.add_argument("parameter", help="Parameter key, or 'output_root' / 'dataset_cache'.")
    hp.add_argument("--method", choices=METHODS, default="tempo")
    rp = commands.add_parser(
        "run", help="Start a separate worker and monitor it; Ctrl+C requests safe stop."
    )
    rp.add_argument("--config", required=True)
    rp.add_argument(
        "--detach",
        action="store_true",
        help="Return immediately; worker saves logs/status to disk.",
    )
    rs = commands.add_parser(
        "resume", help="Restore a compatible committed complete-round checkpoint."
    )
    rs.add_argument("run_dir")
    rs.add_argument("--detach", action="store_true")
    mo = commands.add_parser(
        "monitor", help="Inspect structured status/logs without importing TensorFlow."
    )
    mo.add_argument("run_dir")
    mo.add_argument("--follow", action="store_true")
    st = commands.add_parser("stop", help="Request cooperative stop; does not force-kill a worker.")
    st.add_argument("run_dir")
    results = commands.add_parser(
        "results", help="Inspect one run, keeping metric and protocol scopes distinct."
    )
    results.add_argument("run_dir")
    results.add_argument("--table")
    results.add_argument("--limit", type=int, default=3)
    commands.add_parser(
        "paper", help="Open the copied PDF in the operating system's default viewer."
    )
    commands.add_parser(
        "doctor", help="Inspect installed dependencies without initializing TensorFlow."
    )
    return p


def main(argv=None) -> int:
    configure_terminal()
    p = parser()
    args = p.parse_args(argv)
    console = make_console(args.plain)
    try:
        if args.command in {None, "interactive"}:
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                console.print(
                    "Interactive menus require a terminal. Use --help for non-interactive commands."
                )
                return 2
            return Workbench(console).run()
        if args.command == "methods":
            table(console, ["Key", "Reference method"], METHODS.items())
        elif args.command == "explain":
            explain(console, args.method)
        elif args.command == "config":
            if args.action in {"reference", "verification"}:
                c = (
                    RunConfig.defaults(args.method)
                    if args.action == "reference"
                    else verification_config(args.method)
                )
                path = resolve_path(args.output)
                if path.exists():
                    raise FileExistsError(
                        "Configuration already exists. Choose another output path."
                    )
                save_config(c, path)
                console.print(Text(f"Saved {c.preset}: {path}"))
            elif args.action == "help":
                if args.parameter in {"output_root", "dataset_cache"}:
                    c = RunConfig.defaults(args.method)
                    console.print(
                        Text(
                            f"{args.parameter}: local directory path\nReference default: {getattr(c, args.parameter)}\n"
                            "Relative paths resolve against the project root, independent of launch directory. "
                            "Output and cache locations do not change the scientific formulas. "
                            "Dataset-cache relocation changes recorded absolute paths and can invalidate resume data identity."
                        )
                    )
                else:
                    field = next(
                        (f for f in metadata(args.method) if f.key == args.parameter), None
                    )
                    if field is None:
                        raise ValueError(
                            "Unknown parameter for this method. Use config show to list its settings."
                        )
                    console.print(Text(field.help()))
            else:
                c = RunConfig.load(resolve_path(args.config))
                if args.action == "show":
                    show_config(console, c, args.category)
                elif args.action == "validate":
                    console.print(
                        f"Valid: {METHODS[c.method]}, {c.preset}, {len(c.departures())} departures."
                    )
                elif args.action == "set":
                    for change in args.changes:
                        if "=" not in change:
                            raise ValueError("Each --set requires KEY=JSON.")
                        key, value = change.split("=", 1)
                        if key in {"output_root", "dataset_cache"}:
                            setattr(c, key, str(resolve_path(value.strip('"'))))
                        else:
                            # Validate the combined transaction so dependent bounds can change together.
                            fields = {f.key: f for f in metadata(c.method)}
                            if key not in fields or not fields[key].editable:
                                raise ValueError(f"{key}: unknown or fixed setting.")
                            c.settings[key] = parse_value(value, fields[key].default)
                    c.validate()
                    save_config(c, resolve_path(args.output))
                    console.print("Configuration saved and validated.")
        elif args.command in {"run", "resume"}:
            root = resolve_path(args.run_dir) if args.command == "resume" else None
            c = RunConfig.load(root / "resolved_config.json" if root else resolve_path(args.config))
            review(console, c)
            controller = RunController()
            root = controller.start(c, root)
            wait_for_start(controller)
            console.print(Text(f"Run directory: {root}"))
            if args.detach:
                console.print(
                    "Worker continues independently. Use monitor --follow, stop, results, or resume as appropriate."
                )
                return 0
            return monitor_run(console, root, controller)
        elif args.command == "monitor":
            return monitor_run(console, resolve_path(args.run_dir), follow=args.follow)
        elif args.command == "stop":
            request_stop(resolve_path(args.run_dir))
            console.print(
                "Safe stop requested. A running round finishes before its complete checkpoint is committed."
            )
        elif args.command == "results":
            if args.limit < 1:
                raise ValueError("--limit must be positive.")
            show_results(console, resolve_path(args.run_dir), args.table, args.limit)
        elif args.command == "paper":
            console.print(Text(f"Opened paper: {open_paper()}"))
        elif args.command == "doctor":
            rows = []
            for package in (
                "rich",
                "tensorflow",
                "keras",
                "numpy",
                "scipy",
                "scikit-learn",
                "pandas",
                "Pillow",
                "matplotlib",
                "openpyxl",
                "requests",
            ):
                try:
                    rows.append((package, version(package)))
                except PackageNotFoundError:
                    rows.append((package, 'MISSING: install -e .'))
            table(console, ["Package", "Installed version"], rows)
            console.print(
                Text(
                    f"Python: {sys.version}\nProject resources: {project_root()}\nTensorFlow has not been imported by this command."
                )
            )
        return 0
    except (ValueError, OSError, KeyError) as exc:
        console.print(Text(f"Error: {exc}"))
        return 2


if __name__ == "__main__":
    sys.exit(main())
