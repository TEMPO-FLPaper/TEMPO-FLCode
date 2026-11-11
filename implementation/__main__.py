"""Lightweight terminal entry point, including missing-dependency guidance."""

import sys
from implementation.cli.terminal import configure_terminal


def main():
    configure_terminal()
    try:
        from implementation.cli.app import main as launch
    except ImportError as exc:
        print(
            f'TEMPO-FL cannot start: {exc}\nInstall from the project directory with:\n  python -m pip install -e .\nSee README.md for the tested setup.',
            file=sys.stderr,
        )
        return 1
    return launch()


if __name__ == "__main__":
    sys.exit(main())
