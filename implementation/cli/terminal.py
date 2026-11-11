"""One control-free ASCII display profile for every terminal entry point.

Only presentation is normalized. Configuration values, scientific state and
UTF-8 experiment files are never passed through this display filter.
"""

from __future__ import annotations

import sys
from typing import TextIO


TRANSLATIONS = str.maketrans(
    {
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2022": "*",
        "\u2190": "<-",
        "\u2192": "->",
        "\u2191": "^",
        "\u2193": "v",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00d7": "x",
        "\u00b2": "^2",
        "\u0394": "delta",
        "\u03b1": "alpha",
        "\u03bb": "lambda",
    }
)


class TerminalFilter:
    """Discard terminal controls, including escape sequences split across writes."""

    def __init__(self) -> None:
        self.state = "text"
        self.length = 0

    def feed(self, value: str) -> str:
        output = []
        for char in value:
            code = ord(char)
            if self.state != "text":
                self.length += 1
                if self.length > 4096:
                    self.state, self.length = "text", 0
                    continue
            if self.state == "text":
                if char == "\x1b":
                    self.state, self.length = "escape", 0
                elif char == "\x9b":
                    self.state, self.length = "csi", 0
                elif char in {"\x90", "\x9d", "\x9e", "\x9f"}:
                    self.state, self.length = "string", 0
                elif char in "\n\t" or code >= 32 and not 127 <= code <= 159:
                    output.append(char)
            elif self.state == "escape":
                if char == "[":
                    self.state = "csi"
                elif char in "]P^_":
                    self.state = "string"
                elif " " <= char <= "/":
                    self.state = "escape_intermediate"
                else:
                    self.state = "text"
            elif self.state == "escape_intermediate":
                if "0" <= char <= "~":
                    self.state = "text"
            elif self.state == "csi":
                if "@" <= char <= "~":
                    self.state = "text"
            elif self.state == "string":
                if char in {"\x07", "\x9c"}:
                    self.state = "text"
                elif char == "\x1b":
                    self.state = "string_escape"
            elif self.state == "string_escape":
                self.state = "text" if char == "\\" else "string"
        return (
            "".join(output)
            .translate(TRANSLATIONS)
            .encode("ascii", "backslashreplace")
            .decode("ascii")
        )


def display_text(value: str) -> str:
    """Convert presentation text before Rich measures and wraps it."""
    return TerminalFilter().feed(value)


class SafeOutput:
    """Text stream with ASCII encoding and no cursor/color control sequences."""

    encoding = "ascii"
    errors = "backslashreplace"

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.filter = TerminalFilter()

    def write(self, value: str) -> int:
        text = self.filter.feed(value)
        if text:
            self.stream.write(text)
        return len(value)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def __getattr__(self, name):
        return getattr(self.stream, name)


def safe_output(stream: TextIO) -> SafeOutput:
    return stream if isinstance(stream, SafeOutput) else SafeOutput(stream)


def configure_terminal() -> None:
    """Use the same safe display for prompts, argparse, errors and reports."""
    sys.stdout = safe_output(sys.stdout)
    sys.stderr = safe_output(sys.stderr)
