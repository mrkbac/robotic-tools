import textwrap
from collections.abc import Generator, Iterator
from contextlib import contextmanager


class CodeWriter:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._level = 0
        self._indentation = "    "

    def append(self, lines: "str | CodeWriter | None") -> None:
        if lines is None:
            return

        for line in str(lines).splitlines():
            if line.strip():
                self._lines.append(textwrap.indent(line, self._indentation * self._level))

    @contextmanager
    def indent(self, lines: str | None) -> Generator["CodeWriter", None, None]:
        if lines is not None:
            self.append(lines)
        self._level += 1
        try:
            yield self
        finally:
            self._level -= 1

    def __str__(self) -> str:
        return "\n".join(self._lines)

    def __iter__(self) -> Iterator[str]:
        return iter(self._lines)
