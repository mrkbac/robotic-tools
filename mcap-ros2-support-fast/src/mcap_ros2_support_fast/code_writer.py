import textwrap
import types
from collections.abc import Generator, Iterator
from contextlib import contextmanager


class CodeWriter:
    def __init__(self, *, comments: bool) -> None:
        self._lines: list[str] = []
        self._level = 0
        self._indentation = "    "
        self._level_stack: list[int] = []
        self._comments = comments

    def append(self, lines: "str | CodeWriter | None") -> None:
        if lines is None:
            return

        for line in str(lines).splitlines():
            if line.strip():
                self._lines.append(textwrap.indent(line, self._indentation * self._level))

    def prolog(self, lines: "str | CodeWriter | None") -> None:
        if lines is None:
            return

        prolog_position = self._level_stack[-1] if self._level_stack else 0

        prolog_position += 1

        for line in str(lines).splitlines():
            self._lines.insert(prolog_position, f"{self._indentation * self._level}{line}")
            prolog_position += 1  # Adjust position for subsequent insertions

    def extend(self, lines: list["str | CodeWriter | None"]) -> None:
        for line in lines:
            self.append(line)

    def comment(self, lines: str | None) -> None:
        """Adds a comment to the code."""
        if self._comments is False:
            return
        if lines is None:
            return

        for line in lines.splitlines():
            self._lines.append(f"{self._indentation * self._level}# {line}")

    def __enter__(self) -> "CodeWriter":
        self._level += 1
        self._level_stack.append(len(self._lines))
        return self

    def __exit__(
        self,
        exc_: type[BaseException] | None,
        exc_type_: BaseException | None,
        tb_: types.TracebackType | None,
    ) -> None:
        self._level -= 1
        self._level_stack.pop()

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

    def get_code(self) -> str:
        return "\n".join(self._lines)


def _main() -> str:
    code_writer = CodeWriter(comments=True)
    code_writer.append("def foo():")
    with code_writer:
        code_writer.append("print('Hello, World!')")
        with code_writer.indent("for i in range(10):"):
            code_writer.comment("print('Hello before the for loop :)')")
            code_writer.append("print(i)\nprint(i+2)")
            code_writer.append("")
    return code_writer.get_code()


if __name__ == "__main__":
    print(_main())  # noqa: T201
