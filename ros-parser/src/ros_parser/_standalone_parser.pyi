from typing import Any, Generic, TypeVar

_T = TypeVar("_T")

class Token:
    value: str
    type: str

    def __init__(self, type_: str, value: Any) -> None: ...
    def __int__(self) -> int: ...

class Transformer(Generic[_T]):
    def __init__(self) -> None: ...

class Lark_StandAlone(Generic[_T]):  # noqa: N801
    def __init__(self, transformer: Transformer[_T]) -> None: ...
    def parse(self, text: str) -> _T: ...
