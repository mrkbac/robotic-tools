"""Smoke tests: import every public module to catch compatibility errors early."""

import importlib
import pkgutil

import pytest

import digitalis


_SKIP = frozenset({"digitalis.__main__"})


def _all_submodules() -> list[str]:
    """Walk the digitalis package tree and return all importable module names."""
    modules: list[str] = [digitalis.__name__]
    prefix = digitalis.__name__ + "."
    for info in pkgutil.walk_packages(digitalis.__path__, prefix):
        if info.name not in _SKIP:
            modules.append(info.name)
    return sorted(modules)


@pytest.mark.parametrize("module_name", _all_submodules())
def test_import(module_name: str) -> None:
    """Every module in the package must be importable on all supported Pythons."""
    importlib.import_module(module_name)
