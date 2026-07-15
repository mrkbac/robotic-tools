# Robotic Tools Agent Instructions

## Context

- `uv` workspace of Python libraries and CLIs for robotics data: MCAP, ROS messages, point clouds, video.
- Workspace packages: `digitalis/`, `pymcap-cli/`, `small-mcap/`, `mcap-ros2-support-fast/`, `ros-parser/`, `pointcloud2/`, `pureini/`, `robo-ws-bridge/`.
- `mcap-web-inspector/` is the Bun-based web UI.
- Pure Python only: no ROS runtime, hardware, or real-time assumptions.

## Commands

- Setup: `uv sync --all-groups --all-extras --all-packages`. Web: `bun install` inside `mcap-web-inspector/`.
- Python tests: `uv run pytest <package>/tests -m "not benchmark" --no-cov -q`.
- Single test file / test: `uv run pytest -s <package>/tests/test_x.py[::test_name]`.
- Lint/format only via pre-commit: `pre-commit run --files <paths>` or focused hooks (`ruff-check`, `ruff-format`, `typos`, `ty-check`, `pytest-fast`).
- Web checks: `bun run --cwd mcap-web-inspector` with `oxlint src/`, `tsc -b`, `build`.
- Rerun a failing command after fixing it before declaring it fixed.

## Tooling Rules

- Use `uv` for all Python deps and execution; no `pip`, `pipenv`, Poetry, or manual venv activation.
- `uv add` / `uv remove` for deps; `uvx` for one-off tools; run `pytest` through `uv run`.
- Use `bun` for `mcap-web-inspector/`, not `npm` / `node`.
- Search sibling packages for an existing helper before writing a non-trivial primitive.

## Python Rules

- Python 3.10+ type syntax: `list`, `dict`, `set`, `tuple`, `X | None`.
- Use `from __future__ import annotations` in new modules; add to existing only when it actually helps.
  Exception: command modules under `cmd/` rely on `cyclopts.Annotated[...]` parameter metadata at
  runtime, so the future-import must stay out of them.
- Avoid `Any`, bare `object`, `hasattr`, `getattr` — use concrete types, `Protocol`, or `TypeVar`.
- Prefer `_typeshed` protocol/alias types over `Any` or over-narrow concretes for buffers, paths,
  and streams: `ReadableBuffer`/`WriteableBuffer` (not `bytes`/`bytearray` for `readinto`-style code),
  `StrPath`/`StrOrBytesPath` (not `str | Path`), `SupportsRead`/`SupportsWrite` (not `IO[bytes]` when
  you only read/write). They are stub-only — import under `if TYPE_CHECKING:` and quote the annotation.
- `ty` reads the `.py` source of pure-Python deps, so untyped *pure-Python* packages need no stub.
  Only for compiled/extension deps (C/Rust `.so`/`.pyd` with no readable source and no `py.typed`)
  check for a `types-xxxx` stub package and add it to the dev group — before reaching for `Any`/a cast.
- `ty` honors bare `# type: ignore` and `# ty: ignore[rule]`, but NOT mypy-style `# type: ignore[code]`.
  Use `# ty: ignore[rule]` (with a reason) only when a real fix isn't feasible (generated code,
  third-party stub gaps).
- Imports top-level (rare exceptions: cyclic break, optional dep). Prefer direct imports over broad module imports.
- No module-level side effects; modules must import cleanly.
- Booleans: `is_` for state, `has_` for possession, `was_` only when past tense reads naturally. Name affirmatively — `is_enabled`, not `is_disabled`.
- `set_…` / `get_…` must actually set/get; otherwise rename. Factories read as `create_…`.
- Conversion helpers: `x_to_y`, where `x` and `y` are the type names.
- Distinguish timestamps from durations. Add unit suffixes only when ambiguous.
- Prefer `math.inf`, `pathlib.Path`, and current APIs over deprecated forms.
- Don't silently clamp, snap, or swallow invalid values; surface them.

## Data And Classes

- Use a `dataclass` for known-shaped data; don't pass `dict[str, Any]` between functions that share a schema.
- No index-addressed tuples for structured data.
- Use `T | None` for optional values, not a parallel "present" boolean.
- Objects must be usable after `__init__` — no separate `start()` / `setup()` / `connect()` step.
- Minimize mutable instance state; pass inputs as arguments and return results.
- Don't split into helpers before there's a second caller or a real readability need.

## Testing

- `foo.py` → `test_foo.py`. Plain pytest functions, named `test_<func>_<case>` or `test_<class>_<method>_<case>`.
- Tests must be fast and self-contained; mark slow / e2e / conformance / compat / benchmark with existing markers.
- Add smoke tests for CLI entry points.
- For any reported bug or issue, first write a failing test that reproduces it before changing code. The red test verifies the diagnosis and locks in regression coverage.
- Test oracles must not come from the function under test — hand-write, take from a spec, or use a separate implementation.
- Don't delete existing tests unless behavior is intentionally changing.

## Comments And Scope

- Default to no comments; add only for non-obvious constraints, invariants, or workarounds.
- Keep docstrings in sync with signatures.
- No commented-out code.
- Scope changes to the task — no drive-by style edits, speculative abstractions, compat shims, or validation for impossible internal cases.
- No half-finished implementations; say what blocked you.
