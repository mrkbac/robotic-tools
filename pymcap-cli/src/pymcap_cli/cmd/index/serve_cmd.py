"""``pymcap-cli index serve`` — browse the sidecar catalog with Datasette.

Launches Datasette against the index DB from the *same* interpreter
(``sys.executable -m datasette``), so the bundled ``pymcap_render`` plugin can
``import pymcap_cli``. Datasette and its plugins ship in the ``serve`` extra
(``pip install 'pymcap-cli[serve]'``).
"""

import importlib.resources
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

from cyclopts import Group

from pymcap_cli.cmd._cli_options import (
    IndexDbOption,
    NoBrowserOption,
    ServerHostOption,
    ServerPortOption,
)
from pymcap_cli.cmd.index._helpers import _print_db_needs_migration, _resolve_db
from pymcap_cli.index.db import IndexDbNeedsMigrationError, connect
from pymcap_cli.log_setup import ERR

INDEX_SERVE_OPTIONS_GROUP = Group("Index Serve Options")


def _build_datasette_argv(
    db_link: Path,
    metadata: Path,
    plugins_dir: Path,
    template_dir: Path,
    host: str,
    port: int,
    *,
    open_browser: bool,
) -> list[str]:
    """Assemble the ``python -m datasette serve …`` command line."""
    argv = [
        sys.executable,
        "-m",
        "datasette",
        "serve",
        str(db_link),
        "--metadata",
        str(metadata),
        "--plugins-dir",
        str(plugins_dir),
        "--template-dir",
        str(template_dir),
        "--setting",
        "sql_time_limit_ms",
        "10000",
        "--setting",
        "allow_download",
        "off",
        "--setting",
        "default_cache_ttl",
        "0",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if open_browser:
        argv.append("-o")
    return argv


def _datasette_is_installed() -> bool:
    return importlib.util.find_spec("datasette") is not None


def _create_stable_db_link(db_path: Path, db_link: Path) -> None:
    """Create a stable Datasette DB path, falling back when symlinks are unavailable."""
    resolved = db_path.resolve()
    try:
        db_link.symlink_to(resolved)
    except OSError:
        pass
    else:
        return

    try:
        db_link.hardlink_to(resolved)
    except OSError:
        shutil.copy2(resolved, db_link)
    else:
        return


def _validate_db_readable(db_path: Path) -> bool:
    try:
        conn = connect(db_path, read_only=True)
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return False
    except RuntimeError as exc:
        ERR.print(f"[red]Error:[/red] {exc}")
        return False
    else:
        conn.close()
        return True


def index_serve(
    *,
    db: IndexDbOption = None,
    host: ServerHostOption = "127.0.0.1",
    port: ServerPortOption = 8001,
    no_browser: NoBrowserOption = False,
) -> int:
    """Browse the index catalog in a local Datasette web UI.

    Serves the sidecar DB read-only with the bundled metadata, dashboards, and
    ``pymcap_render`` plugin. Needs the ``serve`` extra
    (``pip install 'pymcap-cli[serve]'``).
    """
    db_path = _resolve_db(db)
    if not db_path.exists():
        ERR.print(f"[red]Error:[/red] no index DB at {db_path}")
        return 1

    if not _datasette_is_installed():
        ERR.print(
            "[red]Error:[/red] Datasette is not installed. Install the serve extra: "
            "`uv run --package pymcap-cli --extra serve pymcap-cli index serve` "
            "or `pip install 'pymcap-cli[serve]'`."
        )
        return 1

    if not _validate_db_readable(db_path):
        return 1

    with ExitStack() as stack:
        # Datasette names a DB by its filename stem; serve through a stable
        # `index.sqlite` path so the metadata/template `/index/...` links
        # resolve regardless of the real --db filename.
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        db_link = tmp / "index.sqlite"
        _create_stable_db_link(db_path, db_link)

        assets = stack.enter_context(
            importlib.resources.as_file(
                importlib.resources.files("pymcap_cli.index").joinpath("datasette")
            )
        )
        argv = _build_datasette_argv(
            db_link,
            assets / "metadata.yaml",
            assets / "plugins",
            assets / "templates",
            host,
            port,
            open_browser=not no_browser,
        )

        url = f"http://{host}:{port}/"
        ERR.print(f"Serving index [cyan]{db_path}[/] on [link={url}]{url}[/link]")
        ERR.print("Press Ctrl-C to stop.")
        try:
            # argv is sys.executable + fixed args + resolved paths, run without a shell.
            return subprocess.run(argv, check=False).returncode  # noqa: S603
        except KeyboardInterrupt:
            ERR.print("Stopping.")
            return 0
