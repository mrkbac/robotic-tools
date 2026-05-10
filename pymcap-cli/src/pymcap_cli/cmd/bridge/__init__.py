"""`bridge` command group — inspect / record / cat live Foxglove bridges.

This package's submodules are named ``inspect``, ``record``, ``cat`` —
identical to the function each module exports. To avoid shadowing the
submodules with the function names, the function names are NOT
re-exported here; callers (including tests that monkeypatch internals)
should import directly from the submodules.
"""

from cyclopts import App

from pymcap_cli.cmd.bridge.cat import cat as _cat
from pymcap_cli.cmd.bridge.inspect import inspect as _inspect
from pymcap_cli.cmd.bridge.record import record as _record

bridge_app = App(
    name="bridge",
    help="Inspect or record from a live Foxglove WebSocket bridge.",
    help_format="rich",
)
bridge_app.default(_inspect)
bridge_app.command(_record, name="record")
bridge_app.command(_cat, name="cat")

__all__ = ["bridge_app"]
