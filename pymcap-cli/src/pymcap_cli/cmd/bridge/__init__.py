"""`bridge` command group — info / record / cat / tf / diag live Foxglove bridges.

This package's submodules are named ``info``, ``record``, ``cat``, ``tf``, ``diag`` —
identical to the function each module exports. To avoid shadowing the
submodules with the function names, the function names are NOT
re-exported here; callers (including tests that monkeypatch internals)
should import directly from the submodules.
"""

from cyclopts import App

from pymcap_cli.cmd.bridge.cat import cat as _cat
from pymcap_cli.cmd.bridge.diag import diag as _diag
from pymcap_cli.cmd.bridge.info import info as _info
from pymcap_cli.cmd.bridge.record import record as _record
from pymcap_cli.cmd.bridge.tf import tf as _tf

bridge_app = App(
    name="bridge",
    help="Inspect, record, or stream from a live Foxglove WebSocket bridge.",
    help_format="rich",
)
bridge_app.command(_info, name="info")
bridge_app.command(_record, name="record")
bridge_app.command(_cat, name="cat")
bridge_app.command(_tf, name="tf")
bridge_app.command(_diag, name="diag")

__all__ = ["bridge_app"]
