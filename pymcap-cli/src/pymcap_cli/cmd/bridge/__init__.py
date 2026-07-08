"""`bridge` command group — info / record / cat / tf / diag live Foxglove bridges.

This package's submodules are named ``info``, ``record``, ``cat``, ``tf``, ``diag`` —
identical to the function each module exports. To avoid shadowing the
submodules with the function names, the function names are NOT
re-exported here; callers (including tests that monkeypatch internals)
should import directly from the submodules.
"""

from cyclopts import App

from pymcap_cli.cmd.bridge.call import call as _call
from pymcap_cli.cmd.bridge.cat import cat as _cat
from pymcap_cli.cmd.bridge.delay import delay as _delay
from pymcap_cli.cmd.bridge.diag import diag as _diag
from pymcap_cli.cmd.bridge.fetch import fetch as _fetch
from pymcap_cli.cmd.bridge.info import info as _info
from pymcap_cli.cmd.bridge.params import params as _params
from pymcap_cli.cmd.bridge.pub import pub as _pub
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
bridge_app.command(_delay, name="delay")
bridge_app.command(_tf, name="tf")
bridge_app.command(_diag, name="diag")
bridge_app.command(_call, name="call")
bridge_app.command(_pub, name="pub")
bridge_app.command(_params, name="params")
bridge_app.command(_fetch, name="fetch")

__all__ = ["bridge_app"]
