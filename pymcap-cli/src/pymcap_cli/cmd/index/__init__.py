"""``pymcap-cli index`` — sidecar catalog of MCAP summaries.

Each subcommand lives in its own module; this file is just the
cyclopts assembler. Shared helpers and constants live in
:mod:`pymcap_cli.cmd.index._helpers`.
"""

from __future__ import annotations

import logging

from cyclopts import App

from pymcap_cli.cmd.index.duplicates_cmd import duplicates_cmd
from pymcap_cli.cmd.index.errors_cmd import errors_cmd
from pymcap_cli.cmd.index.info_cmd import info_cmd
from pymcap_cli.cmd.index.query_cmd import query_cmd
from pymcap_cli.cmd.index.scan_cmd import scan_cmd
from pymcap_cli.cmd.index.schemas_cmd import schemas_cmd
from pymcap_cli.cmd.index.sessions_cmd import sessions_cmd
from pymcap_cli.cmd.index.status_cmd import status_cmd
from pymcap_cli.cmd.index.timeline_cmd import timeline_cmd
from pymcap_cli.cmd.index.topics_cmd import topics_cmd
from pymcap_cli.cmd.index.tree_cmd import tree_cmd

logger = logging.getLogger(__name__)

index_app = App(
    name="index",
    help="Maintain a sidecar SQLite catalog of MCAP summaries for fast recovery.",
)

index_app.command(scan_cmd, name="scan")
index_app.command(status_cmd, name="status")
index_app.command(tree_cmd, name="tree")
index_app.command(query_cmd, name="query")
index_app.command(topics_cmd, name="topics")
index_app.command(schemas_cmd, name="schemas")
index_app.command(duplicates_cmd, name="duplicates")
index_app.command(sessions_cmd, name="sessions")
index_app.command(errors_cmd, name="errors")
index_app.command(timeline_cmd, name="timeline")
index_app.command(info_cmd, name="info")
