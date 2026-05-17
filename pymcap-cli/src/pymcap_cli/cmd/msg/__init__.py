"""Grouped ``msg`` subcommands: resolve, list, and browse ROS2 messages."""

from __future__ import annotations

from cyclopts import App

from pymcap_cli.cmd.msg import def_cmd, list_cmd, serve_cmd

msg_app = App(
    name="msg",
    help="Resolve, list, and browse ROS2 message definitions.",
)

msg_app.command(def_cmd.msg_def, name="def")
msg_app.command(list_cmd.msg_list, name="list")
msg_app.command(serve_cmd.msg_serve, name="serve")
