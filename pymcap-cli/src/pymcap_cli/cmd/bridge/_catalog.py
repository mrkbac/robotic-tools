"""Lichtblick/Digitalis v1 ``catalog.json`` with ranged MCAP downloads."""

from __future__ import annotations

import json
import re
import socket
import threading
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote, urlsplit

if TYPE_CHECKING:
    from socketserver import BaseServer

    from pymcap_cli.cmd.bridge._library import JsonValue, RecordingLibrary

_FILES_PREFIX = "/files/"
_CHUNK_SIZE = 1 << 20
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def ns_to_rfc3339(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    date = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{date}.{nanoseconds:09d}Z"


def create_catalog(library: RecordingLibrary) -> dict[str, JsonValue]:
    """Build a v1 catalog of every single-file recording in ``library``."""
    recordings: list[JsonValue] = []
    details = library.recording_details()
    for entry in library.recordings():
        # ponytail: rosbag2 split directories are skipped; a remote-file source is one MCAP.
        if library.resolve([entry.path])[0].is_dir():
            continue
        recording: dict[str, JsonValue] = {
            "id": entry.path,
            "source": {"type": "remote-file", "url": _FILES_PREFIX[1:] + quote(entry.path)},
            "size": entry.size_bytes,
        }
        summary = details.get(entry.path)
        if summary is not None:
            recording["messages"] = summary.message_count
            if summary.message_count:
                recording["start"] = ns_to_rfc3339(summary.start_time_ns)
                recording["end"] = ns_to_rfc3339(summary.end_time_ns)
        recordings.append(recording)
    return {"version": 1, "name": library.root.name, "recordings": recordings}


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Return the inclusive byte range for a single-range header, or None to send everything.

    Raises ValueError when the range cannot be satisfied.
    """
    match = _RANGE_RE.fullmatch(header.strip())
    if match is None or match.group(1) == match.group(2) == "":
        return None
    first, last = match.group(1), match.group(2)
    if first == "":
        length = int(last)
        if length == 0 or size == 0:
            raise ValueError("unsatisfiable range")
        return max(size - length, 0), size - 1
    start = int(first)
    end = size - 1 if last == "" else min(int(last), size - 1)
    if last != "" and int(last) < start:
        return None
    if start >= size:
        raise ValueError("unsatisfiable range")
    return start, end


class _CatalogHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        request: socket.socket,
        client_address: tuple[str, int],
        server: BaseServer,
        *,
        library: RecordingLibrary,
    ) -> None:
        self.library = library
        super().__init__(request, client_address, server)

    def log_message(self, format: str, *args: str) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Expose-Headers", "Accept-Ranges, Content-Length, Content-Range"
        )
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:
        self._handle(send_body=False)

    def do_GET(self) -> None:
        self._handle(send_body=True)

    def _handle(self, *, send_body: bool) -> None:
        path = urlsplit(self.path).path
        try:
            if path == "/catalog.json":
                self._send_json(create_catalog(self.library), send_body=send_body)
            elif path.startswith(_FILES_PREFIX):
                self._send_file(unquote(path.removeprefix(_FILES_PREFIX)), send_body=send_body)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Digitalis aborts its initial whole-file GET once headers arrive.

    def _send_json(self, body: dict[str, JsonValue], *, send_body: bool) -> None:
        payload = json.dumps(body).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(payload)

    def _send_file(self, logical_path: str, *, send_body: bool) -> None:
        try:
            path = self.library.resolve([logical_path])[0]
        except ValueError:
            path = None
        if path is None or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        with path.open("rb") as file:
            size = path.stat().st_size
            try:
                byte_range = parse_range(self.headers.get("Range", ""), size)
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = (0, size - 1) if byte_range is None else byte_range
            self.send_response(HTTPStatus.OK if byte_range is None else HTTPStatus.PARTIAL_CONTENT)
            if byte_range is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            if not send_body:
                return
            file.seek(start)
            remaining = end - start + 1
            while remaining > 0 and (chunk := file.read(min(remaining, _CHUNK_SIZE))):
                self.wfile.write(chunk)
                remaining -= len(chunk)


class CatalogServer(ThreadingHTTPServer):
    """Serve ``/catalog.json`` and ``/files/<path>`` from a background thread."""

    def __init__(self, library: RecordingLibrary, host: str, port: int) -> None:
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        super().__init__((host, port), partial(_CatalogHandler, library=library))
        self.port = int(self.server_address[1])
        threading.Thread(target=self.serve_forever, daemon=True).start()
