"""Lichtblick/Digitalis v1 ``catalog.json`` with ranged MCAP downloads."""

from __future__ import annotations

import json
import re
import socket
import threading
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote, urlsplit

from pymcap_cli.cmd.bridge._catalog_index import CatalogIndex

if TYPE_CHECKING:
    from pathlib import Path
    from socketserver import BaseServer

    from pymcap_cli.cmd.bridge._library import JsonValue, RecordingLibrary

_FILES_PREFIX = "/files/"
_CHUNK_SIZE = 1 << 20
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_PREVIEW_RE = re.compile(r"/previews/([0-9a-f]{32}/\d+/\d{3}\.jpg)")


def create_catalog(library: RecordingLibrary, index: CatalogIndex) -> dict[str, JsonValue]:
    """Build a v1 catalog of every single-file recording in ``library``.

    Recordings not indexed yet are listed with their basic fields and indexed in the background.
    """
    recordings: list[JsonValue] = []
    for entry in library.recordings():
        path = library.resolve([entry.path])[0]
        # ponytail: rosbag2 split directories are skipped; a remote-file source is one MCAP.
        if path.is_dir():
            continue
        relative = PurePosixPath(entry.path)
        recording: dict[str, JsonValue] = {
            "id": entry.path,
            "name": relative.stem,
            "source": {"type": "remote-file", "url": _FILES_PREFIX[1:] + quote(entry.path)},
            "size": entry.size_bytes,
        }
        directories = relative.parent.parts
        if directories:
            recording["dataset"] = directories[0]
        if len(directories) > 1:
            recording["session"] = "/".join(directories[1:])
        recording.update(index.get(path, entry.path) or {})
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
        index: CatalogIndex,
    ) -> None:
        self.library = library
        self.index = index
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
                self._send_json(create_catalog(self.library, self.index), send_body=send_body)
            elif preview := _PREVIEW_RE.fullmatch(path):
                self._send_preview(self.index.cache_dir / preview.group(1), send_body=send_body)
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

    def _send_preview(self, path: Path, *, send_body: bool) -> None:
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        # The cache key covers the file's size and mtime, so a frame URL never changes content.
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
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
    """Serve ``/catalog.json``, ``/files/<path>`` and ``/previews/...`` from a background thread.

    Indexing every recording starts right away; ``catalog.json`` lists what is ready.
    """

    def __init__(self, library: RecordingLibrary, host: str, port: int, cache_dir: Path) -> None:
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        self.index = CatalogIndex(cache_dir)
        super().__init__((host, port), partial(_CatalogHandler, library=library, index=self.index))
        self.port = int(self.server_address[1])
        threading.Thread(target=create_catalog, args=(library, self.index), daemon=True).start()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def server_close(self) -> None:
        super().server_close()
        self.index.close()
