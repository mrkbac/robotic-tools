"""Tests for the Digitalis ``catalog.json`` served next to ``bridge serve``."""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO
from urllib.parse import urljoin, urlsplit

import pymcap_cli.cmd.bridge._catalog_index as catalog_index_module
import pytest
from pymcap_cli.cmd.bridge._catalog import CatalogServer, parse_range
from pymcap_cli.cmd.bridge._catalog_index import ns_to_rfc3339
from pymcap_cli.cmd.bridge._library import RecordingLibrary
from small_mcap import McapWriter

if TYPE_CHECKING:
    from collections.abc import Iterator

_EXPOSED = "Accept-Ranges, Content-Length, Content-Range"


def _write_mcap(path: Path, topic: str, timestamps_ns: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = McapWriter(stream)
        writer.start()
        writer.add_schema(1, "example/Raw", "text", b"bytes data")
        writer.add_channel(1, topic, "raw", 1)
        for timestamp_ns in timestamps_ns:
            writer.add_message(1, timestamp_ns, b"payload", publish_time=timestamp_ns)
        writer.finish()


def _create_server(tmp_path: Path) -> CatalogServer:
    root = tmp_path / "recordings"
    _write_mcap(root / "example.mcap", "/CAM_FRONT/image_rect_compressed", [1_000_000_005])
    _write_mcap(
        root / "scene" / "run 1" / "example 2.mcap",
        "/LIDAR_TOP",
        [1_700_000_000_000_000_000, 1_700_000_001_500_000_000],
    )
    return CatalogServer(RecordingLibrary(root), "127.0.0.1", 0, tmp_path / "cache")


@pytest.fixture
def server(tmp_path: Path) -> Iterator[CatalogServer]:
    catalog_server = _create_server(tmp_path)
    _request(catalog_server, "/catalog.json")  # schedules indexing like the startup scan
    catalog_server.index.wait()
    yield catalog_server
    catalog_server.shutdown()
    catalog_server.server_close()


def _request(
    server: CatalogServer,
    path: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> tuple[http.client.HTTPResponse, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response, body


def test_ns_to_rfc3339_keeps_nanoseconds() -> None:
    assert ns_to_rfc3339(1_700_000_001_500_000_007) == "2023-11-14T22:13:21.500000007Z"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("bytes=0-9", (0, 9)),
        ("bytes=5-", (5, 99)),
        ("bytes=90-200", (90, 99)),
        ("bytes=-10", (90, 99)),
        ("bytes=-500", (0, 99)),
        ("", None),
        ("bytes=9-0", None),
        ("bytes=0-1,4-5", None),
        ("items=0-9", None),
    ],
)
def test_parse_range_returns_inclusive_range(header: str, expected: tuple[int, int] | None) -> None:
    assert parse_range(header, 100) == expected


@pytest.mark.parametrize("header", ["bytes=100-", "bytes=-0"])
def test_parse_range_rejects_unsatisfiable(header: str) -> None:
    with pytest.raises(ValueError, match="unsatisfiable"):
        parse_range(header, 100)


def test_catalog_server_serves_v1_catalog(server: CatalogServer, tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    response, body = _request(server, "/catalog.json")

    assert response.status == 200
    assert response.getheader("Access-Control-Allow-Origin") == "*"
    assert json.loads(body) == {
        "version": 1,
        "name": "recordings",
        "recordings": [
            {
                "id": "example.mcap",
                "name": "example",
                "source": {"type": "remote-file", "url": "files/example.mcap"},
                "size": (root / "example.mcap").stat().st_size,
                "topics": [
                    {
                        "name": "/CAM_FRONT/image_rect_compressed",
                        "schema": "example/Raw",
                        "messages": 1,
                    }
                ],
                "messages": 1,
                "start": "1970-01-01T00:00:01.000000005Z",
                "end": "1970-01-01T00:00:01.000000005Z",
            },
            {
                "id": "scene/run 1/example 2.mcap",
                "name": "example 2",
                "source": {"type": "remote-file", "url": "files/scene/run%201/example%202.mcap"},
                "size": (root / "scene" / "run 1" / "example 2.mcap").stat().st_size,
                "dataset": "scene",
                "session": "run 1",
                "topics": [{"name": "/LIDAR_TOP", "schema": "example/Raw", "messages": 2}],
                "messages": 2,
                "start": "2023-11-14T22:13:20.000000000Z",
                "end": "2023-11-14T22:13:21.500000000Z",
            },
        ],
    }


def test_catalog_server_lists_recordings_while_indexing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    index_recording = catalog_index_module.index_recording

    def slow_index_recording(*args, **kwargs):
        release.wait(5)
        return index_recording(*args, **kwargs)

    monkeypatch.setattr(catalog_index_module, "index_recording", slow_index_recording)
    catalog_server = _create_server(tmp_path)
    try:
        _, body = _request(catalog_server, "/catalog.json")
        release.set()
        catalog_server.index.wait()
        _, indexed_body = _request(catalog_server, "/catalog.json")
    finally:
        release.set()
        catalog_server.shutdown()
        catalog_server.server_close()

    pending = json.loads(body)["recordings"]
    assert [recording["id"] for recording in pending] == [
        "example.mcap",
        "scene/run 1/example 2.mcap",
    ]
    assert all("messages" not in recording for recording in pending)
    assert all("messages" in recording for recording in json.loads(indexed_body)["recordings"])


def test_catalog_server_relative_url_resolves_to_full_file(
    server: CatalogServer, tmp_path: Path
) -> None:
    catalog_url = f"http://127.0.0.1:{server.port}/catalog.json"
    _, body = _request(server, "/catalog.json")
    source_url = json.loads(body)["recordings"][1]["source"]["url"]

    response, data = _request(server, urlsplit(urljoin(catalog_url, source_url)).path)

    assert response.status == 200
    assert response.getheader("Accept-Ranges") == "bytes"
    assert response.getheader("Content-Length") == str(len(data))
    assert data == (tmp_path / "recordings" / "scene" / "run 1" / "example 2.mcap").read_bytes()


def test_catalog_server_range_request_returns_partial_content(
    server: CatalogServer, tmp_path: Path
) -> None:
    expected = (tmp_path / "recordings" / "example.mcap").read_bytes()

    response, data = _request(server, "/files/example.mcap", headers={"Range": "bytes=8-23"})

    assert response.status == 206
    assert data == expected[8:24]
    assert response.getheader("Content-Range") == f"bytes 8-23/{len(expected)}"
    assert response.getheader("Content-Length") == "16"
    assert response.getheader("Accept-Ranges") == "bytes"
    assert response.getheader("Access-Control-Allow-Origin") == "*"
    assert response.getheader("Access-Control-Expose-Headers") == _EXPOSED


def test_catalog_server_sizes_the_file_it_opened_when_replaced_mid_request(
    server: CatalogServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = tmp_path / "recordings" / "example.mcap"
    original = recording.read_bytes()
    replacement = tmp_path / "replacement.mcap"
    replacement.write_bytes(original * 2)
    open_path = Path.open

    def open_then_replace(path: Path, *args: str) -> BinaryIO:
        file = open_path(path, *args)
        if path == recording:
            replacement.replace(recording)
        return file

    monkeypatch.setattr(Path, "open", open_then_replace)

    response, data = _request(server, "/files/example.mcap")

    assert response.status == 200
    assert response.getheader("Content-Length") == str(len(original))
    assert data == original


def test_catalog_server_rejects_unsatisfiable_range(server: CatalogServer, tmp_path: Path) -> None:
    size = (tmp_path / "recordings" / "example.mcap").stat().st_size

    response, _ = _request(server, "/files/example.mcap", headers={"Range": f"bytes={size}-"})

    assert response.status == 416
    assert response.getheader("Content-Range") == f"bytes */{size}"


def test_catalog_server_answers_range_preflight(server: CatalogServer) -> None:
    response, _ = _request(
        server,
        "/files/example.mcap",
        method="OPTIONS",
        headers={
            "Origin": "https://viewer.example.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "range",
        },
    )

    assert response.status == 204
    assert response.getheader("Access-Control-Allow-Origin") == "*"
    assert response.getheader("Access-Control-Allow-Headers") == "Range"
    assert "GET" in (response.getheader("Access-Control-Allow-Methods") or "")


@pytest.mark.parametrize(
    "path",
    [
        "/files/../outside.mcap",
        "/files/%2E%2E/outside.mcap",
        "/files/missing.mcap",
        "/",
        "/previews/../summary.json",
        "/previews/0123456789abcdef0123456789abcdef/0/000.jpg",
    ],
)
def test_catalog_server_does_not_serve_unknown_paths(
    server: CatalogServer, tmp_path: Path, path: str
) -> None:
    (tmp_path / "outside.mcap").write_bytes(b"secret")

    response, _ = _request(server, path)

    assert response.status == 404
