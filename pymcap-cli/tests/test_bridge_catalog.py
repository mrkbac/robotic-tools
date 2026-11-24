"""Tests for the Digitalis ``catalog.json`` served next to ``bridge serve``."""

from __future__ import annotations

import http.client
import json
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

import pytest
from pymcap_cli.cmd.bridge._catalog import CatalogServer, ns_to_rfc3339, parse_range
from pymcap_cli.cmd.bridge._library import RecordingLibrary
from small_mcap import McapWriter

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

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


@pytest.fixture
def server(tmp_path: Path) -> Iterator[CatalogServer]:
    _write_mcap(tmp_path / "example.mcap", "/CAM_FRONT/image_rect_compressed", [1_000_000_005])
    _write_mcap(
        tmp_path / "scene" / "example 2.mcap",
        "/LIDAR_TOP",
        [1_700_000_000_000_000_000, 1_700_000_001_500_000_000],
    )
    catalog_server = CatalogServer(RecordingLibrary(tmp_path), "127.0.0.1", 0)
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
    response, body = _request(server, "/catalog.json")

    assert response.status == 200
    assert response.getheader("Access-Control-Allow-Origin") == "*"
    assert json.loads(body) == {
        "version": 1,
        "name": tmp_path.name,
        "recordings": [
            {
                "id": "example.mcap",
                "source": {"type": "remote-file", "url": "files/example.mcap"},
                "size": (tmp_path / "example.mcap").stat().st_size,
                "messages": 1,
                "start": "1970-01-01T00:00:01.000000005Z",
                "end": "1970-01-01T00:00:01.000000005Z",
            },
            {
                "id": "scene/example 2.mcap",
                "source": {"type": "remote-file", "url": "files/scene/example%202.mcap"},
                "size": (tmp_path / "scene" / "example 2.mcap").stat().st_size,
                "messages": 2,
                "start": "2023-11-14T22:13:20.000000000Z",
                "end": "2023-11-14T22:13:21.500000000Z",
            },
        ],
    }


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
    assert data == (tmp_path / "scene" / "example 2.mcap").read_bytes()


def test_catalog_server_range_request_returns_partial_content(
    server: CatalogServer, tmp_path: Path
) -> None:
    expected = (tmp_path / "example.mcap").read_bytes()

    response, data = _request(server, "/files/example.mcap", headers={"Range": "bytes=8-23"})

    assert response.status == 206
    assert data == expected[8:24]
    assert response.getheader("Content-Range") == f"bytes 8-23/{len(expected)}"
    assert response.getheader("Content-Length") == "16"
    assert response.getheader("Accept-Ranges") == "bytes"
    assert response.getheader("Access-Control-Allow-Origin") == "*"
    assert response.getheader("Access-Control-Expose-Headers") == _EXPOSED


def test_catalog_server_rejects_unsatisfiable_range(server: CatalogServer, tmp_path: Path) -> None:
    size = (tmp_path / "example.mcap").stat().st_size

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
    "path", ["/files/../outside.mcap", "/files/%2E%2E/outside.mcap", "/files/missing.mcap", "/"]
)
def test_catalog_server_does_not_serve_unknown_paths(
    server: CatalogServer, tmp_path: Path, path: str
) -> None:
    (tmp_path.parent / "outside.mcap").write_bytes(b"secret")

    response, _ = _request(server, path)

    assert response.status == 404
