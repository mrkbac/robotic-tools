"""Tests for pymcap_cli.http_utils module."""

import io
from http.client import HTTPMessage
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest
from pymcap_cli.http_utils import HTTPRangeStream, get_http_file_size, open_http_stream

URL = "http://example.com/file.mcap"


def _make_response(
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    data: bytes = b"",
) -> MagicMock:
    """Create a mock HTTP response that works as a context manager."""
    resp = MagicMock()
    resp.status = status

    msg = HTTPMessage()
    for k, v in (headers or {}).items():
        msg[k] = v
    resp.headers = msg

    resp.read.return_value = data
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# get_http_file_size
# ---------------------------------------------------------------------------


class TestGetHttpFileSize:
    @patch("pymcap_cli.http_utils.urlopen")
    def test_returns_content_length(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_response(headers={"Content-Length": "42000"})
        assert get_http_file_size(URL) == 42000
        # Verify HEAD request was used
        req = mock_urlopen.call_args[0][0]
        assert req.get_method() == "HEAD"

    @patch("pymcap_cli.http_utils.urlopen")
    def test_raises_on_missing_content_length(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_response(headers={})
        with pytest.raises(ValueError, match="Content-Length"):
            get_http_file_size(URL)

    @patch("pymcap_cli.http_utils.urlopen")
    def test_raises_on_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = HTTPError(URL, 404, "Not Found", HTTPMessage(), None)
        with pytest.raises(HTTPError, match="404"):
            get_http_file_size(URL)

    @patch("pymcap_cli.http_utils.urlopen")
    def test_raises_on_url_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = URLError("Connection refused")
        with pytest.raises(URLError, match="Failed to connect"):
            get_http_file_size(URL)


# ---------------------------------------------------------------------------
# HTTPRangeStream — helper to build one without hitting the network
# ---------------------------------------------------------------------------


def _make_stream(size: int = 1000) -> HTTPRangeStream:
    """Create an HTTPRangeStream with mocked range-support verification."""
    with patch("pymcap_cli.http_utils.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _make_response(
            status=206,
            headers={"Content-Range": "bytes 0-0/1000"},
            data=b"\x00",
        )
        return HTTPRangeStream(URL, size)


# ---------------------------------------------------------------------------
# HTTPRangeStream.read — verifies Range header construction
# ---------------------------------------------------------------------------


class TestHTTPRangeStreamRead:
    @patch("pymcap_cli.http_utils.urlopen")
    def test_read_sends_correct_range_header(self, mock_urlopen: MagicMock) -> None:
        stream = _make_stream(size=100)
        mock_urlopen.return_value = _make_response(data=b"hello")
        result = stream.read(5)
        assert result == b"hello"
        assert stream.tell() == 5
        # Verify the Range header
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Range") == "bytes=0-4"

    @patch("pymcap_cli.http_utils.urlopen")
    def test_read_advances_position_for_next_range(self, mock_urlopen: MagicMock) -> None:
        stream = _make_stream(size=100)
        mock_urlopen.return_value = _make_response(data=b"12345")
        stream.read(5)
        mock_urlopen.return_value = _make_response(data=b"67890")
        stream.read(5)
        # Second read should use updated position
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Range") == "bytes=5-9"
        assert stream.tell() == 10

    def test_read_past_eof_returns_empty(self) -> None:
        stream = _make_stream(size=100)
        stream.seek(100)
        result = stream.read(10)
        assert result == b""

    @patch("pymcap_cli.http_utils.urlopen")
    def test_read_clamps_range_to_file_size(self, mock_urlopen: MagicMock) -> None:
        stream = _make_stream(size=10)
        stream.seek(7)
        mock_urlopen.return_value = _make_response(data=b"abc")
        result = stream.read(100)
        assert result == b"abc"
        # Range should be clamped to bytes=7-9 (not 7-106)
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Range") == "bytes=7-9"

    @patch("pymcap_cli.http_utils.urlopen")
    def test_read_minus_one_reads_to_end(self, mock_urlopen: MagicMock) -> None:
        stream = _make_stream(size=5)
        mock_urlopen.return_value = _make_response(data=b"abcde")
        result = stream.read(-1)
        assert result == b"abcde"
        # Range should cover entire file
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Range") == "bytes=0-4"

    def test_read_zero_returns_empty(self) -> None:
        stream = _make_stream(size=100)
        result = stream.read(0)
        assert result == b""

    @patch("pymcap_cli.http_utils.urlopen")
    def test_read_http_error_raises_oserror(self, mock_urlopen: MagicMock) -> None:
        stream = _make_stream(size=100)
        mock_urlopen.side_effect = HTTPError(URL, 500, "ISE", HTTPMessage(), None)
        with pytest.raises(OSError, match="HTTP error"):
            stream.read(10)

    @patch("pymcap_cli.http_utils.urlopen")
    def test_read_url_error_raises_oserror(self, mock_urlopen: MagicMock) -> None:
        stream = _make_stream(size=100)
        mock_urlopen.side_effect = URLError("timeout")
        with pytest.raises(OSError, match="Network error"):
            stream.read(10)


# ---------------------------------------------------------------------------
# HTTPRangeStream.seek
# ---------------------------------------------------------------------------


class TestHTTPRangeStreamSeek:
    def test_seek_set(self) -> None:
        stream = _make_stream()
        assert stream.seek(42, io.SEEK_SET) == 42
        assert stream.tell() == 42

    def test_seek_cur(self) -> None:
        stream = _make_stream()
        stream.seek(10)
        assert stream.seek(5, io.SEEK_CUR) == 15

    def test_seek_end(self) -> None:
        stream = _make_stream(size=1000)
        assert stream.seek(-10, io.SEEK_END) == 990

    def test_seek_invalid_whence(self) -> None:
        stream = _make_stream()
        with pytest.raises(ValueError, match="Invalid whence"):
            stream.seek(0, 999)

    def test_seek_negative_position_raises(self) -> None:
        stream = _make_stream()
        with pytest.raises(ValueError, match="negative position"):
            stream.seek(-1, io.SEEK_SET)


# ---------------------------------------------------------------------------
# HTTPRangeStream.readinto
# ---------------------------------------------------------------------------


class TestHTTPRangeStreamReadinto:
    @patch("pymcap_cli.http_utils.urlopen")
    def test_readinto_fills_buffer(self, mock_urlopen: MagicMock) -> None:
        stream = _make_stream(size=100)
        mock_urlopen.return_value = _make_response(data=b"abcd")
        buf = bytearray(4)
        n = stream.readinto(buf)
        assert n == 4
        assert buf == b"abcd"

    @patch("pymcap_cli.http_utils.urlopen")
    def test_readinto_partial_fill(self, mock_urlopen: MagicMock) -> None:
        stream = _make_stream(size=2)
        mock_urlopen.return_value = _make_response(data=b"xy")
        buf = bytearray(10)
        n = stream.readinto(buf)
        assert n == 2
        assert buf[:2] == b"xy"


# ---------------------------------------------------------------------------
# Closed-stream operations
# ---------------------------------------------------------------------------


class TestHTTPRangeStreamClosed:
    def test_read_on_closed_raises(self) -> None:
        stream = _make_stream()
        stream.close()
        with pytest.raises(ValueError, match="closed"):
            stream.read(1)

    def test_seek_on_closed_raises(self) -> None:
        stream = _make_stream()
        stream.close()
        with pytest.raises(ValueError, match="closed"):
            stream.seek(0)

    def test_tell_on_closed_raises(self) -> None:
        stream = _make_stream()
        stream.close()
        with pytest.raises(ValueError, match="closed"):
            stream.tell()

    def test_closed_property(self) -> None:
        stream = _make_stream()
        assert not stream.closed
        stream.close()
        assert stream.closed


# ---------------------------------------------------------------------------
# _verify_range_support edge cases
# ---------------------------------------------------------------------------


class TestVerifyRangeSupport:
    @patch("pymcap_cli.http_utils.urlopen")
    def test_accept_ranges_none_raises(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_response(status=200, headers={"Accept-Ranges": "none"})
        with pytest.raises(ValueError, match="does not support Range"):
            HTTPRangeStream(URL, 100)

    @patch("pymcap_cli.http_utils.urlopen")
    def test_no_206_and_no_content_range_raises(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_response(status=200, headers={})
        with pytest.raises(ValueError, match="does not support Range"):
            HTTPRangeStream(URL, 100)

    @patch("pymcap_cli.http_utils.urlopen")
    def test_416_is_accepted(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = HTTPError(URL, 416, "Range Not Satisfiable", HTTPMessage(), None)
        # Should not raise
        stream = HTTPRangeStream(URL, 100)
        assert stream.size == 100

    @patch("pymcap_cli.http_utils.urlopen")
    def test_other_http_error_raises(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = HTTPError(URL, 403, "Forbidden", HTTPMessage(), None)
        with pytest.raises(ValueError, match="Failed to verify"):
            HTTPRangeStream(URL, 100)


# ---------------------------------------------------------------------------
# open_http_stream
# ---------------------------------------------------------------------------


class TestOpenHttpStream:
    @patch("pymcap_cli.http_utils.urlopen")
    def test_returns_stream_and_size(self, mock_urlopen: MagicMock) -> None:
        from urllib.parse import urlparse  # noqa: PLC0415

        # First call: HEAD for file size, second call: range verification
        mock_urlopen.side_effect = [
            _make_response(headers={"Content-Length": "500"}),
            _make_response(
                status=206,
                headers={"Content-Range": "bytes 0-0/500"},
                data=b"\x00",
            ),
        ]
        parsed = urlparse(URL)
        stream, size = open_http_stream(parsed)
        assert size == 500
        assert isinstance(stream, HTTPRangeStream)
        assert stream.tell() == 0
        stream.close()
