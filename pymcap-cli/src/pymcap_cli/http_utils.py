"""HTTP utilities for fetching MCAP files from URLs with Range request support."""

import io
from http import HTTPStatus
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult
from urllib.request import Request, urlopen


def get_http_file_size(url: str) -> int:
    """
    Get the size of a remote file using a HEAD request.

    Args:
        url: The URL to check

    Returns:
        The file size in bytes

    Raises:
        ValueError: If Content-Length header is missing
        HTTPError: If the server returns an error status
        URLError: If there's a network error
    """
    req = Request(url, method="HEAD")  # noqa: S310

    try:
        with urlopen(req) as response:  # noqa: S310
            content_length = response.headers.get("Content-Length")
            if content_length is None:
                raise ValueError(
                    f"Server does not provide Content-Length header for {url}. "
                    "HTTP URL support requires Content-Length."
                )
            return int(content_length)
    except HTTPError as e:
        raise HTTPError(
            url, e.code, f"HTTP {e.code} error for {url}: {e.reason}", e.hdrs, e.fp
        ) from e
    except URLError as e:
        raise URLError(f"Failed to connect to {url}: {e.reason}") from e


class HTTPRangeStream(io.RawIOBase):
    """
    A seekable stream that fetches chunks from an HTTP server using Range requests.

    This allows treating remote files as seekable file-like objects without downloading
    the entire file upfront.
    """

    def __init__(self, url: str, size: int) -> None:
        """
        Initialize the HTTP Range stream.

        Args:
            url: The URL to fetch from
            size: The total size of the remote file

        Raises:
            ValueError: If the server doesn't support Range requests
        """
        super().__init__()
        self.url = url
        self.size = size
        self._position = 0
        self._closed = False

        # Verify Range request support with a small initial request
        self._verify_range_support()

    def _verify_range_support(self) -> None:
        """Verify that the server supports Range requests."""
        req = Request(self.url)  # noqa: S310
        req.add_header("Range", "bytes=0-0")

        try:
            with urlopen(req) as response:  # noqa: S310
                accept_ranges = response.headers.get("Accept-Ranges")
                content_range = response.headers.get("Content-Range")

                # Check if server supports ranges (either via Accept-Ranges or by
                # honoring the Range request)
                if accept_ranges == "none":
                    raise ValueError(
                        f"Server explicitly does not support Range requests for {self.url}. "
                        "HTTP URL support requires Range request support."
                    )

                # If we got a 206 Partial Content, server supports ranges
                if response.status != 206 and content_range is None:
                    raise ValueError(
                        f"Server does not support Range requests for {self.url}. "
                        "HTTP URL support requires Range request support."
                    )
        except HTTPError as e:
            if (
                e.code == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
            ):  # Range Not Satisfiable - this is OK, means ranges are supported
                return
            raise ValueError(
                f"Failed to verify Range request support for {self.url}: HTTP {e.code}"
            ) from e

    def read(self, size: int = -1) -> bytes:
        """
        Read bytes from the current position.

        Args:
            size: Number of bytes to read, or -1 to read to end

        Returns:
            The bytes read
        """
        if self._closed:
            raise ValueError("I/O operation on closed file")

        if size == -1:
            size = self.size - self._position

        if size <= 0 or self._position >= self.size:
            return b""

        # Don't read past end of file
        size = min(size, self.size - self._position)

        end_pos = self._position + size - 1
        req = Request(self.url)  # noqa: S310
        req.add_header("Range", f"bytes={self._position}-{end_pos}")

        try:
            with urlopen(req) as response:  # noqa: S310
                data: bytes = response.read()
                self._position += len(data)
                return data
        except HTTPError as e:
            raise OSError(f"HTTP error while reading from {self.url}: {e}") from e
        except URLError as e:
            raise OSError(f"Network error while reading from {self.url}: {e}") from e

    def readinto(self, b: bytearray) -> int:  # type: ignore[override]
        """
        Read bytes into a pre-allocated buffer.

        Args:
            b: Buffer to read into

        Returns:
            Number of bytes read
        """
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """
        Seek to a position in the stream.

        Args:
            offset: Offset to seek to
            whence: Reference point (SEEK_SET, SEEK_CUR, or SEEK_END)

        Returns:
            The new absolute position
        """
        if self._closed:
            raise ValueError("I/O operation on closed file")

        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._position + offset
        elif whence == io.SEEK_END:
            new_pos = self.size + offset
        else:
            raise ValueError(f"Invalid whence value: {whence}")

        if new_pos < 0:
            raise ValueError("Cannot seek to negative position")

        self._position = new_pos
        return self._position

    def tell(self) -> int:
        """Return the current position in the stream."""
        if self._closed:
            raise ValueError("I/O operation on closed file")
        return self._position

    def seekable(self) -> bool:
        """Return whether the stream supports seeking."""
        return True

    def readable(self) -> bool:
        """Return whether the stream supports reading."""
        return True

    def writable(self) -> bool:
        """Return whether the stream supports writing (always False)."""
        return False

    def close(self) -> None:
        """Close the stream."""
        self._closed = True

    @property
    def closed(self) -> bool:
        """Return whether the stream is closed."""
        return self._closed


def open_http_stream(url: ParseResult) -> tuple[io.RawIOBase, int]:
    """
    Open an HTTP URL as a seekable stream using Range requests.

    Args:
        url: The URL to open

    Returns:
        A tuple of (stream, size) where stream is a seekable file-like object

    Raises:
        ValueError: If the server doesn't support Range requests or Content-Length
        HTTPError: If the server returns an error status
        URLError: If there's a network error
    """
    size = get_http_file_size(url.geturl())
    return HTTPRangeStream(url.geturl(), size), size
