import logging
from collections.abc import Callable
from urllib.parse import urlparse

from .mcap.source import McapSource
from .source import Source, SourceStatus
from .types import MessageEvent, SourceInfo
from .websocket.source import WebSocketSource

logger = logging.getLogger(__name__)


def create_source(
    path_or_url: str,
    on_message: Callable[[MessageEvent], None],
    on_source_info: Callable[[SourceInfo], None],
    on_time: Callable[[int], None],
    on_status: Callable[[SourceStatus], None],
) -> Source:
    """Create the appropriate source based on the input path or URL."""
    parsed = urlparse(path_or_url)
    logger.info(f"Creating source for: {path_or_url}")

    if parsed.scheme in ("ws", "wss"):
        return WebSocketSource(
            path_or_url,
            on_message=on_message,
            on_source_info=on_source_info,
            on_time=on_time,
            on_status=on_status,
        )

    return McapSource(
        path_or_url,
        on_message=on_message,
        on_source_info=on_source_info,
        on_time=on_time,
        on_status=on_status,
    )
