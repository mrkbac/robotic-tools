import logging
from urllib.parse import urlparse

from .mcap.source import McapSource
from .source import Source
from .websocket.source import WebSocketSource

logger = logging.getLogger(__name__)


def create_source(path_or_url: str) -> Source:
    """Create the appropriate source based on the input path or URL."""
    parsed = urlparse(path_or_url)
    logger.info(f"Creating source for: {path_or_url}")

    if parsed.scheme in ("ws", "wss"):
        return WebSocketSource(path_or_url)

    return McapSource(path_or_url)
