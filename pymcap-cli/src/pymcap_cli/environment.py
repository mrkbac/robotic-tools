"""Canonical pymcap-cli environment-variable names and legacy fallbacks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cyclopts import Token

PYMCAP_ENV_PREFIX = "PYMCAP_"
BRIDGE_TARGET_ENV = f"{PYMCAP_ENV_PREFIX}BRIDGE"
COMPRESSION_WORKERS_ENV = f"{PYMCAP_ENV_PREFIX}COMPRESSION_WORKERS"
MESSAGE_PATH_VARIABLE_ENV_PREFIX = f"{PYMCAP_ENV_PREFIX}VAR_"
VIDEO_DECODE_WORKERS_ENV = f"{PYMCAP_ENV_PREFIX}VIDEO_DECODE_WORKERS"

LEGACY_COMPRESSION_WORKERS_ENV = "MCAP_COMPRESS_WORKERS"
LEGACY_VIDEO_DECODE_WORKERS_ENV = "VC_DECODE"

COMPRESSION_WORKERS_ENVS = (COMPRESSION_WORKERS_ENV, LEGACY_COMPRESSION_WORKERS_ENV)
VIDEO_DECODE_WORKERS_ENVS = (VIDEO_DECODE_WORKERS_ENV, LEGACY_VIDEO_DECODE_WORKERS_ENV)

logger = logging.getLogger(__name__)

_DEPRECATED_ENV_REPLACEMENTS = {
    LEGACY_COMPRESSION_WORKERS_ENV: (COMPRESSION_WORKERS_ENV, "--compression-workers"),
    LEGACY_VIDEO_DECODE_WORKERS_ENV: (VIDEO_DECODE_WORKERS_ENV, "--video-decode-workers"),
}


def int_with_deprecated_env_warning(_type: type, tokens: Sequence[Token]) -> int:
    """Convert one integer token and warn when it came from a legacy env name."""
    if len(tokens) != 1:
        raise ValueError("Expected exactly one integer.")
    token = tokens[0]
    if token.keyword in _DEPRECATED_ENV_REPLACEMENTS:
        replacement_env, replacement_option = _DEPRECATED_ENV_REPLACEMENTS[token.keyword]
        logger.warning(
            "%s is deprecated; use %s or %s instead.",
            token.keyword,
            replacement_env,
            replacement_option,
        )
    return int(token.value)
