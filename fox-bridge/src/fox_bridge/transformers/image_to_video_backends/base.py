"""Shared primitives for image-to-video transformer backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EncodingConfig:
    """Codec configuration shared by all backends."""

    codec: str
    quality: int
    preset: str
    use_hardware: bool
    timeout_s: float = 30.0


class ImageToVideoBackendError(RuntimeError):
    """Raised when a backend fails to encode a frame."""


class ImageToVideoBackend(ABC):
    """Backend interface for encoding a single JPEG frame into video bitstreams."""

    def __init__(self, config: EncodingConfig) -> None:
        self._config = config

    @property
    def config(self) -> EncodingConfig:
        """Return the encoding configuration."""
        return self._config

    @abstractmethod
    def encode(
        self,
        jpeg_data: bytes,
        input_width: int,
        input_height: int,
        target_width: int,
        target_height: int,
    ) -> bytes:
        """Encode a single frame.

        Args:
            jpeg_data: Compressed JPEG bytes.
            input_width: Width of the JPEG image.
            input_height: Height of the JPEG image.
            target_width: Desired output width after scaling.
            target_height: Desired output height after scaling.

        Returns:
            Encoded video bitstream in Annex-B (H.264/H.265) or equivalent raw format.

        Raises:
            ImageToVideoBackendError: When encoding fails or times out.
        """

