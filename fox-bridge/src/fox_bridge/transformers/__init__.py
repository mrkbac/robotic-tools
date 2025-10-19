"""Message transformers for converting between ROS2 message types."""

from abc import ABC, abstractmethod
from typing import Any


class Transformer(ABC):
    """Base class for message transformers."""

    @abstractmethod
    def get_input_schema(self) -> str:
        """Get the input message schema name (e.g., 'sensor_msgs/CompressedImage')."""
        ...

    @abstractmethod
    def get_output_schema(self) -> str:
        """Get the output message schema name (e.g., 'foxglove_msgs/CompressedVideo')."""
        ...

    @abstractmethod
    def transform(self, message: dict[str, Any]) -> dict[str, Any]:
        """Transform a decoded message dict to another message dict.

        Args:
            message: Decoded input message as a dictionary

        Returns:
            Transformed output message as a dictionary

        Raises:
            TransformError: If transformation fails
        """
        ...

    def can_transform(self, schema_name: str) -> bool:
        """Check if this transformer can handle the given schema.

        Args:
            schema_name: The message schema name to check

        Returns:
            True if this transformer can handle this schema
        """
        return schema_name == self.get_input_schema()


class TransformError(Exception):
    """Raised when message transformation fails."""


class TransformerRegistry:
    """Registry for managing message transformers."""

    def __init__(self) -> None:
        self._transformers: list[Transformer] = []
        self._schema_map: dict[str, Transformer] = {}

    def register(self, transformer: Transformer) -> None:
        """Register a transformer.

        Args:
            transformer: The transformer to register
        """
        self._transformers.append(transformer)
        input_schema = transformer.get_input_schema()
        self._schema_map[input_schema] = transformer

    def get_transformer(self, schema_name: str) -> Transformer | None:
        """Get a transformer for the given schema.

        Args:
            schema_name: The input message schema name

        Returns:
            The transformer, or None if no transformer is registered
        """
        return self._schema_map.get(schema_name)

    def can_transform(self, schema_name: str) -> bool:
        """Check if any transformer can handle the given schema.

        Args:
            schema_name: The message schema name to check

        Returns:
            True if a transformer is registered for this schema
        """
        return schema_name in self._schema_map

    def get_all_transformers(self) -> list[Transformer]:
        """Get all registered transformers.

        Returns:
            List of all transformers
        """
        return self._transformers.copy()


__all__ = ["TransformError", "Transformer", "TransformerRegistry"]
