"""Custom exceptions for Digitalis."""


class DigitalisError(Exception):
    """Base exception for Digitalis-related errors."""


class ReaderError(DigitalisError):
    """Base exception for data reader errors."""


class ChannelNotFoundError(ReaderError):
    """Exception raised when a channel is not found."""


class MessageNotFoundError(ReaderError):
    """Exception raised when no message is found at the specified timestamp."""


class InvalidFileFormatError(ReaderError):
    """Exception raised when file format is invalid or corrupted."""
