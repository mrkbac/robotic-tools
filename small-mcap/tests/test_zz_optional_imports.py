import builtins
import importlib
import sys
from types import ModuleType, SimpleNamespace

import small_mcap.reader as reader_module
import small_mcap.writer as writer_module


def test_optional_compression_import_paths(monkeypatch) -> None:
    writer_state = writer_module.__dict__.copy()
    reader_state = reader_module.__dict__.copy()
    compression = ModuleType("compression")
    compression.__path__ = []  # type: ignore[attr-defined]
    zstd = ModuleType("compression.zstd")
    zstd.compress = lambda data, **_kwargs: b"compressed:" + bytes(data)  # type: ignore[attr-defined]
    compression.zstd = SimpleNamespace(  # type: ignore[attr-defined]
        decompress=lambda data: b"decompressed:" + bytes(data)
    )

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "compression", compression)
        patch.setitem(sys.modules, "compression.zstd", zstd)
        stdlib_writer = importlib.reload(writer_module)
        stdlib_reader = importlib.reload(reader_module)

        assert stdlib_writer._zstd_compress(b"x") == b"compressed:x"
        assert stdlib_writer._zstd_compress(b"x", 5) == b"compressed:x"
        assert stdlib_reader._zstd_decompress(b"x", 1) == b"decompressed:x"

    class FakeZstdCompressor:
        def __init__(self, level: int | None = None) -> None:
            self.level = level

        def compress(self, data: bytes | memoryview) -> bytes:
            return f"compressed-{self.level}:".encode() + bytes(data)

    class FakeZstdDecompressor:
        def decompress(self, data: bytes | memoryview, *, max_output_size: int) -> bytes:
            return f"decompressed-{max_output_size}:".encode() + bytes(data)

    class FakeZstandard(ModuleType):
        ZstdCompressor = FakeZstdCompressor
        ZstdDecompressor = FakeZstdDecompressor

    original_import = builtins.__import__

    def blocked_stdlib_zstd(name, *args, **kwargs):
        if name in {"compression", "compression.zstd"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", blocked_stdlib_zstd)
        patch.setitem(sys.modules, "zstandard", FakeZstandard("zstandard"))
        third_party_writer = importlib.reload(writer_module)
        third_party_reader = importlib.reload(reader_module)

        assert third_party_writer._zstd_compress(b"x") == b"compressed-None:x"
        assert third_party_writer._zstd_compress(b"y") == b"compressed-None:y"
        assert third_party_writer._zstd_compress(b"x", 5) == b"compressed-5:x"
        assert third_party_reader._zstd_decompress(b"x", 4) == b"decompressed-4:x"
        assert third_party_reader._zstd_decompress(b"y", 4) == b"decompressed-4:y"

    def blocked_import(name, *args, **kwargs):
        if name in {"compression", "compression.zstd", "lz4.frame", "zstandard"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", blocked_import)
        no_compression_writer = importlib.reload(writer_module)
        no_compression_reader = importlib.reload(reader_module)

        assert no_compression_writer.lz4_compress is None
        assert no_compression_writer._zstd_compress is None
        assert no_compression_reader.lz4_decompress is None
        assert no_compression_reader._zstd_decompress is None

    writer_module.__dict__.clear()
    writer_module.__dict__.update(writer_state)
    reader_module.__dict__.clear()
    reader_module.__dict__.update(reader_state)
