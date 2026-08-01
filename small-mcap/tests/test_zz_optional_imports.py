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

    original_import = builtins.__import__

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
