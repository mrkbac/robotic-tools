"""Tests for the base package boundary around compression backends."""

import subprocess
import sys
import textwrap


def test_package_import_does_not_require_compression_backends() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCompressionModules(importlib.abc.MetaPathFinder):
            blocked = ("lz4", "zstandard")

            def find_spec(self, fullname, path=None, target=None):
                if any(
                    fullname == name or fullname.startswith(f"{name}.")
                    for name in self.blocked
                ):
                    root_name = fullname.partition(".")[0]
                    raise ModuleNotFoundError(
                        f"No module named '{root_name}'", name=root_name
                    )
                return None

        sys.meta_path.insert(0, BlockCompressionModules())

        import small_mcap

        assert small_mcap.CompressionType.NONE.value == ""
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
