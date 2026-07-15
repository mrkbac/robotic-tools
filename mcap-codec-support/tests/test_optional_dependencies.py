"""Tests for the base package boundary around optional codec dependencies."""

import subprocess
import sys
import textwrap


def test_package_import_does_not_load_optional_dependencies() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockOptionalModules(importlib.abc.MetaPathFinder):
            blocked = (
                "av",
                "DracoPy",
                "mcap_ros2_support_fast",
                "numpy",
                "PIL",
                "pointcloud2",
                "pureini",
            )

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

        sys.meta_path.insert(0, BlockOptionalModules())

        import mcap_codec_support
        from mcap_codec_support._schemas import normalize_schema_name

        assert mcap_codec_support.__all__ == ["create_decoder_factories"]
        assert normalize_schema_name("sensor_msgs/msg/Image") == "sensor_msgs/Image"
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
