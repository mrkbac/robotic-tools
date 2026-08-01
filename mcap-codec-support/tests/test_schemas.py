import pytest
from mcap_codec_support._schemas import normalize_schema_name


@pytest.mark.parametrize("kind", ["msg", "srv", "action"])
def test_normalize_schema_name_removes_ros2_kind(kind: str) -> None:
    assert normalize_schema_name(f"example_interfaces/{kind}/Example") == (
        "example_interfaces/Example"
    )


@pytest.mark.parametrize(
    "name",
    [
        "sensor_msgs/Image",
        "sensor_msgs/idl/Image",
        "Image",
        "sensor_msgs/msg/nested/Image",
    ],
)
def test_normalize_schema_name_preserves_other_shapes(name: str) -> None:
    assert normalize_schema_name(name) == name
