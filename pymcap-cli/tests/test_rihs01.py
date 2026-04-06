"""Tests for pymcap_cli.rihs01."""

import pytest
from pymcap_cli.core.msg_resolver import get_message_definition
from pymcap_cli.rihs01 import compute_rihs01


@pytest.mark.parametrize(
    ("schema_name", "expected_hash"),
    [
        (
            "std_msgs/msg/Byte",
            "RIHS01_41e1a3345f73fe93ede006da826a6ee274af23dd4653976ff249b0f44e3e798f",
        ),
        (
            "geometry_msgs/msg/Accel",
            "RIHS01_dc448243ded9b1fcbcca24aba0c22f013dae06c354ba2d849571c0a2a3f57ca0",
        ),
        (
            "std_msgs/msg/ByteMultiArray",
            "RIHS01_972fec7f50ab3c1d06783c228e79e8a9a509021708c511c059926261ada901d4",
        ),
    ],
)
def test_rihs01(schema_name: str, expected_hash: str) -> None:
    msg_def = get_message_definition(schema_name)
    assert compute_rihs01(schema_name, msg_def.encode()) == expected_hash
