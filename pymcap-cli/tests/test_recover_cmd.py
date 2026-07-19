"""Exit-code contract for `pymcap-cli recover`.

0 = full recovery, 3 = recovered but the input was truncated/corrupt so data was lost,
1 = nothing recoverable.
"""

from pathlib import Path

import small_mcap
from pymcap_cli.cmd import recover_cmd


def _message_stream(path: Path) -> list[tuple[str, int, int, bytes]]:
    """Ordered (topic, log_time, publish_time, data) for every message — an oracle read
    independently of the recover code under test."""
    with path.open("rb") as stream:
        return [
            (channel.topic, message.log_time, message.publish_time, bytes(message.data))
            for _schema, channel, message in small_mcap.read_message(stream)
        ]


def test_recover_always_decode_preserves_every_message(simple_mcap: Path, tmp_path: Path):
    # --always-decode-chunk decodes and re-encodes every chunk (the parallel worker path);
    # the recovered output must carry byte-identical messages in the same order.
    output = tmp_path / "out.mcap"
    exit_code = recover_cmd.recover(
        file=str(simple_mcap), output=output, always_decode_chunk=True, force=True
    )
    assert exit_code == 0
    assert _message_stream(output) == _message_stream(simple_mcap)


def test_recover_full_input_returns_zero(simple_mcap: Path, tmp_path: Path):
    output = tmp_path / "out.mcap"
    assert recover_cmd.recover(file=str(simple_mcap), output=output, force=True) == 0


def test_recover_truncated_input_returns_three(truncated_mcap: Path, tmp_path: Path):
    output = tmp_path / "out.mcap"
    assert recover_cmd.recover(file=str(truncated_mcap), output=output, force=True) == 3


def test_recover_unrecoverable_input_returns_one(tmp_path: Path):
    corrupt = tmp_path / "corrupt.mcap"
    corrupt.write_bytes(b"\x89MCAP0\r\n" + b"\x00" * 8)  # magic only, no recoverable records
    output = tmp_path / "out.mcap"
    assert recover_cmd.recover(file=str(corrupt), output=output, force=True) == 1
