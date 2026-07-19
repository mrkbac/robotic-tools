"""Exit-code contract for `pymcap-cli recover`.

0 = full recovery, 3 = recovered but the input was truncated/corrupt so data was lost,
1 = nothing recoverable.
"""

from pathlib import Path

from pymcap_cli.cmd import recover_cmd


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
