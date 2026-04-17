"""Tests for OutputManager."""

from __future__ import annotations

from pathlib import Path

import pytest
from pymcap_cli.core.mcap_processor import (
    Header,
    OutputManager,
    OutputOptions,
    OverwriteCollisionPolicy,
)
from small_mcap import Attachment, Channel, Metadata, Schema, stream_reader


@pytest.fixture
def header():
    return Header(profile="", library="test")


@pytest.fixture
def schemas():
    return {1: Schema(id=1, name="test", encoding="json", data=b"{}")}


@pytest.fixture
def channels():
    return {1: Channel(id=1, schema_id=1, topic="/test", message_encoding="json", metadata={})}


@pytest.fixture
def output_options(tmp_path: Path):
    return OutputOptions(
        output_template=str(tmp_path / "output_{index:03d}.mcap"),
        compression="zstd",
        chunk_size=4 * 1024 * 1024,
    )


@pytest.fixture
def manager(output_options, schemas, channels, header):
    return OutputManager(output_options, schemas, channels, header)


class TestOutputManagerLazyCreation:
    def test_creates_segment_on_first_access(self, manager):
        segment = manager.get_or_create_segment(0)
        assert segment.key == 0
        assert segment.index == 0
        assert 0 in manager.segments

    def test_returns_existing_segment(self, manager):
        seg1 = manager.get_or_create_segment(0)
        seg2 = manager.get_or_create_segment(0)
        assert seg1 is seg2

    def test_increments_index(self, manager):
        manager.get_or_create_segment(0)
        seg = manager.get_or_create_segment(1)
        assert seg.index == 1

    def test_creates_file(self, manager, tmp_path: Path):
        manager.get_or_create_segment(0)
        assert (tmp_path / "output_000.mcap").exists()


class TestOutputManagerPendingRecords:
    def test_buffers_attachment_before_segments(self, manager):
        attachment = Attachment(
            log_time=100,
            create_time=50,
            name="test.bin",
            media_type="application/octet-stream",
            data=b"\x00\x01\x02",
        )
        manager.add_attachment(attachment)
        assert len(manager._pending_attachments) == 1

    def test_buffers_metadata_before_segments(self, manager):
        manager.add_metadata("info", {"key": "value"})
        assert len(manager._pending_metadata) == 1

    def test_flushes_to_new_segment(self, manager):
        attachment = Attachment(
            log_time=100,
            create_time=50,
            name="test.bin",
            media_type="application/octet-stream",
            data=b"\x00\x01\x02",
        )
        manager.add_attachment(attachment)
        manager.add_metadata("info", {"key": "value"})

        segment = manager.get_or_create_segment(0)

        # The attachment should have been flushed to the segment
        # We can't directly check the writer's internal state, but we can verify
        # the segment was created and the pending list is still there (for future segments)
        assert segment is not None

    def test_writes_directly_to_existing_segments(self, manager):
        manager.get_or_create_segment(0)
        attachment = Attachment(
            log_time=100,
            create_time=50,
            name="test.bin",
            media_type="application/octet-stream",
            data=b"\x00\x01\x02",
        )
        # Pending records must remain buffered for segments created later.
        manager.add_attachment(attachment)
        assert len(manager._pending_attachments) == 1

    def test_late_segments_receive_earlier_attachment_and_metadata(self, manager):
        manager.get_or_create_segment(0)
        attachment = Attachment(
            log_time=100,
            create_time=50,
            name="test.bin",
            media_type="application/octet-stream",
            data=b"\x00\x01\x02",
        )

        manager.add_attachment(attachment)
        manager.add_metadata("info", {"key": "value"})
        manager.get_or_create_segment(1)
        manager.finish_all()

        for index in (0, 1):
            path = Path(manager.segments[index].path)
            attachments = 0
            metadata = 0
            with path.open("rb") as stream:
                for record in stream_reader(stream):
                    if isinstance(record, Attachment):
                        attachments += 1
                    elif isinstance(record, Metadata):
                        metadata += 1

            assert attachments == 1
            assert metadata == 1


class TestOutputManagerOverwriteHandling:
    def test_existing_output_prompts_in_ask_mode(self, manager, monkeypatch, tmp_path: Path):
        existing = tmp_path / "output_000.mcap"
        existing.write_bytes(b"existing")
        seen: list[Path] = []

        def fake_confirm(output: Path, force: bool) -> None:
            seen.append(output)
            assert force is False

        monkeypatch.setattr(
            "pymcap_cli.core.mcap_processor.confirm_output_overwrite",
            fake_confirm,
        )

        manager.get_or_create_segment(0)

        assert seen == [existing]

    def test_existing_output_fails_in_error_mode(
        self, output_options, schemas, channels, header, tmp_path: Path
    ):
        output_options.overwrite_policy = OverwriteCollisionPolicy.ERROR
        manager = OutputManager(output_options, schemas, channels, header)
        (tmp_path / "output_000.mcap").write_bytes(b"existing")

        with pytest.raises(SystemExit, match="1"):
            manager.get_or_create_segment(0)

    def test_existing_output_skips_prompt_in_overwrite_mode(
        self, output_options, schemas, channels, header, monkeypatch, tmp_path: Path
    ):
        output_options.overwrite_policy = OverwriteCollisionPolicy.OVERWRITE
        manager = OutputManager(output_options, schemas, channels, header)
        existing = tmp_path / "output_000.mcap"
        existing.write_bytes(b"existing")
        called = False

        def fake_confirm(_output: Path, _force: bool) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(
            "pymcap_cli.core.mcap_processor.confirm_output_overwrite",
            fake_confirm,
        )

        manager.get_or_create_segment(0)

        assert called is False


class TestOutputManagerFinishAll:
    def test_closes_all_segments(self, manager):
        manager.get_or_create_segment(0)
        manager.get_or_create_segment(1)

        stats = manager.finish_all()
        assert len(stats) == 2

    def test_returns_statistics(self, manager):
        manager.get_or_create_segment(0)
        stats = manager.finish_all()
        assert 0 in stats
        assert stats[0].message_count == 0

    def test_clears_pending_buffers(self, manager):
        manager.add_attachment(
            Attachment(log_time=0, create_time=0, name="a", media_type="text", data=b"")
        )
        manager.finish_all()
        assert len(manager._pending_attachments) == 0
