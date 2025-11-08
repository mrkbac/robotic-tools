"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest

from tests.fixtures.image_mcap_generator import ensure_image_fixtures
from tests.fixtures.mcap_generator import ensure_fixtures


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> Path:
    """Create and return the fixtures directory for the session."""
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def test_fixtures(fixtures_dir) -> dict[str, Path]:
    """Generate all test MCAP fixtures once per session."""
    return ensure_fixtures(fixtures_dir)


@pytest.fixture(scope="session")
def image_fixtures(fixtures_dir) -> dict[str, Path]:
    """Generate all image MCAP fixtures once per session."""
    return ensure_image_fixtures(fixtures_dir)


@pytest.fixture
def simple_mcap(test_fixtures) -> Path:
    """Simple MCAP file with one topic."""
    return test_fixtures["simple"]


@pytest.fixture
def multi_topic_mcap(test_fixtures) -> Path:
    """MCAP file with multiple topics."""
    return test_fixtures["multi_topic"]


@pytest.fixture
def truncated_mcap(test_fixtures) -> Path:
    """Corrupt MCAP file (truncated)."""
    return test_fixtures["truncated"]


@pytest.fixture
def bad_crc_mcap(test_fixtures) -> Path:
    """Corrupt MCAP file (bad CRC)."""
    return test_fixtures["bad_crc"]


@pytest.fixture
def uncompressed_mcap(test_fixtures) -> Path:
    """Uncompressed MCAP file."""
    return test_fixtures["uncompressed"]


@pytest.fixture
def lz4_mcap(test_fixtures) -> Path:
    """LZ4 compressed MCAP file."""
    return test_fixtures["lz4_compressed"]


@pytest.fixture
def large_1mb_mcap(test_fixtures) -> Path:
    """Large MCAP file (~1MB) for benchmarking."""
    return test_fixtures["large_1mb"]


@pytest.fixture
def large_10mb_mcap(test_fixtures) -> Path:
    """Large MCAP file (~10MB) for benchmarking."""
    return test_fixtures["large_10mb"]


@pytest.fixture
def output_file(tmp_path) -> Path:
    """Temporary output file path."""
    return tmp_path / "output.mcap"


@pytest.fixture(scope="session")
def nuscenes_mcap() -> Path:
    """Real-world nuScenes MCAP file for integration testing."""
    return (
        Path(__file__).parent.parent.parent
        / "data"
        / "data"
        / "nuScenes-v1.0-mini-scene-0061-ros2.mcap"
    )


@pytest.fixture
def image_rgb_mcap(image_fixtures) -> Path:
    """MCAP file with RGB Image messages."""
    return image_fixtures["image_rgb"]


@pytest.fixture
def image_compressed_mcap(image_fixtures) -> Path:
    """MCAP file with CompressedImage messages."""
    return image_fixtures["image_compressed"]


@pytest.fixture
def image_small_mcap(image_fixtures) -> Path:
    """Small MCAP file with few image frames for quick tests."""
    return image_fixtures["image_small"]
