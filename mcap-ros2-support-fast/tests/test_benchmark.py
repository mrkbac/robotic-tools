import itertools
from pathlib import Path

import pytest
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from mcap_ros2_support_fast._dynamic import create_decoder
from mcap_ros2_support_fast.decoder import DecoderFactory as DecoderFactoryFast


def _read_all(factory):
    file = (
        Path(__file__).parent.parent.parent
        / "data"
        / "data"
        / "nuScenes-v1.0-mini-scene-0061-ros2.mcap"
    )

    with file.open("rb") as f:
        reader = make_reader(f, decoder_factories=[factory])
        for _ in itertools.islice(reader.iter_decoded_messages(), 100):
            pass


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(DecoderFactory(), id="mcap_ros2"),
        pytest.param(DecoderFactoryFast(create_decoder), id="mcap_ros2_fast"),
    ],
)
def test_benchmark_decoder(benchmark, factory):
    benchmark(_read_all, factory)
