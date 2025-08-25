import itertools
from pathlib import Path

import pytest
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from mcap_ros2_support_fast._dynamic import create_decoder
from mcap_ros2_support_fast.decoder import DecoderFactory as DecoderFactoryFast


def _read_all(factory, msgs: int):
    file = (
        Path(__file__).parent.parent.parent
        / "data"
        / "data"
        / "nuScenes-v1.0-mini-scene-0061-ros2.mcap"
    )

    with file.open("rb") as f:
        reader = make_reader(f, decoder_factories=[factory])
        for _ in itertools.islice(reader.iter_decoded_messages(), msgs):
            pass


@pytest.mark.parametrize(
    ("factory", "msgs"),
    [
        pytest.param(factory, msgs, id=f"{name}-{msgs}")
        for factory, name in [
            (DecoderFactory(), "mcap_ros2"),
            (DecoderFactoryFast(create_decoder), "mcap_ros2_fast"),
        ]
        for msgs in [10, 100, 1_000]
    ],
)
@pytest.mark.benchmark(group="msgs-")
def test_benchmark_decoder(benchmark, factory, msgs):
    benchmark.group += str(msgs)
    benchmark(_read_all, factory, msgs)
