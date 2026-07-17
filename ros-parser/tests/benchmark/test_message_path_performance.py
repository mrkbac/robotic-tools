"""Focused MessagePath microbenchmarks.

Run with:
  uv run pytest ros-parser/tests/benchmark/test_message_path_performance.py \
    --benchmark-only --no-cov -q
"""

from __future__ import annotations

from ros_parser.message_path import MessagePathEvaluator, parse_message_path


def test_benchmark_message_path_parse_complex(benchmark) -> None:
    source = '/lidar.fields[:]{name == "z"}.@length{==1}.@@delta.@abs{!=0}.@@count{<=10}'

    result = benchmark(parse_message_path, source)

    assert result.has_stream


def test_benchmark_message_path_apply_nested_field(benchmark) -> None:
    path = parse_message_path("/odom.pose.pose.position.x")
    message = {"pose": {"pose": {"position": {"x": 12.5, "y": 3.0, "z": -1.0}}}}

    result = benchmark(path.apply, message)

    assert result == 12.5


def test_benchmark_message_path_apply_pointcloud_contract(benchmark) -> None:
    path = parse_message_path("/lidar.@product(width, height){>=100}")
    message = {"width": 34_528, "height": 1}

    result = benchmark(path.apply, message)

    assert result == 34_528


def test_benchmark_message_path_apply_zero_argument_modifier(benchmark) -> None:
    path = parse_message_path("/imu.acceleration.x.@abs")
    message = {"acceleration": {"x": -12.5}}

    result = benchmark(path.apply, message)

    assert result == 12.5


def test_benchmark_message_path_apply_filtered_array(benchmark) -> None:
    path = parse_message_path('/lidar.fields[:]{name == "z"}.@length{==1}')
    message = {
        "fields": [
            {"name": "x", "offset": 0},
            {"name": "y", "offset": 4},
            {"name": "z", "offset": 8},
            {"name": "intensity", "offset": 12},
        ]
    }

    result = benchmark(path.apply, message)

    assert result == 1


def test_benchmark_message_path_evaluator_plain_path(benchmark) -> None:
    path = parse_message_path("/odom.pose.pose.position.x")
    evaluator = MessagePathEvaluator(path)
    message = {"pose": {"pose": {"position": {"x": 12.5}}}}

    result = benchmark(evaluator.observe, message, 0)

    assert result == 12.5


def test_benchmark_message_path_stream_delta_max_1000_messages(benchmark) -> None:
    path = parse_message_path("/position.value.@@delta.@@max")
    messages = [{"value": index} for index in range(1_000)]

    def evaluate_stream() -> float:
        evaluator = MessagePathEvaluator(path)
        for timestamp_ns, message in enumerate(messages):
            evaluator.observe(message, timestamp_ns)
        result = evaluator.finalize()
        assert isinstance(result, float)
        return result

    result = benchmark(evaluate_stream)

    assert result == 1.0


def test_benchmark_message_path_stream_pipeline_1000_messages(benchmark) -> None:
    path = parse_message_path("/position.value.@@delta.@abs{!=0}.@@count")
    messages = [{"value": index // 2} for index in range(1_000)]

    def evaluate_stream() -> int:
        evaluator = MessagePathEvaluator(path)
        for timestamp_ns, message in enumerate(messages):
            evaluator.observe(message, timestamp_ns)
        result = evaluator.finalize()
        assert isinstance(result, int)
        return result

    result = benchmark(evaluate_stream)

    assert result == 499
