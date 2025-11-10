import argparse
import time
from pathlib import Path

import foxglove
from foxglove import Channel, Schema
from small_mcap.reader import get_summary, read_message


def main() -> None:
    parser = argparse.ArgumentParser(description="MCAP to Foxglove WebSocket Proxy")
    parser.add_argument(
        "mcap_file",
        type=str,
        help="Path to the MCAP file to stream",
    )
    args = parser.parse_args()

    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}

    mcap_file = Path(args.mcap_file)

    with mcap_file.open("rb") as f:
        summary = get_summary(f)
        assert summary is not None
        for sch in summary.schemas.values():
            schema = Schema(
                name=sch.name,
                encoding=sch.encoding,
                data=sch.data,
            )
            schemas[sch.id] = schema

        for chan in summary.channels.values():
            schema = schemas.get(chan.schema_id)
            assert schema is not None
            channel = Channel(
                topic=chan.topic,
                message_encoding=chan.message_encoding,
                schema=schema,
            )
            channels[chan.id] = channel
    foxglove.set_log_level("DEBUG")

    _server = foxglove.start_server()

    last_ns: int | None = None
    while True:
        with mcap_file.open("rb") as f:
            for _, c, msg in read_message(f):
                if channel := channels.get(c.id):
                    channel.log(msg.data, log_time=msg.log_time)
                    if last_ns is not None:
                        duration = msg.log_time - last_ns
                        time.sleep(duration / 1e9)
                    last_ns = msg.log_time


if __name__ == "__main__":
    main()
