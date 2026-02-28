"""Compare MCAP zstd compression vs pureini compression for PointCloud2 topics."""

from __future__ import annotations

import sys
from pathlib import Path

import zstandard as zstd
from mcap_ros2_support_fast.decoder import DecoderFactory
from pureini import (
    CompressionOption,
    EncodingInfo,
    EncodingOptions,
    FieldType,
    PointcloudEncoder,
    PointField,
)
from small_mcap import read_message

MCAP_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else None


def bytes_h(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def build_encoding_info(
    msg: object,
    encoding_opt: EncodingOptions,
    compression_opt: CompressionOption,
    resolution: float,
) -> EncodingInfo:
    info = EncodingInfo()
    info.width = msg.width  # type: ignore[attr-defined]
    info.height = msg.height  # type: ignore[attr-defined]
    info.point_step = msg.point_step  # type: ignore[attr-defined]
    info.encoding_opt = encoding_opt
    info.compression_opt = compression_opt
    info.fields = []
    for rf in msg.fields:  # type: ignore[attr-defined]
        info.fields.append(
            PointField(
                name=rf.name,
                offset=rf.offset,
                type=FieldType(rf.datatype),
                resolution=resolution if rf.datatype == FieldType.FLOAT32 else None,
            )
        )
    return info


def main() -> None:
    if MCAP_FILE is None or not MCAP_FILE.exists():
        print(f"Usage: {sys.argv[0]} <mcap_file>")
        return

    decoder_factory = DecoderFactory()
    zstd_cctx = zstd.ZstdCompressor(level=1)

    # Per-topic stats: {topic: [orig_raw, orig_zstd, cld_raw, cld_zstd, count]}
    stats: dict[str, list[int]] = {}

    with MCAP_FILE.open("rb") as f:
        for schema, channel, record in read_message(
            f,
            should_include=lambda _, s: (
                s is not None
                and s.name in {"sensor_msgs/msg/PointCloud2", "sensor_msgs/PointCloud2"}
            ),
        ):
            assert schema is not None
            decoder = decoder_factory.decoder_for("cdr", schema)
            if not decoder:
                continue

            msg = decoder(record.data)
            raw_data = bytes(msg.data)

            if channel.topic not in stats:
                stats[channel.topic] = [0, 0, 0, 0, 0]

            s = stats[channel.topic]
            s[4] += 1  # count

            # Original uncompressed
            s[0] += len(raw_data)

            # Original compressed (zstd level 1, same as MCAP default)
            s[1] += len(zstd_cctx.compress(raw_data))

            # Cloudini encoding info
            info_none = build_encoding_info(
                msg, EncodingOptions.LOSSY, CompressionOption.NONE, 0.01
            )
            info_zstd = build_encoding_info(
                msg, EncodingOptions.LOSSY, CompressionOption.ZSTD, 0.01
            )

            enc_none = PointcloudEncoder(info_none)
            enc_zstd = PointcloudEncoder(info_zstd)

            # Cloudini uncompressed
            cld_none = enc_none.encode(raw_data)
            s[2] += len(cld_none)

            # Cloudini compressed
            cld_zstd = enc_zstd.encode(raw_data)
            s[3] += len(cld_zstd)

            if s[4] % 500 == 0:
                print(f"  {channel.topic}: {s[4]} messages processed...")

    # Print results
    print()
    print(
        f"{'Topic':<45} {'Msgs':>7}  {'Orig Raw':>10}  {'Orig ZSTD':>10}  "
        f"{'Cld Raw':>10}  {'Cld ZSTD':>10}  {'Ratio':>6}"
    )
    print("-" * 140)

    totals = [0, 0, 0, 0, 0]
    for topic in sorted(stats):
        s = stats[topic]
        for i in range(5):
            totals[i] += s[i]
        ratio = s[0] / s[3] if s[3] > 0 else 0
        mcap_ratio = s[0] / s[1] if s[1] > 0 else 0
        print(
            f"{topic:<45} {s[4]:>7}  {bytes_h(s[0]):>10}  {bytes_h(s[1]):>10}  "
            f"{bytes_h(s[2]):>10}  {bytes_h(s[3]):>10}  {ratio:>5.1f}x"
        )

    print("-" * 140)
    ratio = totals[0] / totals[3] if totals[3] > 0 else 0
    mcap_ratio = totals[0] / totals[1] if totals[1] > 0 else 0
    print(
        f"{'TOTAL':<45} {totals[4]:>7}  {bytes_h(totals[0]):>10}  {bytes_h(totals[1]):>10}  "
        f"{bytes_h(totals[2]):>10}  {bytes_h(totals[3]):>10}  {ratio:>5.1f}x"
    )
    print()
    print(f"MCAP zstd ratio:     {mcap_ratio:.2f}x")
    print(f"Cloudini+zstd ratio: {ratio:.2f}x")
    print(f"Improvement over MCAP zstd: {totals[1] / totals[3]:.2f}x smaller")


if __name__ == "__main__":
    main()
