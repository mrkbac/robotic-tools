# mcap-codec-support

High-performance decoder and encoder factories for robotics data in MCAP files.
It supports regular ROS2 CDR messages, H.264/H.265/VP9/AV1 video, and
Cloudini/Draco point clouds.

Install all decoding backends:

```console
uv add "mcap-codec-support[all]"
```

Read every supported message type with `small-mcap`:

```python
from mcap_codec_support import create_decoder_factories
from small_mcap import read_message_decoded

with open("input.mcap", "rb") as stream:
    messages = read_message_decoded(
        stream,
        decoder_factories=create_decoder_factories(video_format="raw"),
    )
    for message in messages:
        print(message.channel.topic, message.decoded_message)
```

`video_format="raw"` produces ROS2 `Image`-shaped dictionaries.
`video_format="compressed"` produces JPEG `CompressedImage`-shaped
dictionaries. Compressed point clouds are returned as `PointCloud2`-shaped
dictionaries. All other ROS2 messages use the fast CDR decoder.

For more control, compose the individual `VideoDecompressFactory`,
`PointCloudDecompressFactory`, and ROS2 `DecoderFactory` instances yourself.
Keep specialized factories before the general ROS2 factory.
