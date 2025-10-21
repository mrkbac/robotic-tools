"""CLI entry point for fox-bridge proxy."""

import argparse
import asyncio
import logging
import signal
import sys

from fox_bridge.proxy import ProxyBridge
from fox_bridge.transformers import TransformerRegistry
from fox_bridge.transformers.image_to_video import ImageToVideoTransformer
from fox_bridge.transformers.pointcloud_voxel import PointCloudVoxelTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Foxglove WebSocket proxy - forwards topics with optional transformations"
    )
    parser.add_argument(
        "source_ws",
        help="WebSocket URL of the upstream Foxglove bridge (e.g., ws://localhost:8765)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Port to listen on for downstream clients (default: 8766)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104
        help="Host to listen on for downstream clients (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--throttle-hz",
        type=float,
        default=1.0,
        help="Topic throttle rate in Hz (default: 1.0; set to 0 to disable)",
    )

    parser.add_argument(
        "--image-codec",
        default="h264",
        help="Video codec to use for image compression (default: h264)",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=23,
        help="CRF/quality value for image compression (lower is higher quality, default: 23)",
    )
    parser.add_argument(
        "--image-preset",
        default="fast",
        help="Encoder preset for image compression (default: fast)",
    )
    parser.add_argument(
        "--image-max-dimension",
        type=int,
        default=480,
        help="Maximum width/height used when downscaling images before encoding (default: 480)",
    )
    parser.add_argument(
        "--image-disable-hw",
        dest="image_use_hardware",
        action="store_false",
        help="Disable hardware acceleration for image compression",
    )

    parser.add_argument(
        "--pointcloud-voxel-size",
        type=float,
        default=0.1,
        help="Voxel size (in meters) for point cloud downsampling (default: 0.1)",
    )
    parser.add_argument(
        "--pointcloud-keep-nans",
        dest="pointcloud_skip_nans",
        action="store_false",
        help="Keep NaN points when voxelizing point clouds (default: drop NaNs)",
    )

    parser.set_defaults(image_use_hardware=True, pointcloud_skip_nans=True)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create transformer registry and register transformers
    registry = TransformerRegistry()

    # Register image to video transformer
    image_transformer = ImageToVideoTransformer(
        codec=args.image_codec,
        quality=args.image_quality,
        preset=args.image_preset,
        use_hardware=args.image_use_hardware,
        max_dimension=args.image_max_dimension,
    )
    registry.register(image_transformer)

    pointcloud_transformer = PointCloudVoxelTransformer(
        voxel_size=args.pointcloud_voxel_size,
        skip_nans=args.pointcloud_skip_nans,
    )
    registry.register(pointcloud_transformer)

    logger.info("Registered transformers:")
    for transformer in registry.get_all_transformers():
        logger.info(f"  {transformer.get_input_schema()} -> {transformer.get_output_schema()}")

    # Create proxy bridge with transformers
    bridge = ProxyBridge(
        upstream_url=args.source_ws,
        listen_host=args.host,
        listen_port=args.port,
        transformer_registry=registry,
        default_throttle_hz=args.throttle_hz,
    )

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler() -> None:
        logger.info("Received shutdown signal")
        asyncio.create_task(bridge.stop())  # noqa: RUF006

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await bridge.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception:
        logger.exception("Unexpected error in proxy bridge")
        sys.exit(1)
    finally:
        await bridge.stop()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Exiting")


if __name__ == "__main__":
    main()
