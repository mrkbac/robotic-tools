import argparse
import asyncio
import contextlib
import logging
import signal
import sys

from rich.console import Console
from rich.logging import RichHandler

from websocket_proxy.dashboard import DashboardRenderer
from websocket_proxy.proxy import ProxyBridge
from websocket_proxy.transformers import TransformerRegistry
from websocket_proxy.transformers.image_to_video import ImageToVideoTransformer
from websocket_proxy.transformers.pointcloud_voxel import PointCloudVoxelTransformer

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
        "--max-message-size",
        type=int,
        default=0,
        help="Maximum websocket message size in bytes (<=0 disables limit, default: unlimited)",
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

    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the live dashboard display",
    )
    parser.add_argument(
        "--dashboard-refresh-rate",
        type=float,
        default=1.0,
        help="Dashboard refresh rate in seconds (default: 1.0)",
    )

    parser.set_defaults(image_use_hardware=True, pointcloud_skip_nans=True)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    # Create shared console for dashboard and logging
    console = Console()

    # Create transformer registry and register transformers (BEFORE configuring logging)
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

    # Create proxy bridge with transformers
    bridge = ProxyBridge(
        upstream_url=args.source_ws,
        listen_host=args.host,
        listen_port=args.port,
        transformer_registry=registry,
        default_throttle_hz=args.throttle_hz,
        max_message_size=args.max_message_size if args.max_message_size > 0 else None,
    )

    # Create dashboard if enabled (with shared console for logging integration)
    dashboard = None
    if not args.no_dashboard:
        dashboard = DashboardRenderer(
            bridge, refresh_rate=args.dashboard_refresh_rate, console=console
        )
        # Start dashboard BEFORE configuring logging
        dashboard.start_sync()

    # NOW configure logging with Rich handler (after dashboard is started)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=args.verbose,
            )
        ],
    )

    logger.info("Registered transformers:")
    for transformer in registry.get_all_transformers():
        logger.info(f"  {transformer.get_input_schema()} -> {transformer.get_output_schema()}")

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler() -> None:
        logger.info("Received shutdown signal")
        asyncio.create_task(bridge.stop())  # noqa: RUF006

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        if dashboard:
            # Dashboard already started above (before logging config)
            # Just create a background task for dashboard updates
            dashboard_task = asyncio.create_task(dashboard.run_updates())

            try:
                # Start proxy (this will block until stop() is called)
                await bridge.start()
            finally:
                # Cancel dashboard updates
                dashboard_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await dashboard_task
                await dashboard.stop()
        else:
            # No dashboard - just start proxy
            await bridge.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception:
        logger.exception("Unexpected error in proxy bridge")
        sys.exit(1)
    finally:
        await bridge.stop()
        if dashboard:
            await dashboard.stop()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Exiting")


if __name__ == "__main__":
    main()
