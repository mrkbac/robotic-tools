import argparse
import logging

from websocket_bridge.client import WebSocketBridgeClient

logging.basicConfig(level=logging.DEBUG)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="WebSocket server URL")
    args = parser.parse_args()

    client = WebSocketBridgeClient(args.url)
    await client.connect()

    await asyncio.sleep(30)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
