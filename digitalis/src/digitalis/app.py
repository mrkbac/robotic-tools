import argparse
import os


def _configure_for_ssh() -> None:
    if os.environ.get("SSH_CONNECTION"):
        os.environ.setdefault("TEXTUAL_FPS", "5")
        os.environ.setdefault("TEXTUAL_ANIMATIONS", "none")


def main() -> None:
    _configure_for_ssh()

    parser = argparse.ArgumentParser(description="Digitalis - MCAP Topic Browser")
    parser.add_argument(
        "file_or_url",
        help="Path to MCAP file or WebSocket URL to browse",
        default="ws://localhost:8765",
    )
    args = parser.parse_args()

    # Imported here so env vars set above land before textual reads them at import time.
    from digitalis._runtime import DigitalisApp, configure_logging  # noqa: PLC0415

    configure_logging()
    DigitalisApp(args.file_or_url).run()


if __name__ == "__main__":
    main()
