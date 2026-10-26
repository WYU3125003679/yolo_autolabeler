"""Command-line entry point for the local annotation server."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from .server import make_server


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autolabeler",
        description="Start the local YOLO annotation web application.",
    )
    parser.add_argument("workspace", type=Path, help="workspace directory")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser automatically",
    )
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"workspace is not an existing directory: {workspace}")

    server = make_server((args.host, args.port), workspace)
    address = server.server_address
    url = f"http://{address[0]}:{address[1]}"
    print(f"Autolabeler workspace: {workspace}")
    print(f"Open {url} in your browser. Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping autolabeler.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
