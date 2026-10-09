import contextlib
import http.server
import threading
from collections.abc import Generator
from pathlib import Path
from typing import override

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "data" / "runs"
RUN_DIRS = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())
GOLDEN_DIR = REPO / "tests" / "golden"


class QuietHandler(http.server.BaseHTTPRequestHandler):
    """A request handler that logs nothing."""

    @override
    def log_message(self, format, *args):
        pass


@contextlib.contextmanager
def local_server(handler: type[http.server.BaseHTTPRequestHandler]) -> Generator[str]:
    """`handler` served on a free local port from a background thread; yields the base URL."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
