"""Offline upstream stand-in for exercising the real App container lifecycle."""

from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread

from fastapi import FastAPI

DATA = Path("/data")


class Supervisor(BaseHTTPRequestHandler):
    def do_GET(self):
        assert self.path == "/addons/self/info"
        self.respond({"data": {"hostname": "test-epexpredictor"}})

    def do_POST(self):
        assert self.path == "/discovery"
        assert self.headers["Authorization"] == "Bearer smoke-token"
        body = self.rfile.read(int(self.headers["Content-Length"]))
        (DATA / "discovery.json").write_bytes(body)
        self.respond({"result": "ok"})

    def respond(self, data):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *args):
        pass


@asynccontextmanager
async def lifespan(app):
    starts = DATA / "starts"
    starts.write_text(str(int(starts.read_text()) + 1 if starts.exists() else 1))
    (DATA / "environment.json").write_text(json.dumps({
        key: os.environ.get(key) for key in (
            "EPEXPREDICTOR_DATADIR", "EPEXPREDICTOR_USE_NORDPOOL", "EPEXPREDICTOR_ENTSOE_API_KEY"
        )
    }))
    server = ThreadingHTTPServer(("127.0.0.1", 80), Supervisor)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        (DATA / "stopped").write_text(starts.read_text())


app = FastAPI(lifespan=lifespan)
