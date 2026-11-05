"""Announce the ready upstream server through Supervisor discovery."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from urllib.request import Request, urlopen

from predictor.api.priceapi import app


def announce():
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        with urlopen(Request("http://supervisor/addons/self/info", headers=headers), timeout=10) as response:
            host = json.load(response)["data"]["hostname"]
        body = json.dumps({"service": "epexpredictor", "config": {"url": f"http://{host}:8000"}}).encode()
        with urlopen(Request("http://supervisor/discovery", data=body, headers=headers), timeout=10):
            pass
    except (OSError, ValueError, KeyError):
        logging.getLogger(__name__).warning("Supervisor discovery failed; manual integration setup remains available")


upstream_lifespan = app.router.lifespan_context


@asynccontextmanager
async def lifespan(application):
    async with upstream_lifespan(application):
        await asyncio.to_thread(announce)
        yield


app.router.lifespan_context = lifespan
