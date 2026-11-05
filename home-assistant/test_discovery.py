"""Test the App's Supervisor boundary without loading forecasting dependencies."""

from contextlib import asynccontextmanager
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError


def load_wrapper():
    @asynccontextmanager
    async def lifespan(app):
        yield

    upstream = SimpleNamespace(app=SimpleNamespace(router=SimpleNamespace(lifespan_context=lifespan)))
    spec = importlib.util.spec_from_file_location("ha_app", Path(__file__).parent / "epexpredictor/ha_app.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"predictor.api.priceapi": upstream}):
        spec.loader.exec_module(module)
    return module


class DiscoveryTests(unittest.TestCase):
    def test_uses_supervisor_hostname_and_announces_only_endpoint(self):
        wrapper = load_wrapper()
        response = MagicMock()
        response.__enter__.return_value = BytesIO(b'{"data":{"hostname":"example-epexpredictor"}}')
        with patch.dict(os.environ, {"SUPERVISOR_TOKEN": "test-token"}), patch.object(wrapper, "urlopen", side_effect=[response, MagicMock()]) as request:
            wrapper.announce()
            first, second = [call.args[0] for call in request.call_args_list]
            self.assertEqual(first.full_url, "http://supervisor/addons/self/info")
            self.assertEqual(json.loads(second.data), {"service": "epexpredictor", "config": {"url": "http://example-epexpredictor:8000"}})

    def test_missing_supervisor_token_skips_discovery(self):
        wrapper = load_wrapper()
        with patch.dict(os.environ, {}, clear=True), patch.object(wrapper, "urlopen") as request:
            wrapper.announce()
            request.assert_not_called()

    def test_discovery_failure_does_not_stop_server_or_log_token(self):
        wrapper = load_wrapper()
        with patch.dict(os.environ, {"SUPERVISOR_TOKEN": "test-token"}), patch.object(wrapper, "urlopen", side_effect=URLError("unavailable")), self.assertLogs("ha_app", "WARNING") as logs:
            wrapper.announce()
        self.assertNotIn("test-token", " ".join(logs.output))


if __name__ == "__main__":
    unittest.main()
