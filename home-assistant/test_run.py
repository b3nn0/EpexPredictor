"""Verify the configuration boundary without starting training or fetching data."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("app_run", Path(__file__).parent / "epexpredictor/run.py")
app_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_run)


class LauncherTests(unittest.TestCase):
    def test_options_and_process_handoff(self):
        with patch.object(Path, "read_text", return_value='{"entsoe_api_key":" key ","use_nordpool":true}'), patch.dict(os.environ, {}, clear=True), patch.object(os, "execvp") as execute:
            app_run.main()
            self.assertEqual(os.environ["EPEXPREDICTOR_ENTSOE_API_KEY"], "key")
            self.assertEqual(os.environ["EPEXPREDICTOR_USE_NORDPOOL"], "true")
            self.assertEqual(os.environ["EPEXPREDICTOR_DATADIR"], "/data")
            self.assertEqual(execute.call_args.args[1], ["uvicorn", "predictor.api.priceapi:app", "--host", "0.0.0.0", "--port", "8000"])

    def test_empty_key_does_not_enable_key_only_regions(self):
        with patch.object(Path, "read_text", return_value='{}'), patch.dict(os.environ, {"EPEXPREDICTOR_ENTSOE_API_KEY": "old"}, clear=True), patch.object(os, "execvp"):
            app_run.main()
            self.assertNotIn("EPEXPREDICTOR_ENTSOE_API_KEY", os.environ)
            self.assertEqual(os.environ["EPEXPREDICTOR_USE_NORDPOOL"], "false")

    def test_invalid_options_never_start_server_or_leak_key(self):
        for content in ['{"entsoe_api_key":"secret","use_nordpool":"false"}', '[]', '{']:
            with self.subTest(content=content), patch.object(Path, "read_text", return_value=content), patch.object(os, "execvp") as execute:
                with self.assertRaises(SystemExit) as error:
                    app_run.main()
                self.assertNotIn("secret", str(error.exception))
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
