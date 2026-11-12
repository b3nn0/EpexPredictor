"""Run the real container with an offline upstream stand-in, including S6 init."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def wait_for(path, expected):
    for _ in range(60):
        if path.exists() and path.read_text() == expected:
            return
        time.sleep(1)
    raise AssertionError(f"Timed out waiting for {path.name}")


def check(image):
    stub = Path(__file__).resolve().parent / "fixtures/priceapi.py"
    name = "epexpredictor-lifecycle-smoke"
    with TemporaryDirectory() as temp:
        data = Path(temp)
        (data / "options.json").write_text(json.dumps({"entsoe_api_key": "test-key", "use_nordpool": True}))
        try:
            docker("run", "-d", "--name", name, "--add-host", "supervisor:127.0.0.1",
                   "-e", "SUPERVISOR_TOKEN=smoke-token", "-v", f"{data}:/data",
                   "-v", f"{stub}:/code/predictor/api/priceapi.py:ro", image)
            for count in (1, 2):
                if count == 2:
                    docker("start", name)
                wait_for(data / "starts", str(count))
                expected = {"service": "epexpredictor", "config": {"url": "http://test-epexpredictor:8000"}}
                wait_for(data / "discovery.json", json.dumps(expected))
                environment = json.loads((data / "environment.json").read_text())
                assert environment == {"EPEXPREDICTOR_DATADIR": "/data", "EPEXPREDICTOR_USE_NORDPOOL": "true", "EPEXPREDICTOR_ENTSOE_API_KEY": "test-key"}
                docker("exec", name, "python", "-c",
                       "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/openapi.json').close()")
                docker("stop", "--time", "20", name)
                assert (data / "stopped").read_text() == str(count), "Graceful shutdown did not finish"
                assert docker("inspect", "-f", "{{.State.ExitCode}}", name) != "137", "Container required SIGKILL"
                (data / "discovery.json").unlink()
            print("Container smoke passed: options, discovery, HTTP readiness, persistent data, graceful stop/restart")
        except BaseException:
            subprocess.run(["docker", "logs", name], check=False)
            raise
        finally:
            subprocess.run(["docker", "rm", "-f", name], check=False)


if __name__ == "__main__":
    check(sys.argv[1])
