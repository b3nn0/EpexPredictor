"""Translate Supervisor options, then replace this process with upstream Uvicorn."""

import json
import os
from pathlib import Path


def main():
    try:
        options = json.loads(Path("/data/options.json").read_text())
        key = options.get("entsoe_api_key", "")
        nordpool = options.get("use_nordpool", False)
        if not isinstance(key, str) or not isinstance(nordpool, bool):
            raise ValueError
    except (OSError, ValueError, AttributeError):
        raise SystemExit("Invalid App options: check entsoe_api_key and use_nordpool.") from None

    os.environ["EPEXPREDICTOR_DATADIR"] = "/data"
    os.environ["EPEXPREDICTOR_USE_NORDPOOL"] = str(nordpool).lower()
    if key.strip():
        os.environ["EPEXPREDICTOR_ENTSOE_API_KEY"] = key.strip()
    else:
        os.environ.pop("EPEXPREDICTOR_ENTSOE_API_KEY", None)
    os.execvp("uvicorn", ["uvicorn", "predictor.api.priceapi:app", "--host", "0.0.0.0", "--port", "8000"])


if __name__ == "__main__":
    main()
