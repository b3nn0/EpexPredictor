"""Small HTTP client for the existing compact forecast API."""

import asyncio
from datetime import datetime
from math import isfinite


def validate_forecast(data):
    if not isinstance(data, dict) or not isinstance(data.get("prices"), list) or not data["prices"]:
        raise ValueError("Expected a nonempty prices array")
    previous = None
    for item in data["prices"]:
        if not isinstance(item, dict):
            raise ValueError("Expected price objects with startsAt and total")
        try:
            starts_at = datetime.fromisoformat(item["startsAt"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Price objects must carry an ISO startsAt timestamp")
        total = item.get("total")
        if not isinstance(total, (int, float)) or isinstance(total, bool) or not isfinite(total):
            raise ValueError("Forecast contains nonnumeric or nonfinite values")
        if starts_at.tzinfo is None:
            raise ValueError("Price timestamps must be timezone-aware")
        if previous is not None and starts_at <= previous:
            raise ValueError("Forecast timestamps must be strictly increasing")
        previous = starts_at
    return data


async def fetch(session, config):
    params = {
        "region": config["region"], "surcharge": config["surcharge"],
        "taxPercent": config["tax_percent"], "unit": "EUR_PER_KWH", "hours": 120,
    }
    async with asyncio.timeout(60):
        async with session.get(config["url"] + "/prices", params=params) as response:
            response.raise_for_status()
            return validate_forecast(await response.json())
