"""Small HTTP client for the existing compact forecast API."""

import asyncio
from math import isfinite


def validate_forecast(data):
    if not isinstance(data, dict):
        raise ValueError("Expected a forecast object")
    stamps, prices = data.get("s"), data.get("t")
    if not isinstance(stamps, list) or not isinstance(prices, list) or not stamps or len(stamps) != len(prices):
        raise ValueError("Expected equally sized, nonempty timestamp and price arrays")
    if any(type(v) not in (int, float) or not isfinite(v) for v in stamps + prices):
        raise ValueError("Forecast contains nonnumeric or nonfinite values")
    if any(v <= 0 for v in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Forecast timestamps must be positive and strictly increasing")
    return {"s": stamps, "t": prices}


async def fetch(session, config, evaluation=False):
    params = {
        "region": config["region"], "surcharge": config["surcharge"],
        "taxPercent": config["tax_percent"], "unit": "EUR_PER_KWH", "hours": 120,
        "evaluation": str(evaluation).lower(),
    }
    async with asyncio.timeout(60):
        async with session.get(config["url"] + "/prices_short", params=params) as response:
            response.raise_for_status()
            return validate_forecast(await response.json())
