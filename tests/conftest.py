"""Shared pytest fixtures for EpexPredictor tests."""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from predictor.model.priceregion import PriceRegionName


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for data storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_region():
    """Return a sample price region for testing."""
    return PriceRegionName.DE.to_region()


@pytest.fixture
def sample_datetime():
    """Return a sample datetime for testing."""
    return datetime(2025, 11, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_date_range(sample_datetime):
    """Return a sample date range for testing."""
    start = sample_datetime
    end = start + timedelta(days=7)
    return start, end


@pytest.fixture
def sample_weather_data():
    """Create sample weather data DataFrame."""
    dates = pd.date_range(
        start="2025-11-01",
        end="2025-11-02",
        freq="15min",
        tz="UTC"
    )
    data = {
        "wind_speed_80m_0": [5.0] * len(dates),
        "wind_speed_80m_1": [6.0] * len(dates),
        "temperature_2m_0": [10.0] * len(dates),
        "temperature_2m_1": [11.0] * len(dates),
        "global_tilted_irradiance_0": [100.0] * len(dates),
        "global_tilted_irradiance_1": [110.0] * len(dates),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def sample_price_data():
    """Create sample price data DataFrame."""
    dates = pd.date_range(
        start="2025-10-23", # longer range for lagged features
        end="2025-11-02",
        freq="15min",
        tz="UTC"
    )
    # Simulate realistic price patterns (higher during day, lower at night)
    prices = []
    for dt in dates:
        hour = dt.hour
        if 6 <= hour <= 20:
            prices.append(8.0 + (hour - 12) * 0.5)  # Day prices
        else:
            prices.append(4.0)  # Night prices

    df = pd.DataFrame({"price": prices}, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def sample_aux_data():
    """Create sample auxiliary data DataFrame."""
    dates = pd.date_range(
        start="2025-11-01",
        end="2025-11-02",
        freq="15min",
        tz="UTC"
    )
    data = {
        "holiday": [0.0] * len(dates),
        "day_0": [1 if d.weekday() == 0 else 0 for d in dates],
        "day_1": [1 if d.weekday() == 1 else 0 for d in dates],
        "day_2": [1 if d.weekday() == 2 else 0 for d in dates],
        "day_3": [1 if d.weekday() == 3 else 0 for d in dates],
        "day_4": [1 if d.weekday() == 4 else 0 for d in dates],
        "day_5": [1 if d.weekday() == 5 else 0 for d in dates],
        "sr_influence": [60] * len(dates),
        "ss_influence": [60] * len(dates),
    }
    # Add time slot columns (format: i_{hour}_{minute})
    for h in range(24):
        for m in range(0, 60, 15):
            data[f"i_{h}_{m}"] = [1 if (d.hour == h and d.minute == m) else 0 for d in dates]

    df = pd.DataFrame(data, index=dates)
    df.index.name = "time"
    return df

@pytest.fixture
def sample_entsoe_data():
    dates = pd.date_range(
        start="2025-11-01",
        end="2025-11-02",
        freq="15min",
        tz="UTC"
    )
    data = {
        "maxload": [500] * len(dates),
        "minload" : [500] * len(dates)
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "time"
    return df



@pytest.fixture
def mock_aiohttp_response():
    """Create a mock aiohttp response."""
    def _create_response(json_data, status=200):
        mock_response = AsyncMock()
        mock_response.status = status
        mock_response.json = AsyncMock(return_value=json_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        return mock_response
    return _create_response


@pytest.fixture
def extended_price_data():
    """Create extended price data for lagged features testing."""
    dates = pd.date_range(
        start="2025-10-20", end="2025-11-02", freq="15min", tz="UTC"
    )
    df = pd.DataFrame({"price": [8.0] * len(dates)}, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def mocked_predictor(
    sample_region, sample_weather_data, sample_price_data, sample_aux_data, sample_entsoe_data, extended_price_data
):
    """Create a PricePredictor with all stores mocked."""
    from predictor.model.pricepredictor import PricePredictor

    predictor = PricePredictor(sample_region)
    predictor.weatherstore.get_data = AsyncMock(return_value=sample_weather_data)
    predictor.pricestore.get_data = AsyncMock(return_value=sample_price_data)
    predictor.auxstore.get_data = AsyncMock(return_value=sample_aux_data)
    predictor.entsoestore.get_data = AsyncMock(return_value=sample_entsoe_data)
    return predictor
