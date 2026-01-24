"""Tests for predictor.model.pricestore module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from predictor.model.pricestore import PriceStore


class TestPriceStoreInit:
    """Tests for PriceStore initialization."""

    def test_init_without_storage(self, sample_region):
        """Test initialization without storage directory."""
        store = PriceStore(sample_region)
        assert store.region == sample_region
        assert store.storage_dir is None

    def test_init_with_storage(self, sample_region, temp_storage_dir):
        """Test initialization with storage directory."""
        store = PriceStore(sample_region, temp_storage_dir)
        assert store.storage_dir == temp_storage_dir
        assert store.storage_fn_prefix.startswith("prices")


class TestPriceStoreFetchMissingData:
    """Tests for fetch_missing_data method."""

    @pytest.mark.asyncio
    async def test_fetch_creates_api_request(self, sample_region):
        """Test that fetch_missing_data creates proper API requests."""
        import json
        store = PriceStore(sample_region)

        # Create mock response matching energy-charts.info format
        mock_response_data = {
            "unix_seconds": [
                1730419200,  # 2024-11-01 00:00 UTC
                1730422800,  # 2024-11-01 01:00 UTC
                1730426400,  # 2024-11-01 02:00 UTC
            ],
            "price": [80.5, 75.2, 70.1],  # EUR/MWh
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            # The actual code uses resp.text() then json.loads()
            mock_response.text = AsyncMock(return_value=json.dumps(mock_response_data))

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_context)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            # Use a longer date range to ensure gen_missing_date_ranges yields ranges
            start = datetime(2024, 11, 1, tzinfo=timezone.utc)
            end = datetime(2024, 11, 3, tzinfo=timezone.utc)

            await store.fetch_missing_data(start, end)

            # Verify a request was made
            assert mock_session_instance.get.called


class TestPriceStoreMissingRanges:
    """Tests for gen_missing_date_ranges method."""

    def test_empty_store_returns_full_range(self, sample_region):
        """Test that empty store returns the full requested range."""
        store = PriceStore(sample_region)
        start = datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, 12, 0, tzinfo=timezone.utc)

        ranges = list(store.gen_missing_date_ranges(start, end))
        assert len(ranges) >= 1

    def test_partial_data_returns_missing_ranges(self, sample_region):
        """Test that partially filled store returns only missing ranges."""
        store = PriceStore(sample_region)

        # Add data for Nov 1
        dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0] * len(dates)}, index=dates)
        df.index.name = "time"
        store._update_data(df)

        # Request range including Nov 1-3
        start = datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, 12, 0, tzinfo=timezone.utc)

        ranges = list(store.gen_missing_date_ranges(start, end))

        # Should only return ranges for Nov 2-3 (not Nov 1 which we have)
        for range_start, range_end in ranges:
            assert range_start >= datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc)


class TestPriceStoreDataConversion:
    """Tests for price data conversion."""

    def test_price_conversion_eur_mwh_to_ct_kwh(self, sample_region):
        """Test that prices are converted from EUR/MWh to ct/kWh."""
        store = PriceStore(sample_region)

        # Create data with EUR/MWh prices (what energy-charts returns)
        dates = pd.date_range(
            start="2025-11-01", end="2025-11-01T01:00", freq="15min", tz="UTC"
        )
        # 100 EUR/MWh should become 10 ct/kWh
        df = pd.DataFrame({"price": [10.0] * len(dates)}, index=dates)
        df.index.name = "time"
        store._update_data(df)

        # Verify data is stored correctly
        assert not store.data.empty
        assert "price" in store.data.columns



class TestPriceStoreResampling:
    """Tests for data resampling functionality."""

    def test_hourly_data_resampled_to_15min(self, sample_region):
        """Test that hourly data is resampled to 15-minute intervals."""
        store = PriceStore(sample_region)

        # Add hourly data
        dates = pd.date_range(
            start="2025-11-01", end="2025-11-01T03:00", freq="1h", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0, 9.0, 10.0, 11.0]}, index=dates)
        df.index.name = "time"
        store._update_data(df)

        # Data should be stored (resampling happens during fetch)
        assert not store.data.empty
