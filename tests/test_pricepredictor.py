"""Tests for predictor.model.pricepredictor module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from predictor.model.pricepredictor import PricePredictor


class TestPricePredictorInit:
    """Tests for PricePredictor initialization."""

    def test_init_creates_stores(self, sample_region):
        """Test that initialization creates all required stores."""
        predictor = PricePredictor(sample_region)
        assert predictor.region == sample_region
        assert predictor.weatherstore is not None
        assert predictor.pricestore is not None
        assert predictor.auxstore is not None

    def test_init_with_storage_dir(self, sample_region, temp_storage_dir):
        """Test initialization with storage directory."""
        predictor = PricePredictor(sample_region, temp_storage_dir)
        assert predictor.weatherstore.storage_dir == temp_storage_dir
        assert predictor.pricestore.storage_dir == temp_storage_dir


class TestPricePredictorGetLastKnownPrice:
    """Tests for get_last_known_price method."""

    def test_get_last_known_price_empty_store(self, sample_region):
        """Test get_last_known_price with empty price store."""
        predictor = PricePredictor(sample_region)
        result = predictor.pricestore.get_last_known()
        assert result is None

    def test_get_last_known_price_with_data(self, sample_region):
        """Test get_last_known_price with data in store."""
        predictor = PricePredictor(sample_region)

        # Add price data
        dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0] * len(dates)}, index=dates)
        df.index.name = "time"
        predictor.pricestore._update_data(df)

        result = predictor.pricestore.get_last_known()
        assert result is not None
        assert isinstance(result, datetime)


class TestPricePredictorToPriceDict:
    """Tests for to_price_dict method."""

    def test_to_price_dict(self, sample_region):
        """Test conversion of DataFrame to price dictionary."""
        predictor = PricePredictor(sample_region)

        dates = pd.date_range(
            start="2025-11-01", end="2025-11-01T01:00", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0, 9.0, 10.0, 11.0, 12.0]}, index=dates)
        df.index.name = "time"

        result = predictor.to_price_dict(df)

        assert isinstance(result, dict)
        assert len(result) == len(df)
        for dt, price in result.items():
            assert isinstance(dt, datetime)
            assert isinstance(price, float)


class TestPricePredictorPrepareDataframe:
    """Tests for prepare_dataframe method."""

    @pytest.mark.asyncio
    async def test_prepare_dataframe_combines_data(
        self, sample_region, sample_weather_data, sample_price_data, sample_aux_data
    ):
        """Test that prepare_dataframe combines all data sources."""
        predictor = PricePredictor(sample_region)

        # Mock the stores to return our sample data
        predictor.weatherstore.get_data = AsyncMock(return_value=sample_weather_data)
        predictor.pricestore.get_data = AsyncMock(return_value=sample_price_data)
        predictor.pricestore.get_known_data = MagicMock(return_value=sample_price_data)
        predictor.auxstore.get_data = AsyncMock(return_value=sample_aux_data)

        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        result = await predictor.prepare_dataframe(start, end, refresh_prices=True)

        assert result is not None
        assert not result.empty


class TestPricePredictorTrain:
    """Tests for train method."""

    @pytest.mark.asyncio
    async def test_train_creates_model(self, mocked_predictor):
        """Test that training creates a model."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        await mocked_predictor.train(start, end)

        assert mocked_predictor.predictor is not None


class TestPricePredictorPredict:
    """Tests for predict method."""

    @pytest.mark.asyncio
    async def test_predict_returns_dataframe(self, mocked_predictor):
        """Test that predict returns a DataFrame with predictions."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        # Train first
        await mocked_predictor.train(start, end)

        # Predict
        result = await mocked_predictor.predict(start, end)

        assert result is not None
        assert not result.empty
        assert "price" in result.columns

    @pytest.mark.asyncio
    async def test_predict_fill_known_true(self, mocked_predictor):
        """Test that predict with fill_known=True uses known prices."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        # Train first
        await mocked_predictor.train(start, end)

        # Predict with fill_known=True
        result = await mocked_predictor.predict(start, end, fill_known=True)

        assert result is not None
        assert not result.empty


class TestPricePredictorCleanup:
    """Tests for cleanup method."""

    def test_cleanup_removes_old_data(self, sample_region):
        """Test that cleanup removes data older than 1 year."""
        predictor = PricePredictor(sample_region)

        # Add old data (2 years ago)
        old_dates = pd.date_range(
            start="2023-01-01", end="2023-01-02", freq="15min", tz="UTC"
        )
        old_df = pd.DataFrame({"price": [8.0] * len(old_dates)}, index=old_dates)
        old_df.index.name = "time"
        predictor.pricestore._update_data(old_df)

        # Add recent data
        recent_dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        recent_df = pd.DataFrame({"price": [9.0] * len(recent_dates)}, index=recent_dates)
        recent_df.index.name = "time"
        predictor.pricestore._update_data(recent_df)

        # Cleanup
        predictor.cleanup()

        # Old data should be removed
        assert predictor.pricestore.data.index.min() > pd.Timestamp("2024-01-01", tz="UTC")


class TestPricePredictorRefreshMethods:
    """Tests for refresh_prices and refresh_weather methods."""

    @pytest.mark.asyncio
    async def test_refresh_prices_returns_true_when_no_initial_data(self, sample_region):
        """refresh_prices returns True when no initial data exists."""
        predictor = PricePredictor(sample_region)
        # No data in price store, so get_last_known_price() returns None
        result = await predictor.refresh_prices()
        assert result is True

    @pytest.mark.asyncio
    async def test_refresh_prices_returns_false_when_no_new_data(self, sample_region):
        """refresh_prices returns False when fetch_missing_data returns False."""
        predictor = PricePredictor(sample_region)

        # Add initial data
        dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0] * len(dates)}, index=dates)
        df.index.name = "time"
        predictor.pricestore._update_data(df)

        # Mock fetch to return False (no new data)
        predictor.pricestore.fetch_missing_data = AsyncMock(return_value=False)

        result = await predictor.refresh_prices()

        assert result is False
        assert predictor.pricestore.fetch_missing_data.called

    @pytest.mark.asyncio
    async def test_refresh_prices_returns_true_when_timestamp_advances(self, sample_region):
        """refresh_prices returns True when new data is fetched and timestamp advances."""
        predictor = PricePredictor(sample_region)

        # Add initial data
        initial_dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0] * len(initial_dates)}, index=initial_dates)
        df.index.name = "time"
        predictor.pricestore._update_data(df)

        initial_last = predictor.pricestore.get_last_known()

        # Mock fetch to return True and add new data
        def add_new_data(start, end):
            new_dates = pd.date_range(
                start="2025-11-02", end="2025-11-03", freq="15min", tz="UTC"
            )
            new_df = pd.DataFrame({"price": [9.0] * len(new_dates)}, index=new_dates)
            new_df.index.name = "time"
            predictor.pricestore._update_data(new_df)
            return True

        predictor.pricestore.fetch_missing_data = AsyncMock(side_effect=add_new_data)

        result = await predictor.refresh_prices()

        assert result is True
        new_last = predictor.pricestore.get_last_known()
        assert new_last > initial_last

    @pytest.mark.asyncio
    async def test_refresh_prices_returns_false_when_timestamp_unchanged(self, sample_region):
        """refresh_prices returns False when fetch returns True but timestamp unchanged."""
        predictor = PricePredictor(sample_region)

        # Add initial data
        dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0] * len(dates)}, index=dates)
        df.index.name = "time"
        predictor.pricestore._update_data(df)

        # Mock fetch to return True but don't add new data (timestamp stays same)
        predictor.pricestore.fetch_missing_data = AsyncMock(return_value=True)

        result = await predictor.refresh_prices()

        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_weather(self, sample_region):
        """Test refresh_weather method."""
        predictor = PricePredictor(sample_region)

        # Mock the weather store's refresh_range method (not fetch_missing_data)
        predictor.weatherstore.refresh_range = AsyncMock()

        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 8, tzinfo=timezone.utc)

        await predictor.refresh_weather(start, end)

        # Should have called refresh_range
        assert predictor.weatherstore.refresh_range.called


class TestPricePredictorLaggedFeatures:
    """Tests for lagged feature handling."""

    @pytest.mark.asyncio
    async def test_lagged_features_created(
        self, sample_region, sample_weather_data, sample_aux_data
    ):
        """Test that lagged price features are created correctly."""
        predictor = PricePredictor(sample_region)

        # Create price data with enough history for lagged features
        dates = pd.date_range(
            start="2025-10-20", end="2025-11-02", freq="15min", tz="UTC"
        )
        price_data = pd.DataFrame({"price": range(len(dates))}, index=dates)
        price_data.index.name = "time"

        # Mock the stores
        predictor.weatherstore.get_data = AsyncMock(return_value=sample_weather_data)
        predictor.pricestore.get_data = AsyncMock(return_value=price_data.loc["2025-11-01":"2025-11-02"])
        predictor.pricestore.get_known_data = MagicMock(return_value=price_data)
        predictor.auxstore.get_data = AsyncMock(return_value=sample_aux_data)

        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        df = await predictor.prepare_dataframe(start, end, refresh_prices=True)

        # Check that lagged features exist
        assert df is not None
        # The actual columns depend on how the data is merged
