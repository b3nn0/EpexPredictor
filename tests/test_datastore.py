"""Tests for predictor.model.datastore module."""

import os
from datetime import datetime, timezone

import pandas as pd
import pytest

from predictor.model.datastore import DataStore


class ConcreteDataStore(DataStore):
    """Concrete implementation of DataStore for testing."""

    async def fetch_missing_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Mock implementation that creates sample data."""
        dates = pd.date_range(start=start, end=end, freq="15min", tz="UTC")
        df = pd.DataFrame({"value": range(len(dates))}, index=dates)
        df.index.name = "time"
        self._update_data(df)
        return df


class TestDataStoreInit:
    """Tests for DataStore initialization."""

    def test_init_without_storage(self, sample_region):
        """Test initialization without storage directory."""
        store = ConcreteDataStore(sample_region)
        assert store.region == sample_region
        assert store.storage_dir is None
        assert store.storage_fn_prefix is None
        assert store.data.empty

    def test_init_with_storage(self, sample_region, temp_storage_dir):
        """Test initialization with storage directory."""
        store = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        assert store.storage_dir == temp_storage_dir
        assert store.storage_fn_prefix == "test"


class TestDataStoreGetKnownData:
    """Tests for get_known_data method."""

    def test_get_known_data_empty(self, sample_region):
        """Test getting data from empty store."""
        store = ConcreteDataStore(sample_region)
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        result = store.get_known_data(start, end)
        assert result.empty

    def test_get_known_data_with_data(self, sample_region):
        """Test getting data from populated store."""
        store = ConcreteDataStore(sample_region)

        # Add some data
        dates = pd.date_range(
            start="2025-01-01", end="2025-01-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"value": range(len(dates))}, index=dates)
        df.index.name = "time"
        store._update_data(df)

        # Get subset of data
        start = datetime(2025, 1, 1, 6, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        result = store.get_known_data(start, end)

        assert not result.empty
        assert result.index[0] >= pd.Timestamp(start)
        assert result.index[-1] <= pd.Timestamp(end)


class TestDataStoreGetData:
    """Tests for async get_data method."""

    @pytest.mark.asyncio
    async def test_get_data_fetches_missing(self, sample_region):
        """Test that get_data fetches missing data."""
        store = ConcreteDataStore(sample_region)
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 6, tzinfo=timezone.utc)

        result = await store.get_data(start, end)
        assert not result.empty


class TestDataStoreDropMethods:
    """Tests for drop_after and drop_before methods."""

    def test_drop_after(self, sample_region):
        """Test dropping data after a specific datetime."""
        store = ConcreteDataStore(sample_region)

        # Add data
        dates = pd.date_range(
            start="2025-01-01", end="2025-01-03", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"value": range(len(dates))}, index=dates)
        df.index.name = "time"
        store._update_data(df)

        # Drop after Jan 2
        cutoff = datetime(2025, 1, 2, tzinfo=timezone.utc)
        store.drop_after(cutoff)

        assert store.data.index.max() <= pd.Timestamp(cutoff)

    def test_drop_before(self, sample_region):
        """Test dropping data before a specific datetime."""
        store = ConcreteDataStore(sample_region)

        # Add data
        dates = pd.date_range(
            start="2025-01-01", end="2025-01-03", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"value": range(len(dates))}, index=dates)
        df.index.name = "time"
        store._update_data(df)

        # Drop before Jan 2
        cutoff = datetime(2025, 1, 2, tzinfo=timezone.utc)
        store.drop_before(cutoff)

        assert store.data.index.min() >= pd.Timestamp(cutoff)

    def test_drop_after_empty_store(self, sample_region):
        """Test drop_after on empty store doesn't crash."""
        store = ConcreteDataStore(sample_region)
        cutoff = datetime(2025, 1, 2, tzinfo=timezone.utc)
        store.drop_after(cutoff)  # Should not raise

    def test_drop_before_empty_store(self, sample_region):
        """Test drop_before on empty store doesn't crash."""
        store = ConcreteDataStore(sample_region)
        cutoff = datetime(2025, 1, 2, tzinfo=timezone.utc)
        store.drop_before(cutoff)  # Should not raise


class TestDataStoreUpdateData:
    """Tests for _update_data method."""

    def test_update_data_new(self, sample_region):
        """Test adding new data to empty store."""
        store = ConcreteDataStore(sample_region)

        dates = pd.date_range(
            start="2025-01-01", end="2025-01-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"value": range(len(dates))}, index=dates)
        df.index.name = "time"
        store._update_data(df)

        assert len(store.data) == len(df)

    def test_update_data_merge(self, sample_region):
        """Test merging new data with existing data."""
        store = ConcreteDataStore(sample_region)

        # Add initial data
        dates1 = pd.date_range(
            start="2025-01-01", end="2025-01-02", freq="15min", tz="UTC"
        )
        df1 = pd.DataFrame({"value": [1] * len(dates1)}, index=dates1)
        df1.index.name = "time"
        store._update_data(df1)

        # Add overlapping data with different values
        dates2 = pd.date_range(
            start="2025-01-01T12:00", end="2025-01-03", freq="15min", tz="UTC"
        )
        df2 = pd.DataFrame({"value": [2] * len(dates2)}, index=dates2)
        df2.index.name = "time"
        store._update_data(df2)

        # Check that we have data from full range
        assert store.data.index.min() == pd.Timestamp("2025-01-01", tz="UTC")
        assert store.data.index.max() == pd.Timestamp("2025-01-03", tz="UTC")

    def test_update_data_removes_duplicates(self, sample_region):
        """Test that duplicate timestamps are handled correctly."""
        store = ConcreteDataStore(sample_region)

        dates = pd.date_range(
            start="2025-01-01", end="2025-01-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"value": range(len(dates))}, index=dates)
        df.index.name = "time"

        # Add same data twice
        store._update_data(df)
        store._update_data(df)

        # Should not have duplicates
        assert len(store.data) == len(df)


class TestDataStoreSerialization:
    """Tests for serialize and load methods."""

    def test_get_storage_file_without_dir(self, sample_region):
        """Test get_storage_file returns None without storage dir."""
        store = ConcreteDataStore(sample_region)
        assert store.get_storage_file() is None

    def test_get_storage_file_with_dir(self, sample_region, temp_storage_dir):
        """Test get_storage_file returns proper path."""
        store = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        expected = f"{temp_storage_dir}/test_{sample_region.bidding_zone}.json.gz"
        assert store.get_storage_file() == expected

    def test_serialize_and_load(self, sample_region, temp_storage_dir):
        """Test serialization and loading of data."""
        # Create store and add data
        store1 = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        dates = pd.date_range(
            start="2025-01-01", end="2025-01-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"value": range(len(dates))}, index=dates)
        df.index.name = "time"
        store1._update_data(df)
        store1.serialize()

        # Verify file exists
        assert os.path.exists(store1.get_storage_file())

        # Create new store and load data
        store2 = ConcreteDataStore(sample_region, temp_storage_dir, "test")

        # Data should be loaded
        assert len(store2.data) == len(store1.data)
        assert store2.data.index.equals(store1.data.index)

    def test_serialize_without_storage_dir(self, sample_region):
        """Test serialize does nothing without storage dir."""
        store = ConcreteDataStore(sample_region)
        dates = pd.date_range(
            start="2025-01-01", end="2025-01-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"value": range(len(dates))}, index=dates)
        df.index.name = "time"
        store._update_data(df)
        store.serialize()  # Should not raise
