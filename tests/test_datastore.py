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
        expected = f"{temp_storage_dir}/test_{sample_region.bidding_zone_entsoe}.json.gz"
        assert store.get_storage_file() == expected

    @pytest.mark.asyncio
    async def test_serialize_and_load(self, sample_region, temp_storage_dir):
        """Test serialization and loading of data."""
        # Create store and add data
        store1 = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        dates = pd.date_range(
            start="2025-01-01", end="2025-01-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"value": range(len(dates))}, index=dates)
        df.index.name = "time"
        store1._update_data(df)
        await store1.serialize()

        # Verify file exists
        assert os.path.exists(store1.get_storage_file())

        # Create new store and load data
        store2 = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()

        # Data should be loaded
        assert len(store2.data) == len(store1.data)
        assert store2.data.index.equals(store1.data.index)

    @pytest.mark.asyncio
    async def test_serialize_without_storage_dir(self, sample_region):
        """Test serialize does nothing without storage dir."""
        store = ConcreteDataStore(sample_region)
        dates = pd.date_range(
            start="2025-01-01", end="2025-01-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"value": range(len(dates))}, index=dates)
        df.index.name = "time"
        store._update_data(df)
        await store.serialize()  # Should not raise


class TestDataStorePersistenceEdgeCases:
    """Tests for edge cases in persistence loading."""

    def test_load_nonexistent_file(self, sample_region, temp_storage_dir):
        """Test loading when no persisted file exists results in empty data."""
        store = ConcreteDataStore(sample_region, temp_storage_dir, "nonexistent")
        assert store.data.empty

    @pytest.mark.asyncio
    async def test_load_corrupted_json_file(self, sample_region, temp_storage_dir):
        """Test loading from corrupted JSON file raises a ValueError."""
        import gzip

        # Create a corrupted gzip file
        storage_path = f"{temp_storage_dir}/test_{sample_region.bidding_zone_entsoe}.json.gz"
        with gzip.open(storage_path, 'wt') as f:
            f.write("{ this is not valid json }")

        # Loading should raise a ValueError (pandas raises this for invalid JSON)
        with pytest.raises(ValueError):
            await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()

    def test_load_empty_json_file(self, sample_region, temp_storage_dir):
        """Test loading from empty JSON object."""
        import gzip

        # Create an empty JSON object file
        storage_path = f"{temp_storage_dir}/test_{sample_region.bidding_zone_entsoe}.json.gz"
        with gzip.open(storage_path, 'wt') as f:
            f.write("{}")

        store = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        assert store.data.empty

    @pytest.mark.asyncio
    async def test_load_with_int64_index_epoch_ms(self, sample_region, temp_storage_dir):
        """Test loading data where index is saved as epoch milliseconds (Int64Index).

        This is the default behavior when using to_json() without special handling.
        The load() method should convert it back to DatetimeIndex.
        """
        import gzip
        import json

        # Create JSON with epoch milliseconds as keys (simulating to_json output)
        dates = pd.date_range(start="2025-01-01", periods=10, freq="15min", tz="UTC")
        data = {}
        for col in ["value"]:
            data[col] = {str(int(d.timestamp() * 1000)): i for i, d in enumerate(dates)}

        storage_path = f"{temp_storage_dir}/test_{sample_region.bidding_zone_entsoe}.json.gz"
        with gzip.open(storage_path, 'wt') as f:
            json.dump(data, f)

        store = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()

        # Verify data was loaded correctly
        assert not store.data.empty
        assert isinstance(store.data.index, pd.DatetimeIndex)
        assert store.data.index.tz is not None  # Should be UTC
        assert len(store.data) == 10

    @pytest.mark.asyncio
    async def test_load_with_naive_datetime_index(self, sample_region, temp_storage_dir):
        """Test loading legacy data where index is stored as naive ISO datetime strings.

        This simulates an older persisted file whose index is encoded as ISO datetimes
        without timezone information. The load() method should localize the
        resulting DatetimeIndex to UTC.
        """
        import gzip
        import json

        # Create JSON with ISO datetime strings as keys (produces naive DatetimeIndex)
        naive_dates = pd.date_range(start="2025-01-01", periods=10, freq="15min")
        data = {
            "value": {d.isoformat(): i for i, d in enumerate(naive_dates)}
        }

        storage_path = f"{temp_storage_dir}/test_{sample_region.bidding_zone_entsoe}.json.gz"
        with gzip.open(storage_path, 'wt') as f:
            json.dump(data, f)

        store = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()

        # Verify data was loaded and localized to UTC
        assert not store.data.empty
        assert isinstance(store.data.index, pd.DatetimeIndex)
        assert store.data.index.tz is not None  # Should be localized to UTC
        assert str(store.data.index.tz) == "UTC"
        assert len(store.data) == 10

        # Verify the timestamps match the original naive times interpreted as UTC
        expected_utc = naive_dates.tz_localize("UTC")
        expected_utc.name = "time"  # load() sets the index name to "time"
        pd.testing.assert_index_equal(store.data.index, expected_utc)

    @pytest.mark.asyncio
    async def test_load_preserves_data_values(self, sample_region, temp_storage_dir):
        """Test that loading preserves the original data values."""
        # Create and serialize data
        store1 = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        dates = pd.date_range(start="2025-01-01", periods=10, freq="15min", tz="UTC")
        df = pd.DataFrame({"value": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]}, index=dates)
        df.index.name = "time"
        store1._update_data(df)
        await store1.serialize()

        # Load into new store
        store2 = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()

        # Values should match
        assert store2.data["value"].tolist() == pytest.approx([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

    @pytest.mark.asyncio
    async def test_load_with_nan_values(self, sample_region, temp_storage_dir):
        """Test that loading data with NaN values drops them correctly."""
        import gzip
        import json
        import math

        # Create JSON with some NaN values
        dates = pd.date_range(start="2025-01-01", periods=5, freq="15min", tz="UTC")
        data = {
            "value": {
                str(int(dates[0].timestamp() * 1000)): 1.0,
                str(int(dates[1].timestamp() * 1000)): None,  # NaN in JSON
                str(int(dates[2].timestamp() * 1000)): 3.0,
                str(int(dates[3].timestamp() * 1000)): None,  # NaN in JSON
                str(int(dates[4].timestamp() * 1000)): 5.0,
            }
        }

        storage_path = f"{temp_storage_dir}/test_{sample_region.bidding_zone_entsoe}.json.gz"
        with gzip.open(storage_path, 'wt') as f:
            json.dump(data, f)

        store = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()

        # NaN rows should be dropped
        assert len(store.data) == 3
        assert all(not math.isnan(v) for v in store.data["value"])

    @pytest.mark.asyncio
    async def test_load_index_has_correct_name(self, sample_region, temp_storage_dir):
        """Test that loaded data has index named 'time'."""
        store1 = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        dates = pd.date_range(start="2025-01-01", periods=5, freq="15min", tz="UTC")
        df = pd.DataFrame({"value": range(5)}, index=dates)
        df.index.name = "time"
        store1._update_data(df)
        await store1.serialize()

        store2 = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()
        assert store2.data.index.name == "time"

    @pytest.mark.asyncio
    async def test_load_index_is_utc(self, sample_region, temp_storage_dir):
        """Test that loaded data has UTC timezone on index."""
        store1 = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        dates = pd.date_range(start="2025-01-01", periods=5, freq="15min", tz="UTC")
        df = pd.DataFrame({"value": range(5)}, index=dates)
        df.index.name = "time"
        store1._update_data(df)
        await store1.serialize()

        store2 = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()
        assert store2.data.index.tz is not None
        assert str(store2.data.index.tz) == "UTC"

    @pytest.mark.asyncio
    async def test_load_multiple_columns(self, sample_region, temp_storage_dir):
        """Test loading data with multiple columns."""
        store1 = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        dates = pd.date_range(start="2025-01-01", periods=5, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "value1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "value2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "value3": [100.0, 200.0, 300.0, 400.0, 500.0],
        }, index=dates)
        df.index.name = "time"
        store1._update_data(df)
        await store1.serialize()

        store2 = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()

        assert list(store2.data.columns) == ["value1", "value2", "value3"]
        assert store2.data["value1"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
        assert store2.data["value2"].tolist() == pytest.approx([10.0, 20.0, 30.0, 40.0, 50.0])

    @pytest.mark.asyncio
    async def test_load_large_dataset(self, sample_region, temp_storage_dir):
        """Test loading a larger dataset (1 year of 15-min data)."""
        store1 = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        # 1 year of 15-min intervals = 35040 rows
        dates = pd.date_range(start="2025-01-01", periods=35040, freq="15min", tz="UTC")
        df = pd.DataFrame({"value": range(35040)}, index=dates)
        df.index.name = "time"
        store1._update_data(df)
        await store1.serialize()

        store2 = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()

        assert len(store2.data) == 35040
        assert store2.data.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
        assert store2.data.index[-1] == pd.Timestamp("2025-12-31 23:45:00", tz="UTC")

    @pytest.mark.asyncio
    async def test_serialize_overwrites_existing_file(self, sample_region, temp_storage_dir):
        """Test that serialize overwrites existing data file."""
        # Create first version
        store1 = ConcreteDataStore(sample_region, temp_storage_dir, "test")
        dates1 = pd.date_range(start="2025-01-01", periods=5, freq="15min", tz="UTC")
        df1 = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates1)
        df1.index.name = "time"
        store1._update_data(df1)
        await store1.serialize()

        # Create second version with different data
        store2 = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()
        dates2 = pd.date_range(start="2025-06-01", periods=3, freq="15min", tz="UTC")
        df2 = pd.DataFrame({"value": [100, 200, 300]}, index=dates2)
        df2.index.name = "time"
        store2.data = pd.DataFrame()  # Clear loaded data
        store2._update_data(df2)
        await store2.serialize()

        # Load and verify only second version exists
        store3 = await ConcreteDataStore(sample_region, temp_storage_dir, "test").load()
        assert len(store3.data) == 3
        assert store3.data["value"].tolist() == [100, 200, 300]
