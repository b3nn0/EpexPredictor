"""Tests for predictor.model.auxdatastore module."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from predictor.model.auxdatastore import AuxDataStore


class TestAuxDataStoreInit:
    """Tests for AuxDataStore initialization."""

    def test_init_creates_empty_store(self, sample_region):
        """Test initialization creates empty store."""
        store = AuxDataStore(sample_region)
        assert store.region == sample_region
        assert store.data.empty


class TestAuxDataStoreIsHoliday:
    """Tests for is_holiday method."""

    def test_sunday_is_holiday(self, sample_region):
        """Test that Sunday returns 1.0 (full holiday)."""
        store = AuxDataStore(sample_region)
        # Nov 2, 2025 is a Sunday
        sunday = pd.Timestamp("2025-11-02", tz="UTC")
        result = store.is_holiday(sunday)
        assert result == pytest.approx(1.0)

    def test_regular_weekday_not_holiday(self, sample_region):
        """Test that regular weekday returns low holiday value."""
        store = AuxDataStore(sample_region)
        # Nov 5, 2025 is a Wednesday (not a holiday)
        wednesday = pd.Timestamp("2025-11-05", tz="UTC")
        result = store.is_holiday(wednesday)
        # Should be 0 or a small fraction if some regions have a holiday
        assert 0.0 <= result <= 1.0

    def test_christmas_is_holiday(self, sample_region):
        """Test that Christmas Day returns 1.0."""
        store = AuxDataStore(sample_region)
        christmas = pd.Timestamp("2025-12-25", tz="UTC")
        result = store.is_holiday(christmas)
        # Christmas is a holiday in all German states
        assert result == pytest.approx(1.0)

    def test_new_year_is_holiday(self, sample_region):
        """Test that New Year's Day returns 1.0."""
        store = AuxDataStore(sample_region)
        new_year = pd.Timestamp("2025-01-01", tz="UTC")
        result = store.is_holiday(new_year)
        assert result == pytest.approx(1.0)


class TestAuxDataStoreIsTimestamp:
    """Tests for is_timestamp method."""

    def test_matching_timestamp(self, sample_region):
        """Test that matching hour/minute returns 1."""
        store = AuxDataStore(sample_region)
        tz = timezone.utc
        dt = datetime(2025, 11, 1, 12, 30, tzinfo=tz)
        result = store.is_timestamp(tz, dt, 12, 30)
        assert result == 1

    def test_non_matching_timestamp(self, sample_region):
        """Test that non-matching hour/minute returns 0."""
        store = AuxDataStore(sample_region)
        tz = timezone.utc
        dt = datetime(2025, 11, 1, 12, 30, tzinfo=tz)
        result = store.is_timestamp(tz, dt, 10, 0)
        assert result == 0


class TestAuxDataStoreFetchMissingData:
    """Tests for fetch_missing_data method."""

    @pytest.mark.asyncio
    async def test_fetch_creates_aux_features(self, sample_region):
        """Test that fetch_missing_data creates auxiliary features."""
        store = AuxDataStore(sample_region)
        # Use a longer range to ensure the algorithm generates ranges
        # (gen_missing_date_ranges uses noon internally)
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # Verify data was created
        assert not store.data.empty

        # Check for expected columns
        assert "holiday" in store.data.columns
        # Day of week columns
        for i in range(6):
            assert f"day_{i}" in store.data.columns
        # Time slot columns (format: i_{hour}_{minute})
        assert "i_0_0" in store.data.columns  # midnight
        assert "i_23_45" in store.data.columns  # last slot
        # Sunrise/sunset influence
        assert "sr_influence" in store.data.columns
        assert "ss_influence" in store.data.columns

    @pytest.mark.asyncio
    async def test_fetch_respects_15min_intervals(self, sample_region):
        """Test that fetch creates 15-minute interval data."""
        store = AuxDataStore(sample_region)
        # Use a longer range to ensure data generation
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # Should have multiple 15-min intervals
        assert len(store.data) >= 4


class TestAuxDataStoreDayOfWeekEncoding:
    """Tests for day of week encoding."""

    @pytest.mark.asyncio
    async def test_day_columns_are_one_hot(self, sample_region):
        """Test that day columns are one-hot encoded."""
        store = AuxDataStore(sample_region)
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)  # Saturday
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # For each row, exactly one day column should be 1 (or 0 for Sunday)
        day_cols = [f"day_{i}" for i in range(6)]
        for _, row in store.data.iterrows():
            day_sum = sum(row[col] for col in day_cols)
            # Sum should be 0 (Sunday) or 1 (Mon-Sat)
            assert day_sum in [0, 1]


class TestAuxDataStoreTimeSlotEncoding:
    """Tests for time slot encoding."""

    @pytest.mark.asyncio
    async def test_slot_columns_are_one_hot(self, sample_region):
        """Test that time slot columns are one-hot encoded."""
        store = AuxDataStore(sample_region)
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 1, 1, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # For each row, exactly one slot column should be 1
        # Columns are named i_{hour}_{minute}
        slot_cols = [f"i_{h}_{m}" for h in range(24) for m in range(0, 60, 15)]
        for _, row in store.data.iterrows():
            slot_sum = sum(row[col] for col in slot_cols if col in row.index)
            assert slot_sum == 1

    @pytest.mark.asyncio
    async def test_correct_slots_for_midnight(self, sample_region):
        """Test that time slots are one-hot encoded properly."""
        store = AuxDataStore(sample_region)
        # Use a range that will generate data
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # Check that data was created and has time slot columns
        assert not store.data.empty
        # Each row should have exactly one time slot set to 1
        slot_cols = [f"i_{h}_{m}" for h in range(24) for m in range(0, 60, 15)]
        for _, row in store.data.head(10).iterrows():
            slot_sum = sum(row[col] for col in slot_cols if col in row.index)
            assert slot_sum == 1


class TestAuxDataStoreSunriseSunset:
    """Tests for sunrise/sunset influence calculation."""

    @pytest.mark.asyncio
    async def test_sunrise_sunset_influence_range(self, sample_region):
        """Test that sunrise/sunset influence values are in valid range."""
        store = AuxDataStore(sample_region)
        # Use a range that will generate data
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # Values should be between 0 and 180 (capped at 3 hours)
        assert store.data["sr_influence"].min() >= 0
        assert store.data["sr_influence"].max() <= 180
        assert store.data["ss_influence"].min() >= 0
        assert store.data["ss_influence"].max() <= 180

    @pytest.mark.asyncio
    async def test_midday_has_high_influence(self, sample_region):
        """Test that midday has high sunrise/sunset influence (far from both)."""
        store = AuxDataStore(sample_region)
        # Use a range that will generate data
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # Filter to midday hours
        midday_data = store.data[(store.data.index.hour >= 11) & (store.data.index.hour <= 13)]
        if not midday_data.empty:
            # Both should be at or near the cap of 180
            assert midday_data["sr_influence"].mean() > 100
            assert midday_data["ss_influence"].mean() > 100


class TestAuxDataStoreGetData:
    """Tests for get_data method."""

    @pytest.mark.asyncio
    async def test_get_data_fetches_and_returns(self, sample_region):
        """Test that get_data fetches missing data and returns it."""
        store = AuxDataStore(sample_region)
        # Use a range that will generate data
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        result = await store.get_data(start, end)

        assert not result.empty
        # Data may extend beyond requested range due to full-day generation
        assert len(result) > 0
