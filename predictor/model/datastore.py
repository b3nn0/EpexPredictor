import abc
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Self

import pandas as pd

from .priceregion import PriceRegion

log = logging.getLogger(__name__)


class DataStore:
    """
    Base class for caching data store with delta-fetching and serialization
    """

    data: pd.DataFrame
    region: PriceRegion
    storage_dir: str|None
    storage_fn_prefix: str|None
    
    # Used during model performance evaluation to not accidently access prices we shouldn't know about yet
    horizon_cutoff: datetime|None

    # actually known horizon limit of the data source. Do not query missing data after the actual source horizon constantly
    known_source_horizon: datetime|None
    # when is next revalidation due?
    source_horizon_revalitation_ts: datetime|None

    last_updated: datetime

    def __init__(self, region : PriceRegion, storage_dir: str|None = None, storage_fn_prefix: str|None = None):
        self.data = pd.DataFrame()
        self.region = region
        self.storage_dir = storage_dir
        self.storage_fn_prefix = storage_fn_prefix

        self.last_updated = datetime(1970, 1, 1, tzinfo=timezone.utc)

        self.horizon_cutoff = None
        self.known_source_horizon = None
        self.source_horizon_revalitation_ts = None


    def set_source_horizon(self, horizon: datetime, revalidation_ts: datetime|None):
        self.known_source_horizon = horizon
        self.source_horizon_revalitation_ts = revalidation_ts

    def apply_horizon(self, start: datetime, end: datetime) -> tuple[datetime, datetime]:
        if self.horizon_cutoff:
            start = min(start, self.horizon_cutoff)
            end = min(end, self.horizon_cutoff)

        if self.known_source_horizon and (self.source_horizon_revalitation_ts is None or datetime.now(timezone.utc) < self.source_horizon_revalitation_ts):
            start = min(start, self.known_source_horizon)
            end = min(end, self.known_source_horizon)
        return (start, end)
    

    
    async def get_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        start, end = self.apply_horizon(start, end)

        await self.fetch_missing_data(start, end)

        last_known = self.get_last_known()
        if last_known and last_known < end:
            # source horizon reached - remember/reschedule source query
            self.known_source_horizon = last_known
            self.source_horizon_revalitation_ts = self.get_next_horizon_revalidation_time()

        if self.horizon_cutoff and self.horizon_cutoff < end:
            end = self.horizon_cutoff
        return self.data.loc[start:end]
    
    def needs_horizon_revalidation(self):
        return self.source_horizon_revalitation_ts is not None and datetime.now(timezone.utc) > self.source_horizon_revalitation_ts

    @abc.abstractmethod
    async def fetch_missing_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_next_horizon_revalidation_time(self) -> datetime|None:
        pass


    def gen_missing_date_ranges(self, start: datetime, end: datetime) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        # Full 15-minute grid
        needed = pd.date_range(start=pd.to_datetime(start).floor("15min"), end=pd.to_datetime(end).ceil("15min"), freq="15min")

        # Reindex to find missing timestamps
        missing = self.data.reindex(needed).isna().all(axis=1)

        # Keep only missing slots
        missing = missing[missing]

        if missing.empty:
            return []

        # Group consecutive 15-minute gaps
        groups = (
            missing.index
            .to_series()
            .diff()
            .ne(pd.Timedelta("15min"))
            .cumsum()
        )

        ranges = (
            missing.index
            .to_series()
            .groupby(groups)
            .agg(["min", "max"])
        )

        return list(ranges.itertuples(index=False, name=None))


    def get_last_known(self) -> datetime|None:
        data = self.data
        if self.horizon_cutoff:
            data = data[:self.horizon_cutoff]
        if len(data) == 0:
            return None
        return data.index[-1]


    def drop_after(self, dt: datetime):
        if self.data.empty:
            return
        self.data = self.data[self.data.index <= pd.to_datetime(dt, utc=True)]

    def drop_before(self, dt: datetime):
        if self.data.empty:
            return
        self.data = self.data[self.data.index >= pd.to_datetime(dt, utc=True)]

    def _update_data(self, df: pd.DataFrame) -> bool:
        olddata = self.data
        self.data = df.combine_first(self.data).dropna() # keeps new data from df, fills it with existing data from self

        changed = not olddata.round(decimals=10).equals(self.data.round(decimals=10))
        if changed:
            self.last_updated = datetime.now(timezone.utc)
        return changed


    def get_storage_file(self):
        if self.storage_dir is None or self.storage_fn_prefix is None:
            return None
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        return f"{self.storage_dir}/{self.storage_fn_prefix}_{self.region.bidding_zone_entsoe}.json.gz"

    async def serialize(self):
        fn = self.get_storage_file()
        if fn is not None:
            log.info(f"storing new {self.storage_fn_prefix} data for {self.region.bidding_zone_entsoe}")
            await asyncio.to_thread(self.data.to_json, fn, compression='gzip')
    
    async def load(self) -> Self:
        fn = self.get_storage_file()
        if fn is not None and os.path.exists(fn):
            log.info(f"loading persisted {self.storage_fn_prefix} data for {self.region.bidding_zone_entsoe}")
            self.data = await asyncio.to_thread(pd.read_json, fn, compression='gzip')

            # Handle index type: to_json saves DatetimeIndex as epoch milliseconds,
            # which read_json loads as Int64Index. Convert back to DatetimeIndex.
            if pd.api.types.is_integer_dtype(self.data.index.dtype):
                # Index values are epoch milliseconds
                self.data.index = pd.to_datetime(self.data.index, unit='ms', utc=True)
            elif isinstance(self.data.index, pd.DatetimeIndex) and self.data.index.tz is None:
                # Index is DatetimeIndex but naive, localize to UTC
                self.data.index = self.data.index.tz_localize("UTC")
            elif not isinstance(self.data.index, pd.DatetimeIndex):
                # Unexpected index type - log warning and attempt conversion
                log.warning(f"Unexpected index type {type(self.data.index).__name__} in persisted data, "
                           f"attempting datetime conversion")
                self.data.index = pd.to_datetime(self.data.index, utc=True)

            self.data.index.set_names("time", inplace=True)
            self.data.dropna(inplace=True)

            self.last_updated = datetime.fromtimestamp(os.path.getmtime(fn), tz=timezone.utc)
        return self



