import abc
import logging
import os
from datetime import datetime, timezone

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
    horizon_cutoff: datetime|None = None

    def __init__(self, region : PriceRegion, storage_dir: str|None = None, storage_fn_prefix: str|None = None):
        self.data = pd.DataFrame()
        self.region = region
        self.storage_dir = storage_dir
        self.storage_fn_prefix = storage_fn_prefix

        self.load()


    def get_known_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        if self.horizon_cutoff and self.horizon_cutoff < end:
            end = self.horizon_cutoff
        return self.data.loc[start:end]
    
    async def get_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        await self.fetch_missing_data(start, end)

        if self.horizon_cutoff and self.horizon_cutoff < end:
            end = self.horizon_cutoff
        return self.data.loc[start:end]
    
    @abc.abstractmethod
    async def fetch_missing_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        pass

    def get_last_known(self) -> datetime|None:
        data = self.data
        if self.horizon_cutoff:
            data = data[:self.horizon_cutoff]
        if len(data) == 0:
            return None
        return data.dropna().reset_index().iloc[-1]["time"]


    def drop_after(self, dt: datetime):
        if self.data.empty:
            return
        self.data = self.data[self.data.index <= pd.to_datetime(dt, utc=True)]

    def drop_before(self, dt: datetime):
        if self.data.empty:
            return
        self.data = self.data[self.data.index >= pd.to_datetime(dt, utc=True)]

    def _update_data(self, df: pd.DataFrame):
        self.data = df.combine_first(self.data).dropna() # keeps new data from df, fills it with existing data from self


    def get_storage_file(self):
        if self.storage_dir is None or self.storage_fn_prefix is None:
            return None
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        return f"{self.storage_dir}/{self.storage_fn_prefix}_{self.region.bidding_zone}.json.gz"

    def serialize(self):
        fn = self.get_storage_file()
        if fn is not None:
            log.info(f"storing new {self.storage_fn_prefix} data for {self.region.bidding_zone}")
            self.data.to_json(fn, compression='gzip')
    
    def load(self):
        fn = self.get_storage_file()
        if fn is not None and os.path.exists(fn):
            log.info(f"loading persisted {self.storage_fn_prefix} data for {self.region.bidding_zone}")
            self.data = pd.read_json(fn, compression='gzip')

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



