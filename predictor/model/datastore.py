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

    data : pd.DataFrame
    region : PriceRegion
    storage_dir : str|None
    storage_fn_prefix : str|None
    

    def __init__(self, region : PriceRegion, storage_dir: str|None = None, storage_fn_prefix: str|None = None):
        self.data = pd.DataFrame()
        self.region = region
        self.storage_dir = storage_dir
        self.storage_fn_prefix = storage_fn_prefix

        self.load()

    def get_known_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        return self.data.loc[start:end]
    
    async def get_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        await self.fetch_missing_data(start, end)
        return self.data.loc[start:end]
    
    @abc.abstractmethod
    async def fetch_missing_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        pass


    def drop_after(self, dt: datetime):
        if self.data.empty:
            return
        self.data = self.data[self.data.index <= pd.to_datetime(dt, utc=True)]

    def drop_before(self, dt: datetime):
        if self.data.empty:
            return
        self.data = self.data[self.data.index >= pd.to_datetime(dt, utc=True)]

    def _update_data(self, df : pd.DataFrame):
        # TODO: isn't there a nicer way to do this?
        self.data = pd.concat([self.data, df.dropna()]).dropna()
        self.data = self.data.reset_index().drop_duplicates(subset='time', keep='last').set_index("time").sort_index()


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
            self.data.index = self.data.index.tz_localize("UTC") # type: ignore
            self.data.index.set_names("time", inplace=True)
            self.data.dropna(inplace=True)



