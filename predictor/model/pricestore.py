import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Generator

import aiohttp
import pandas as pd

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

class PriceStore(DataStore):
    """
    Fetches and caches price info from api.energy-charts.info
    TODO: add more price sources, e.g. for SE1-SE4, which is not available from energy-charts
    """

    data : pd.DataFrame
    region : PriceRegion
    storage_dir : str|None

    update_lock: asyncio.Lock
    

    def __init__(self, region : PriceRegion, storage_dir=None):
        super().__init__(region, storage_dir, "prices_v3")
        self.update_lock = asyncio.Lock()



    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        async with self.update_lock:
            start = start.astimezone(timezone.utc)
            end = end.astimezone(timezone.utc)

            updated = False

            for rstart, rend in self.gen_missing_date_ranges(start, end):
                start_of_day = rstart.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = rend.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                start_formatted = start_of_day.isoformat().replace("+00:00", "Z")
                end_formatted = end_of_day.isoformat().replace("+00:00", "Z")
                url = f"https://api.energy-charts.info/price?bzn={self.region.bidding_zone}&start={start_formatted}&end={end_formatted}"
                log.info(f"Fetching price data for {self.region.bidding_zone}: {url}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers={"accept": "application/json"}) as resp:
                        txt = await resp.text()
                        if "no content available" in txt:
                            continue
                        
                        data = json.loads(txt)
                        timestamps = data["unix_seconds"]
                        prices = data["price"]
                        df = pd.DataFrame.from_dict(dict(zip(timestamps, prices)), orient="index", columns=["price"])
                        df.index = pd.to_datetime(df.index, unit="s", utc=True)
                        df.index.name = "time"
                        df["price"] = df["price"] / 10
                        
                        # for BE, a few hours are missing in late september.. make sure they are filled or stuff will get out of hand
                        # TODO: this will become an issue if we ever have a region where a whole day or so is missing, and the df is empty.
                        # will solve that when needed.
                        df = df.sort_index().resample('15min').ffill().bfill()

                        self._update_data(df)
                        updated = True

        
            if updated:
                log.info(f"price data updated for {self.region.bidding_zone}")
                self.data.sort_index(inplace=True)
                self.serialize()
            return updated

    def gen_missing_date_ranges(self, start: datetime, end: datetime) -> Generator[tuple[datetime, datetime]]:
        start = start.replace(hour=12, minute=0, second=0, microsecond=0)

        curr = start

        rangestart = None
        while curr <= end:
            next_day = curr + timedelta(days=1)

            if rangestart is not None and (next_day in self.data.index or next_day > end or (curr - rangestart).total_seconds() > 60 * 60 * 24 * 90):
                # We have the next timeslot already OR its the last timeslot OR the current range exceeds 90 days
                yield (rangestart, curr)
                rangestart = None

            if rangestart is None and curr not in self.data.index:
                rangestart = curr

            curr = next_day
