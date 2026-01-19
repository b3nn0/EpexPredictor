import asyncio
import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Generator, cast

import pandas as pd
from astral import Observer, sun

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

class AuxDataStore(DataStore):
    """
    Used as in-memory store for computed data (holidays, slot number, day of week etc)
    Not stored to disk, just used as a cache to make retraining faster
    """

    data : pd.DataFrame
    region : PriceRegion

    update_lock: asyncio.Lock


    def __init__(self, region : PriceRegion, storage_dir: str | None = None):
        super().__init__(region)
        self.update_lock = asyncio.Lock()


    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        updated = False

        async with self.update_lock:
            for rstart, rend in self.gen_missing_date_ranges(start, end):
                df = await asyncio.to_thread(self._compute_data, rstart, rend)
                if len(df) > 0:
                    self._update_data(df)
                    updated = True

            if updated:
                log.info(f"aux data updated for {self.region.bidding_zone}")
                self.data.sort_index(inplace=True)
                self.serialize()
            return updated
        

    def _compute_data(self, rstart: datetime, rend: datetime) -> pd.DataFrame:
        """
        Careful: will be called in separate thread
        """
        tzlocal = self.region.get_timezone_info()

        # make it full day to be sure
        rstart = rstart.replace(hour=0, minute=0, second=0, microsecond=0)
        rend = rend.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        log.info(f"computing aux data from {rstart.isoformat()} to {rend.isoformat()}")

        df = pd.DataFrame(data={"time": [pd.to_datetime(rstart, utc=True), pd.to_datetime(rend, utc=True)]})
        df.set_index("time", inplace=True)
        df = cast(pd.DataFrame, df.resample('15min').ffill())
        df.reset_index(inplace=True)

        df["holiday"] = df["time"].apply(lambda t: self.is_holiday(t.astimezone(tzlocal)))
        for i in range(6):
            df[f"day_{i}"] = df["time"].apply(lambda t, i=i: 1 if t.astimezone(tzlocal).weekday() == i else 0)
        
        
        observer = Observer(latitude=statistics.mean(self.region.latitudes), longitude=statistics.mean(self.region.longitudes))

        df["sunelevation"] = df["time"].apply(lambda t: sun.elevation(observer, t))
        df["azimuth"] = df["time"].apply(lambda t: sun.azimuth(observer, t))
        df["sr_influence"] = df["time"].apply(lambda t: (t - sun.sunrise(observer, date=t)).total_seconds())
        df["ss_influence"] = df["time"].apply(lambda t: (t - sun.sunset(observer, date=t)).total_seconds())

        df["morningpeak"] = df["time"].apply(lambda t: (t - t.replace(hour=6, minute=0)).total_seconds())
        df["eveningpeak"]  = df["time"].apply(lambda t: (t - t.replace(hour=18, minute=0)).total_seconds())

        df.set_index("time", inplace=True)
        return df



    def gen_missing_date_ranges(self, start: datetime, end: datetime) -> Generator[tuple[datetime, datetime]]:
        start = start.replace(minute=0, second=0, microsecond=0)

        curr = start

        rangestart = None
        while curr <= end:
            next = curr + timedelta(minutes=15)

            if rangestart is not None and (next in self.data.index or next > end):
                yield (rangestart, curr)
                rangestart = None

            if rangestart is None and curr not in self.data.index:
                rangestart = curr

            curr = next


    def is_holiday(self, t : pd.Timestamp) -> float:
        if t.weekday() == 6:
            return 1

        date = t.date()

        cnt_holiday = sum(bool(date in h)
                      for h in self.region.holidays)
        return cnt_holiday / len(self.region.holidays)
