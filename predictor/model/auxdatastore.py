import logging
from datetime import datetime, timedelta, timezone, tzinfo
import statistics
from typing import Generator, cast

from astral import LocationInfo, sun
import pandas as pd
import pytz
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


    def __init__(self, region : PriceRegion, storage_dir: str | None = None):
        super().__init__(region)


    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)

        updated = False

        tzlocal = pytz.timezone(self.region.timezone)

        for rstart, rend in self.gen_missing_date_ranges(start, end):
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
            
            timecols : list[pd.Series|pd.DataFrame] = []
            for h in range(24):
                for m in range(0, 60, 15):
                    col = df["time"].apply(lambda t, h=h, m=m: self.is_timestamp(tzlocal, t, h, m))
                    col.name = f"i_{h}_{m}"
                    timecols.append(col)
            
            locinfo = LocationInfo(name=self.region.country_code, region=self.region.country_code, timezone=self.region.timezone, latitude=statistics.mean(self.region.latitudes), longitude=statistics.mean(self.region.longitudes))
            sr_influence = df["time"].apply(lambda t, loc=locinfo: min(180, abs((t - sun.sun(loc.observer, date=t)["sunrise"]).total_seconds() / 60)))
            sr_influence.name = "sr_influence"
            timecols.append(sr_influence)

            ss_influence = df["time"].apply(lambda t, loc=locinfo: min(180, abs((t - sun.sun(loc.observer, date=t)["sunset"]).total_seconds() / 60)))
            ss_influence.name = "ss_influence"
            timecols.append(ss_influence)

            timecols.insert(0, df)
            df = pd.concat(timecols, axis=1)
            df.set_index("time", inplace=True)

            self._update_data(df)

            updated = True

    
        if updated:
            log.info(f"aux data updated for {self.region.bidding_zone}")
            self.data.sort_index(inplace=True)
            self.serialize()
        return updated

    def gen_missing_date_ranges(self, start: datetime, end: datetime) -> Generator[tuple[datetime, datetime]]:
        start = start.replace(hour=12, minute=0, second=0, microsecond=0)

        curr = start

        rangestart = None
        while curr <= end:
            next_day = curr + timedelta(days=1)

            if rangestart is not None and (next_day in self.data.index or next_day > end):
                yield (rangestart, curr)
                rangestart = None

            if rangestart is None and curr not in self.data.index:
                rangestart = curr

            curr = next_day


    def is_timestamp(self, tz : tzinfo, t : datetime, h : int, m : int) -> int:
        local = t.astimezone(tz)
        return 1 if local.hour == h and local.minute == m else 0
    
    def is_holiday(self, t : pd.Timestamp) -> float:
        if t.weekday() == 6:
            return 1

        date = t.date()

        cnt_holiday = sum(bool(date in h)
                      for h in self.region.holidays)
        return cnt_holiday / len(self.region.holidays)
