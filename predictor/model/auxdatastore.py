import asyncio
import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import override

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

    @override
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
                log.info(f"{self.region.bidding_zone_entsoe}: aux data updated")
                self.data.sort_index(inplace=True)
                await self.serialize()
            return updated
    
    @override
    def get_next_horizon_revalidation_time(self) -> datetime|None:
        return None

    def _compute_data(self, rstart: datetime, rend: datetime) -> pd.DataFrame:
        """
        Careful: will be called in separate thread
        """
        tzlocal = self.region.get_timezone_info()

        # make it full day to be sure
        rstart = rstart.replace(hour=0, minute=0, second=0, microsecond=0)
        rend = rend.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        log.info(f"{self.region.bidding_zone_entsoe}: computing aux data from {rstart.isoformat()} to {rend.isoformat()}")

        times: pd.DatetimeIndex = pd.date_range(rstart, rend, freq="15min")
        df = pd.DataFrame(index=times)
        df.index.name = "time"
        local = times.tz_convert(tzlocal)

        # holiday and weekday depend only on the local date, so compute once per unique date
        local_dates = local.normalize()
        df["holiday"] = local_dates.map({d: self.is_holiday(d) for d in local_dates.unique()})
        weekday = local.weekday
        for i in range(6):
            df[f"day_{i}"] = (weekday == i).astype(int)

        observer = Observer(latitude=statistics.mean(self.region.latitudes), longitude=statistics.mean(self.region.longitudes))

        # one zenith_and_azimuth call yields both angles; elevation = 90 - zenith
        zenith_azimuth = [sun.zenith_and_azimuth(observer, t) for t in times]
        df["sunelevation"] = [90.0 - z for z, _ in zenith_azimuth]
        df["azimuth"] = [a for _, a in zenith_azimuth]

        # sunrise/sunset only change once per day, so cache them per date
        sunrise_cache: dict[pd.Timestamp, datetime] = {}
        sunset_cache: dict[pd.Timestamp, datetime] = {}
        sr_influence = []
        ss_influence = []
        for t in times:
            day = t.normalize()
            if day not in sunrise_cache:
                sunrise_cache[day] = sun.sunrise(observer, date=t)
                sunset_cache[day] = sun.sunset(observer, date=t)
            sr_influence.append((t - sunrise_cache[day]).total_seconds())
            ss_influence.append((t - sunset_cache[day]).total_seconds())
        df["sr_influence"] = sr_influence
        df["ss_influence"] = ss_influence

        df["morningpeak"] = ((times.hour - 8) * 3600 + times.minute * 60).astype(float)
        df["eveningpeak"] = ((times.hour - 19) * 3600 + times.minute * 60).astype(float)

        return df


    
    def is_holiday(self, t : pd.Timestamp) -> float:
        if t.weekday() == 6:
            return 1

        date = t.date()

        cnt_holiday = sum(bool(date in h)
                      for h in self.region.holidays)
        return cnt_holiday / len(self.region.holidays)
