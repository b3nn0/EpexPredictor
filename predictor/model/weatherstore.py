import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Generator, override

import aiohttp
import pandas as pd
from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

class WeatherStore(DataStore):
    """
    Fetches and caches weather data from OpenMeteo
    TODO: add management of allowed API calls, delay queries if needed
    """

    data: pd.DataFrame
    region: PriceRegion
    storage_dir: str|None

    update_lock: asyncio.Lock
    

    def __init__(self, region : PriceRegion, storage_dir: str|None =None):
        super().__init__(region, storage_dir, "weather_v2")
        self.update_lock = asyncio.Lock()

    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        return self.last_updated + timedelta(hours=6)

    async def refresh_range(self, rstart: datetime, rend: datetime) -> bool:
        async with self.update_lock:
            lats = ",".join(map(str, self.region.latitudes))
            lons = ",".join(map(str, self.region.longitudes))

            updated = False
            if self.needs_history_query(rstart):
                host = "historical-forecast-api.open-meteo.com"
            else:
                host = "api.open-meteo.com"

            url = f"https://{host}/v1/forecast?latitude={lats}&longitude={lons}&azimuth=0&tilt=0&start_date={rstart.date().isoformat()}&end_date={rend.date().isoformat()}&minutely_15=wind_speed_80m,temperature_2m,global_tilted_irradiance,pressure_msl,relative_humidity_2m&timezone=UTC"
            log.info(f"Fetching weather data for {self.region.bidding_zone_entsoe}: {url}")

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        data = await resp.text()

                        data = json.loads(data)
                        frames = []
                        for i, fc in enumerate(data):
                            df = pd.DataFrame()

                            df["time"] = fc["minutely_15"]["time"]
                            df[f"wind_{i}"] = fc["minutely_15"]["wind_speed_80m"]
                            df[f"temp_{i}"] = fc["minutely_15"]["temperature_2m"]
                            df[f"irradiance_{i}"] = fc["minutely_15"]["global_tilted_irradiance"]
                            df[f"pressure_{i}"] = fc["minutely_15"]["pressure_msl"]        
                            df[f"humidity_{i}"] = fc["minutely_15"]["relative_humidity_2m"]
                            
                            df.set_index("time", inplace=True)
                            df = df.dropna()
                            frames.append(df)

                        df = pd.concat(frames, axis=1).reset_index()
                        df["time"] = pd.to_datetime(df["time"], utc=True)
                        df.set_index("time", inplace=True)

                        updated = self._update_data(df) or updated
            except Exception as e:
                log.warning(f"Failed to fetch weather data...: error: {str(e)}")
                raise e
            finally:
                if updated:
                    log.info(f"weather data updated for {self.region.bidding_zone_entsoe}")
                    self.data.sort_index(inplace=True)
                    await self.serialize()

            return updated

    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)

        updated = False

        for rstart, rend in self.gen_missing_date_ranges(start, end):
            updated = await self.refresh_range(rstart, rend) or updated

        return updated

    @override
    def gen_missing_date_ranges(self, start: datetime, end: datetime) -> Generator[tuple[datetime, datetime]]:
        # a random hour of the day so we can easily check if we already have that day.
        # OpenMeteo only has full day queries anyway.
        start = start.replace(hour=12, minute=0, second=0, microsecond=0)
        end = end.replace(hour=12, minute=0, second=0, microsecond=0)

        curr = start

        rangestart = None
        while curr <= end:
            next_day = curr + timedelta(days=1)

            apiswitch = rangestart is not None and self.needs_history_query(rangestart) != self.needs_history_query(next_day)


            if rangestart is not None and (next_day in self.data.index or next_day > end or apiswitch or (curr - rangestart).total_seconds() > 60 * 60 * 24 * 90):
                # We have the next timeslot already OR its the last timeslot OR the current range exceeds 90 days (max for openmeteo) OR we need to change APIs
                yield (rangestart, curr)
                rangestart = None

            if rangestart is None and curr not in self.data.index:
                rangestart = curr

            curr = next_day

    def needs_history_query(self, dt: datetime) -> bool:
        """
        If query is older than 90 days, we need OpenMeteo's historical data API.
        We switch to historical API at 60 days to be safe.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=60)
        return dt < cutoff
