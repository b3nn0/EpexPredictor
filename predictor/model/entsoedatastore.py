import asyncio
import logging
from datetime import datetime, timedelta, timezone
from math import nan
import os

from entsoe import entsoe
import pandas as pd

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

class EntsoeDataStore(DataStore):
    """
    Fetches additional forecast data from Entso-E (if API key is configured)
    """

    data : pd.DataFrame
    region : PriceRegion
    storage_dir : str|None
    entsoe_api_key : str|None

    update_lock: asyncio.Lock
    

    def __init__(self, region : PriceRegion, storage_dir=None):
        super().__init__(region, storage_dir, "entsoe_v1")
        if not self.region.use_entsoe_load_forecast:
            self.data = self.data.drop(self.data.index)
        self.update_lock = asyncio.Lock()
        self.entsoe_api_key = os.getenv("EPEXPREDICTOR_ENTSOE_API_KEY", None)
        if self.entsoe_api_key is None or len(self.entsoe_api_key) == 0:
            self.entsoe_api_key = None
            log.warning("EPEXPREDICTOR_ENTSOE_API_KEY is not defined. Skipping Entso-E data. Expect reduced model performance")



    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        if self.entsoe_api_key is None or not self.region.use_entsoe_load_forecast:
            return False
        
        async with self.update_lock:
            start = start.astimezone(timezone.utc)
            end = end.astimezone(timezone.utc)

            updated = False

            for rstart, rend in self.gen_missing_date_ranges(start, end):
                updated = await self.refresh_range(rstart, rend) or updated
            
            return updated

    async def refresh_range(self, rstart: datetime, rend: datetime) -> bool:
        if self.entsoe_api_key is None or not self.region.use_entsoe_load_forecast:
            return False

        log.info(f"Fetching Entso-E data from {rstart.isoformat()} to {rend.isoformat()}")
        client = entsoe.EntsoePandasClient(api_key=self.entsoe_api_key)

        # Entso-E api always seems to cut things a bit short... and it gives us a bit of buffer for interpolation
        qstart = rstart - timedelta(days=2)
        qend = rend + timedelta(days=2)

        # A31 = daily data, week-forecast
        # Columns "Max Forecasted Load" and "Min Forecasted Load"
        load_forecast = await asyncio.to_thread(client.query_load_forecast, self.region.bidding_zone_entsoe, start=pd.to_datetime(qstart), end=pd.to_datetime(qend), process_type="A31")
        load_forecast = load_forecast.resample("15min").ffill()
        assert isinstance(load_forecast.index, pd.DatetimeIndex)

        # Max load is typically observed for morning/evening peaks, min load at night
        def resample_load_to_hourly(row):
            assert isinstance(row.name, pd.Timestamp)
            maxload = row["Max Forecasted Load"]
            minload = row["Min Forecasted Load"]
            if row.name.hour == 11 and row.name.minute == 30:
                return maxload
            elif row.name.hour == 19 and row.name.minute == 0:
                return maxload
            elif row.name.hour == 14 and row.name.minute == 30:
                return (3 * maxload + minload) / 4.0
            elif row.name.hour == 3 and row.name.minute == 0:
                return minload
            return nan

        load_hourly = load_forecast.apply(resample_load_to_hourly, axis=1)
        load_hourly = load_hourly.interpolate(method='cubic').dropna()
        load_hourly.name = "load"

        assert isinstance(load_hourly.index, pd.DatetimeIndex)
        load_hourly.index = load_hourly.index.tz_convert("UTC")

        hourly_df = pd.DataFrame(load_hourly)

        updated = False
        if len(hourly_df) > 0:
            updated = self._update_data(hourly_df)
    
        if updated:
            log.info(f"Entso-E data updated for {self.region.bidding_zone_entsoe}")
            self.data.sort_index(inplace=True)
            await self.serialize()
        return updated



    def get_next_horizon_revalidation_time(self) -> datetime | None:
        return datetime.now(timezone.utc) + timedelta(hours=3)