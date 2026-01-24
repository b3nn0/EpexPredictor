import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
import os
from typing import override

import aiohttp
from entsoe import entsoe
import pandas as pd

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

class PriceStore(DataStore):
    """
    Fetches and caches price info from api.energy-charts.info
    TODO: add more price sources, e.g. for SE1-SE4, which is not available from energy-charts
    """

    data: pd.DataFrame
    region: PriceRegion
    storage_dir: str|None

    update_lock: asyncio.Lock

    entsoe_api_key: str|None = None
    

    def __init__(self, region : PriceRegion, storage_dir=None):
        super().__init__(region, storage_dir, "prices_v3")
        self.update_lock = asyncio.Lock()

        self.entsoe_api_key = os.getenv("EPEXPREDICTOR_ENTSOE_API_KEY", None)
        if self.entsoe_api_key is None or len(self.entsoe_api_key) == 0:
            self.entsoe_api_key = None
            logging.warning("EPEXPREDICTOR_ENTSOE_API_KEY is not defined. Not all bidding zones are available.")

    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        # Refresh more often when the horizon is fairly small (after 13:00 local time if the following day is not yet known)
        price_update_frequency = 12 * 60 * 60
        last_known = self.get_last_known()
        assert last_known is not None
        if (last_known - datetime.now(timezone.utc)).total_seconds() <= 60 * 60 * 11:
            price_update_frequency = 5 * 60
        return datetime.now(timezone.utc) + timedelta(seconds=price_update_frequency) 


    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        async with self.update_lock:
            start = start.astimezone(timezone.utc)
            end = end.astimezone(timezone.utc)

            updated = False

            for rstart, rend in self.gen_missing_date_ranges(start, end):
                prices = await self.fetch_prices_best_try(rstart, rend)
                if prices is not None:
                    self._update_data(prices)
                    updated = True
        
            if updated:
                log.info(f"price data updated for {self.region.bidding_zone_entsoe}")
                self.data.sort_index(inplace=True)
                self.serialize()
            return updated


    async def fetch_prices_best_try(self, rstart: datetime, rend: datetime) -> pd.DataFrame | None:
        last_known_before = self.get_last_known()
        df = await self.fetch_prices_energycharts(rstart, rend)
        if df is not None and len(df) > 0:
            return df
        
        df = await self.fetch_prices_entsoe(rstart, rend)
        if df is not None and len(df) > 0:
            return df
    
        before_str = last_known_before.isoformat() if last_known_before else "never"
        logging.info(f"Unable to fetch prices for {self.region.bidding_zone_entsoe} - no newer prices available from any provider. Prices available until {before_str}")
        return None
    
    async def fetch_prices_energycharts(self, rstart: datetime, rend: datetime) -> pd.DataFrame | None:
        try:
            if self.region.bidding_zone_energycharts is not None:
                start_of_day = rstart.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = rend.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                start_formatted = start_of_day.isoformat().replace("+00:00", "Z")
                end_formatted = end_of_day.isoformat().replace("+00:00", "Z")
                url = f"https://api.energy-charts.info/price?bzn={self.region.bidding_zone_energycharts}&start={start_formatted}&end={end_formatted}"
                log.info(f"Fetching price data for {self.region.bidding_zone_energycharts}: {url}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers={"accept": "application/json"}) as resp:
                        txt = await resp.text()
                        if "no content available" in txt:
                            return None
                        
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

                        return df
        except Exception as e:
            logging.error(f"Failed to fetch prices from energy-charts: {e}")


    async def fetch_prices_entsoe(self, rstart: datetime, rend: datetime) -> pd.DataFrame | None:
        try:
            if self.entsoe_api_key is None or self.region.bidding_zone_entsoe is None:
                return None

            logging.info(f"Fetching prices from {rstart.isoformat()} to {rend.isoformat()} from Entso-E")
            client = entsoe.EntsoePandasClient(api_key=self.entsoe_api_key)
            prices_series = await asyncio.to_thread(client.query_day_ahead_prices, self.region.bidding_zone_entsoe, pd.to_datetime(rstart), pd.to_datetime(rend))
            prices = prices_series.to_frame("price")
            prices["price"] = prices["price"] / 10
            prices.index = prices.index.tz_convert("UTC") # type: ignore
            prices = prices.resample("15min").ffill().bfill()
            return prices
        except Exception as e:
            logging.error(f"Failed to fetch prices from entso-e: {e}")



