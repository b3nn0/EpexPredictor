import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import override

import aiohttp
import pandas as pd

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

class GasPriceStore(DataStore):
    """
    Fetches and caches day-ahead natural gas prices from instrat.pl 
    See https://www.bundesnetzagentur.de/DE/Gasversorgung/aktuelle_gasversorgung/_svg/Gaspreise/Gaspreise.html
    and https://energy.instrat.pl/en/prices/gas-dam/
    Unfortunately, BNA is not a proper API any more, so we fall back to instrat as a proxy.
    """

    data : pd.DataFrame
    region : PriceRegion
    storage_dir : str|None

    update_lock: asyncio.Lock
    

    def __init__(self, region : PriceRegion, storage_dir=None):
        super().__init__(region, storage_dir, "gasprices_v2")
        self.update_lock = asyncio.Lock()



    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        async with self.update_lock:
            if not self.region.use_de_nat_gas_price:
                return False

            start = start.astimezone(timezone.utc)
            end = end.astimezone(timezone.utc)

            updated = False

            for rstart, rend in self.gen_missing_date_ranges(start, end):
                qstart = rstart - timedelta(days=1)
                qend = rend + timedelta(days=7) # last few days sometimes missing from result?
                start_formatted = qstart.strftime("%d-%m-%YT%H:%M:%SZ")
                end_formatted = qend.strftime("%d-%m-%YT%H:%M:%SZ")

                url = f"https://energy-api.instrat.pl/api/prices/gas_price_rdn_daily?date_from={start_formatted}&date_to={end_formatted}&aggregation_timeframe=day&aggregation_type=avg"
                log.info(f"{self.region.bidding_zone_entsoe}: fetching natural gas price data: {url}")

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers={"accept": "application/json"}) as resp:
                        txt = await resp.text()
                        try:
                            data = json.loads(txt)
                            df = pd.DataFrame.from_dict(data).drop(columns=["volume", "indeks"])
                            df = df.set_index("date")
                            df.index.name = "time"
                            df.index = pd.to_datetime(df.index)

                            df = df.rename(columns={"price": "gasprice"})

                            df = df.resample('15min').ffill()

                            self._update_data(df)
                            updated = True
                        except Exception as e:
                            log.warning(f"{self.region.bidding_zone_entsoe}: failed to update gas prices. Probably no data available for given time range - ignoring error: {e}")

        
            if updated:
                log.info(f"{self.region.bidding_zone_entsoe}: gas price data updated")
                await self.serialize()
            return updated


    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        return datetime.now(timezone.utc) + timedelta(hours=12)