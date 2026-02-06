import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import override

import aiohttp
import pandas as pd

from .datastore import DataStore
from .priceregion import PriceRegion, PriceRegionName

log = logging.getLogger(__name__)

class GasPriceStore(DataStore):
    """
    Fetches and caches natural gas prices from bundesnetzagentur.de. Only German gas prices supported for now, but should serve as a 
    rough indication for other markets, too.
    See https://www.bundesnetzagentur.de/DE/Gasversorgung/aktuelle_gasversorgung/_svg/Gaspreise/Gaspreise.html
    """

    data : pd.DataFrame
    region : PriceRegion
    storage_dir : str|None

    update_lock: asyncio.Lock
    

    def __init__(self, region : PriceRegion, storage_dir=None):
        super().__init__(region, storage_dir, "gasprices")
        self.update_lock = asyncio.Lock()



    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        async with self.update_lock:
            if not self.region.use_de_nat_gas_price:
                return False

            start = start.astimezone(timezone.utc)
            end = end.astimezone(timezone.utc)

            updated = False

            for rstart, rend in self.gen_missing_date_ranges(start, end):
                tzgerman = PriceRegionName.DE.to_region().get_timezone_info()
                qstart = rstart - timedelta(days=5) # sometimes a few days are missing - make sure we always try to cover the requested time range
                qend = rend + timedelta(days=5)
                start_formatted = qstart.astimezone(tzgerman).strftime("%d.%m.%Y")
                end_formatted = qend.astimezone(tzgerman).strftime("%d.%m.%Y")

                url = f"https://www.bundesnetzagentur.de/_tools/SVG/js2/_functions/json.html?view=json&id=870302&xMin={start_formatted}&xMax={end_formatted}&singleType=1"
                log.info(f"Fetching natural gas price data for {self.region.bidding_zone_entsoe}: {url}")

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers={"accept": "application/json"}) as resp:
                        txt = await resp.text()
                        try:
                            data = json.loads(txt)

                            timestamps = data["labels"]
                            prices = data["datasets"][1]["data"]
                            pricedict = {}

                            for i, t in enumerate(timestamps):
                                gasprice = prices[i]
                                if isinstance(gasprice, float): # sometimes "null" ??
                                    time = pd.to_datetime(datetime.strptime(t, "%d.%m.%Y").replace(tzinfo=timezone.utc))
                                    pricedict[time] = gasprice
                            
                            df = pd.DataFrame.from_dict(pricedict, orient="index", columns=["gasprice"])
                            df = df.resample('15min').ffill()

                            updated = self._update_data(df) or updated
                        except Exception as e:
                            log.warning(f"failed to update gas prices. Probably no data available for given time range - ignoring error: {e}")
                            raise e
                        

        
            if updated:
                log.info("gas price data updated")
                await self.serialize()

            return updated


    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        return datetime.now(timezone.utc) + timedelta(hours=12)