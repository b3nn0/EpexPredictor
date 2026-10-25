import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
import os
from typing import override
from zoneinfo import ZoneInfo

import aiohttp
from aiohttp import ClientTimeout
from entsoe import entsoe
import pandas as pd

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

class PriceStore(DataStore):
    """
    Fetches and caches price info from api.energy-charts.info
    """

    data: pd.DataFrame
    region: PriceRegion
    storage_dir: str|None

    update_lock: asyncio.Lock

    entsoe_api_key: str|None = None
    use_nordpool: bool = False
   

    def __init__(self, region : PriceRegion, storage_dir=None):
        super().__init__(region, storage_dir, "prices_v3")
        self.update_lock = asyncio.Lock()

        self.entsoe_api_key = os.getenv("EPEXPREDICTOR_ENTSOE_API_KEY", None)
        if self.entsoe_api_key is None or len(self.entsoe_api_key) == 0:
            self.entsoe_api_key = None
            log.warning("EPEXPREDICTOR_ENTSOE_API_KEY is not defined. Not all bidding zones are available.")

        # Nord Pool data is only used when explicitly enabled (licensing depends on use case)
        self.use_nordpool = os.getenv("EPEXPREDICTOR_USE_NORDPOOL", "").strip().lower() in ("1", "true", "yes", "on")

    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        # Day-ahead prices for the following day are published around 13:00 local time.
        _PUBLISH_HOUR = 13
        
        # Decide the next check from the last known price cutoff, not just wall-clock time.
        tz = self.region.get_timezone_info()
        localnow = datetime.now(tz=tz)
        now = datetime.now(timezone.utc)

        last_known = self.get_last_known()
        if last_known is None:
            # No data at all (cold start) -> fetch immediately
            log.info(f"{self.region.bidding_zone_entsoe}: no price data yet -> fetching immediately")
            return now

        # The horizon the source should already cover:
        #   before 13:00 local -> up to end of today
        #   after  13:00 local -> up to end of tomorrow
        # Use noon of the relevant day as the "is this day known?" marker.
        today = localnow.replace(hour=12, minute=0, second=0, microsecond=0)
        if localnow.hour < _PUBLISH_HOUR:
            expected_marker = today
            next_publish = localnow.replace(hour=_PUBLISH_HOUR, minute=0, second=0, microsecond=0)
        else:
            expected_marker = today + timedelta(days=1)
            next_publish = expected_marker.replace(hour=_PUBLISH_HOUR, minute=0, second=0, microsecond=0)

        if last_known < today:
            # We don't even have today's prices -> cache is stale (e.g. startup). Fetch now.
            log.info(f"{self.region.bidding_zone_entsoe}: prices only known until {last_known.isoformat()} -> fetching immediately")
            return now

        if last_known >= expected_marker:
            # Expected horizon is already in the data -> wait for the next publish window.
            nextupdate = next_publish.astimezone(timezone.utc)
            log.info(f"{self.region.bidding_zone_entsoe}: prices known until {last_known.isoformat()}. Next update: {nextupdate.isoformat()}")
            return nextupdate

        # Today is known but the next day's batch isn't published yet -> retry in 5 minutes.
        # The refresh loop re-invokes this on each wake, giving 13:05, 13:10, 13:15, ...
        nextupdate = now + timedelta(minutes=5)
        log.info(f"{self.region.bidding_zone_entsoe}: expecting new prices soon -> next price update: {nextupdate.isoformat()}")
        return nextupdate

    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        """
        Fetch missing price data from energy-charts.info or ENTSO-E.
        
        This method attempts to fetch price data from energy-charts.info first.
        If energy-charts returns data but no update is necessary (i.e., _update_data returns False),
        it will fall back to ENTSO-E to ensure we have the most recent data available.
        
        Args:
            start: Start date/time for the data range
            end: End date/time for the data range
            
        Returns:
            True if new data was fetched and updated, False otherwise
        """
        async with self.update_lock:
            start = start.astimezone(timezone.utc)
            end = end.astimezone(timezone.utc)

            updated = False
            checked = False

            for rstart, rend in self.gen_missing_date_ranges(start, end):
                checked = True
                updated |= await self._fetch_and_update_from_energycharts(rstart, rend)
                if not updated:
                    # If energy-charts didn't update anything, try ENTSO-E as fallback
                    updated |= await self._fetch_and_update_from_entsoe(rstart, rend)
                if not updated:
                    # Nord Pool is lowest priority (only used if explicitly enabled)
                    updated |= await self._fetch_and_update_from_nordpool(rstart, rend)

        
            if updated:
                log.info(f"{self.region.bidding_zone_entsoe}: price data updated")
                self.data.sort_index(inplace=True)
                await self.serialize()
            elif checked:
                log.info(f"{self.region.bidding_zone_entsoe}: unable to fetch prices - no newer prices available from any provider. Prices available until {self.get_last_known()}")
            return updated

    async def _fetch_and_update_from_energycharts(self, rstart: datetime, rend: datetime) -> bool:
        """Fetch price data from energy-charts and update cache if new data is available."""
        prices = await self.fetch_prices_energycharts(rstart, rend)
        if prices is not None and len(prices) > 0:
            if self._is_invalid_zero_data(prices):
                log.warning(f"{self.region.bidding_zone_entsoe}: discarding zero-price data from energy-charts")
                return False
            return self._update_data(prices)
        return False

    async def _fetch_and_update_from_entsoe(self, rstart: datetime, rend: datetime) -> bool:
        """Fetch price data from ENTSO-E and update cache if new data is available."""
        log.info(f"{self.region.bidding_zone_entsoe}: trying ENTSO-E fallback")
        entsoe_prices = await self.fetch_prices_entsoe(rstart, rend)
        if entsoe_prices is not None and len(entsoe_prices) > 0:
            if self._is_invalid_zero_data(entsoe_prices):
                log.warning(f"{self.region.bidding_zone_entsoe}: discarding zero-price data from ENTSO-E")
                return False
            return self._update_data(entsoe_prices)
        return False

    async def _fetch_and_update_from_nordpool(self, rstart: datetime, rend: datetime) -> bool:
        """Fetch price data from Nord Pool and update cache if new data is available."""
        log.info(f"{self.region.bidding_zone_entsoe}: trying Nord Pool fallback")
        nordpool_prices = await self.fetch_prices_nordpool(rstart, rend)
        if nordpool_prices is not None and len(nordpool_prices) > 0:
            if self._is_invalid_zero_data(nordpool_prices):
                log.warning(f"{self.region.bidding_zone_entsoe}: discarding zero-price data from Nord Pool")
                return False
            return self._update_data(nordpool_prices)
        return False

    def _is_invalid_zero_data(self, df: pd.DataFrame) -> bool:
        """Check if the last 24 hours of data are all zero (invalid)."""
        last_24h = df.tail(20 * 4)
        return bool((last_24h["price"] == 0).all())

    
    async def fetch_prices_energycharts(self, rstart: datetime, rend: datetime) -> pd.DataFrame | None:
        try:
            if self.region.bidding_zone_energycharts is not None:
                # The market day runs on local time. Snapping to UTC midnight cuts the
                # first hours of the day off the request for any zone east of UTC.
                tzlocal = self.region.get_timezone_info()
                start_of_day = rstart.astimezone(tzlocal).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
                end_of_day = (rend.astimezone(tzlocal).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).astimezone(timezone.utc)
                start_formatted = start_of_day.isoformat().replace("+00:00", "Z")
                end_formatted = end_of_day.isoformat().replace("+00:00", "Z")
                url = f"https://api.energy-charts.info/price?bzn={self.region.bidding_zone_energycharts}&start={start_formatted}&end={end_formatted}"
                log.info(f"{self.region.bidding_zone_entsoe}: fetching price data: {url}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers={"accept": "application/json"}, timeout=ClientTimeout(total=8)) as resp:
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
            log.error(f"{self.region.bidding_zone_entsoe}: failed to fetch prices from energy-charts: {e}")


    async def fetch_prices_entsoe(self, rstart: datetime, rend: datetime) -> pd.DataFrame | None:
        try:
            if self.entsoe_api_key is None or self.region.bidding_zone_entsoe is None:
                return None
            # Entso-E always response a bit tight...
            qstart = rstart - timedelta(days=1)
            qend = rend + timedelta(days=2)
            log.info(f"{self.region.bidding_zone_entsoe}: fetching prices from {rstart.isoformat()} to {rend.isoformat()} from Entso-E")
            client = entsoe.EntsoePandasClient(api_key=self.entsoe_api_key)
            prices_series = await asyncio.to_thread(client.query_day_ahead_prices, self.region.bidding_zone_entsoe, pd.to_datetime(qstart), pd.to_datetime(qend))
            prices = prices_series.to_frame("price")
            prices["price"] = prices["price"] / 10
            prices.index = prices.index.tz_convert("UTC") # type: ignore
            prices = prices.resample("15min").ffill().bfill()
            return prices
        except Exception as e:
            log.error(f"{self.region.bidding_zone_entsoe}: failed to fetch prices from entso-e: {e}")

    async def fetch_prices_nordpool(self, rstart: datetime, rend: datetime) -> pd.DataFrame | None:
        """Fetch day-ahead prices from the Nord Pool data portal API (EUR/MWh, 15min slots, UTC)."""
        try:
            if not self.use_nordpool or self.region.bidding_zone_nordpool is None:
                return None
            # Nord Pool only provides ~4 weeks of history - don't query the API for older data
            if rstart < datetime.now(timezone.utc) - timedelta(days=28):
                log.info(f"{self.region.bidding_zone_entsoe}: nordpool only provides ~4 weeks of history, skipping {rstart.isoformat()}")
                return None
            area = self.region.bidding_zone_nordpool
            # Delivery dates are calendar days in CET, one API request per day
            tzcet = ZoneInfo("Europe/Berlin")
            day = rstart.astimezone(tzcet).date()
            last_day = rend.astimezone(tzcet).date()
            frames = []
            async with aiohttp.ClientSession() as session:
                while day <= last_day:
                    url = f"https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices?date={day.isoformat()}&market=DayAhead&deliveryArea={area}&currency=EUR"
                    log.info(f"{self.region.bidding_zone_entsoe}: fetching price data from nordpool: {url}")
                    async with session.get(url, headers={"accept": "application/json"}, timeout=ClientTimeout(total=8)) as resp:
                        if resp.status == 204:
                            # Prices are only published up to ~the next day. An empty day means all
                            # later days are unpublished too, so stop here.
                            log.info(f"{self.region.bidding_zone_entsoe}: nordpool has no prices yet for {day.isoformat()}, stopping")
                            break
                        if resp.status != 200:
                            log.warning(f"{self.region.bidding_zone_entsoe}: nordpool returned HTTP {resp.status} for {day.isoformat()}")
                            return None
                        data = await resp.json()
                    for entry in data.get("multiAreaEntries", []):
                        price = entry.get("entryPerArea", {}).get(area)
                        if price is not None:
                            frames.append((entry["deliveryStart"], price))
                    day += timedelta(days=1)
            if len(frames) == 0:
                return None
            df = pd.DataFrame.from_dict(dict(frames), orient="index", columns=["price"])
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.loc[rstart:rend]
            df["price"] = df["price"] / 10
            df.index.name = "time"
            df = df.sort_index().resample("15min").ffill().bfill()
            return df
        except Exception as e:
            log.error(f"{self.region.bidding_zone_entsoe}: failed to fetch prices from nordpool: {e}")



