import asyncio
import bisect
from io import BytesIO
import logging
import os
from matplotlib.figure import Figure
import pandas as pd
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from predictor.model.priceregion import PriceRegion, PriceRegionName
import predictor.model.pricepredictor as pp

app = FastAPI(title="EPEX day-ahead prediction API", description="""
API can be used free of charge on a fair use premise.
There are no guarantees on availability or correctnes of the data.
This is an open source project, feel free to host it yourself. [Source code and docs](https://github.com/b3nn0/EpexPredictor)

### Attribution
Electricity prices provided under CC-BY-4.0 by [energy-charts.info](https://api.energy-charts.info/)

[Weather data by Open-Meteo.com](https://open-meteo.com/)
""")


logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)




logging.getLogger("uvicorn.error").handlers.clear()
logging.getLogger("uvicorn.error").handlers.extend(logging.getLogger().handlers)
logging.getLogger("uvicorn.access").handlers.clear()
logging.getLogger("uvicorn.access").handlers.extend(logging.getLogger().handlers)

log = logging.getLogger(__name__)

@app.get("/",  include_in_schema=False)
def api_docs():
    return RedirectResponse("/docs")


USE_PERSISTENT_TESTDATA = os.getenv("USE_PERSISTENT_TEST_DATA", "false").lower() in ("yes", "true", "t", "1")
EPEXPREDICTOR_DATADIR = os.getenv("EPEXPREDICTOR_DATADIR")
TRAINING_DAYS = 120
DEFAULT_TIMEZONE = "Europe/Berlin"


class PriceUnit(str, Enum):
    CT_PER_KWH = "CT_PER_KWH" #1.0
    EUR_PER_KWH = "EUR_PER_KWH"# 1 / 100.0
    EUR_PER_MWH = "EUR_PER_MWH"# 1 / 100.0 * 1000

    def convert(self, ct_per_kwh) -> float:
        if self.value == self.EUR_PER_KWH:
            return ct_per_kwh / 100.0
        elif self.value == self.EUR_PER_MWH:
            return ct_per_kwh / 100.0 * 1000
        return ct_per_kwh

class OutputFormat(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class PriceModel(BaseModel):
    """Price at a specific time. Output-only model, uses camelCase for API compatibility."""

    starts_at: datetime = Field(serialization_alias="startsAt")
    total: float

class PricesModelShort(BaseModel):
    s: list[int]
    t: list[float]

class PricesModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prices: list[PriceModel]
    known_until: datetime = Field(serialization_alias="knownUntil")


    
class RegionPriceManager:
    predictor : pp.PricePredictor

    last_weather_update : datetime = datetime(1980, 1, 1, tzinfo=timezone.utc)
    last_price_update : datetime = datetime(1980, 1, 1, tzinfo=timezone.utc)

    last_known_price : datetime = datetime.now(timezone.utc)

    cachedprices : Dict[datetime, float] = {}
    cachedeval : Dict[datetime, float] = {}

    update_task: asyncio.Task | None = None

    def __init__(self, region: PriceRegion):
        self.predictor = pp.PricePredictor(region, storage_dir=EPEXPREDICTOR_DATADIR)

    def _normalize_start_ts(self, start_ts: datetime | None, tz: ZoneInfo) -> datetime:
        """Normalize start_ts to the target timezone."""
        if start_ts is None:
            return datetime.now(tz=tz)
        if start_ts.tzinfo is None:
            return start_ts.replace(tzinfo=tz)
        return start_ts.astimezone(tz)

    def _compute_hourly_averages(self, prediction: Dict[datetime, float], tz: ZoneInfo) -> Dict[datetime, float]:
        """Compute hourly averages from 15-minute interval predictions."""
        hourly_averages: Dict[datetime, list] = {}
        for dt in sorted(prediction.keys()):
            hour_key = dt.astimezone(tz).replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_averages:
                hourly_averages[hour_key] = []
            hourly_averages[hour_key].append(prediction[dt])
        return {hour_dt: sum(prices) / len(prices) for hour_dt, prices in hourly_averages.items() if prices}

    async def prices(self, hours: int = -1, fixed_price: float = 0.0, tax_percent: float = 0.0, start_ts: datetime | None = None,
                    unit: PriceUnit = PriceUnit.CT_PER_KWH, evaluation: bool = False, hourly: bool = False,
                    timezone: str = DEFAULT_TIMEZONE, format: OutputFormat = OutputFormat.LONG) -> PricesModel | PricesModelShort:

        await self.update_in_background()

        tz = ZoneInfo(timezone)
        start_ts = self._normalize_start_ts(start_ts, tz)
        end_ts = start_ts + timedelta(hours=hours) if hours >= 0 else datetime(2999, 1, 1, tzinfo=tz)

        prediction = self.cachedeval if evaluation else self.cachedprices
        if hourly:
            prediction = self._compute_hourly_averages(prediction, tz)

        prices: list[PriceModel] = []
        dts = sorted(prediction.keys())
        start_index = max(0, bisect.bisect_right(dts, start_ts) - 1)
        end_index = min(len(dts) - 1, bisect.bisect_right(dts, end_ts))

        for dt in dts[start_index:end_index]:
            total = (prediction[dt] + fixed_price) * (1 + tax_percent / 100.0)
            total = unit.convert(total)
            prices.append(PriceModel(starts_at=dt.astimezone(tz), total=round(total, 4)))

        if format == OutputFormat.SHORT:
            return self.format_short(prices)
        return PricesModel(prices=prices, known_until=self.last_known_price.astimezone(tz))

        
    def format_short(self, prices: List[PriceModel]) -> PricesModelShort:
        return PricesModelShort(
            s=[round(p.starts_at.timestamp()) for p in prices],
            t=[round(p.total, 4) for p in prices]
        )


    async def update_in_background(self):
        if self.update_task is None:
            self.update_task = asyncio.create_task(self.update_data_if_needed())
        
        if len(self.cachedprices) == 0:
            await self.update_task # sync refresh on first call


    async def update_data_if_needed(self):
        try:
            currts = datetime.now(timezone.utc)

            price_age = currts - self.last_price_update
            weather_age = currts - self.last_weather_update

            self.is_currently_updating = True

            # Update prices every 12 hours. If it's after 13:00 local, and we don't have prices for the next day yet, update every 5 minutes
            latest_price = self.predictor.pricestore.get_last_known()
            price_update_frequency = 12 * 60 * 60
            if latest_price is None or (latest_price - datetime.now(timezone.utc)).total_seconds() <= 60 * 60 * 11:
                price_update_frequency = 5 * 60

            retrain = False
            if price_age.total_seconds() > price_update_frequency:
                self.last_price_update = currts
                if await self.predictor.refresh_prices():
                    lastknown = self.predictor.pricestore.get_last_known()
                    if lastknown is not None:
                        log.info(f"Prices updated, now available until {lastknown.isoformat()}")
                    retrain = True


            if weather_age.total_seconds() > 60 * 60 * 3:  # update weather every 3 hours
                start = datetime.now(timezone.utc) - timedelta(days=1)
                end = datetime.now(timezone.utc) + timedelta(days=8)
                await self.predictor.refresh_weather(start, end)
                self.last_weather_update = currts
                retrain = True

            if retrain:
                train_start = datetime.now(timezone.utc) - timedelta(days=TRAINING_DAYS)
                train_end = datetime.now(timezone.utc) + timedelta(days=7) # will ensure all weather data is fetched immediately, not partially for training and then partially for prediction
                
               
                await self.predictor.train(train_start, train_end)
                newprices, neweval = await self.predictor.predict(train_start, train_end), await self.predictor.predict(train_start, train_end, fill_known=False)
                self.cachedprices = self.predictor.to_price_dict(newprices)
                self.cachedeval = self.predictor.to_price_dict(neweval)
                lastknown = self.predictor.pricestore.get_last_known()
                if lastknown is not None:
                    self.last_known_price = lastknown

                self.predictor.cleanup()

        finally:
            self.update_task = None



class Prices:
    region_prices: Dict[PriceRegion, RegionPriceManager] = {}

    async def prices(self, hours: int = -1, fixed_price: float = 0.0, tax_percent: float = 0.0, start_ts: datetime | None = None,
                    region: PriceRegion = PriceRegion.DE, unit: PriceUnit = PriceUnit.CT_PER_KWH, evaluation: bool = False, hourly: bool = False,
                    timezone: str = DEFAULT_TIMEZONE, format: OutputFormat = OutputFormat.LONG):
        if region not in self.region_prices:
            self.region_prices[region] = RegionPriceManager(region)
        return await self.region_prices[region].prices(hours, fixed_price, tax_percent, start_ts, unit, evaluation, hourly, timezone, format)
    
    def get_predictor(self, region):
        if region not in self.region_prices:
            self.region_prices[region] = RegionPriceManager(region)
        return self.region_prices[region].predictor


prices_handler = Prices()


@app.get("/prices")
async def get_prices(
    hours: int = Query(-1, description="How many hours to predict"),
    fixed_price: float = Query(0.0, description="Add this fixed amount to all prices (ct/kWh)", alias="fixedPrice"),
    tax_percent: float = Query(0.0, description="Tax % to add to the final price", alias="taxPercent"),
    start_ts: datetime | None = Query(None, description="Start output from this time. At most ~90 days in the past", alias="startTs"),
    region: PriceRegionName = Query(PriceRegionName.DE, description="Region/bidding zone", alias="country"),
    evaluation: bool = Query(False, description="Switches to evaluation mode. All values will be generated by the model, instead of only future values. Useful to evaluate model performance."),
    unit: PriceUnit = Query(PriceUnit.CT_PER_KWH, description="Unit of output"),
    hourly: bool = Query(False, description="Output hourly average prices (if your energy provider uses hourly prices)"),
    timezone: str = Query(DEFAULT_TIMEZONE, description=f"Timezone for startTs and output timestamps. Default is {DEFAULT_TIMEZONE}")) -> PricesModel:
    """
    Get price prediction - verbose output format with objects containing full ISO timestamp and price
    """
    res = await prices_handler.prices(hours, fixed_price, tax_percent, start_ts, region.to_region(), unit, evaluation, hourly, timezone, format=OutputFormat.LONG)
    assert isinstance(res, PricesModel)
    return res


@app.get("/prices_short")
async def get_prices_short(
    hours: int = Query(-1, description="How many hours to predict"),
    fixed_price: float = Query(0.0, description="Add this fixed amount to all prices (ct/kWh)", alias="fixedPrice"),
    tax_percent: float = Query(0.0, description="Tax % to add to the final price", alias="taxPercent"),
    start_ts: datetime | None = Query(None, description="Start output from this time. At most ~90 days in the past", alias="startTs"),
    region: PriceRegionName = Query(PriceRegionName.DE, description="Region/bidding zone", alias="country"),
    evaluation: bool = Query(False, description="Switches to evaluation mode. All values will be generated by the model, instead of only future values. Useful to evaluate model performance."),
    unit: PriceUnit = Query(PriceUnit.CT_PER_KWH, description="Unit of output"),
    hourly: bool = Query(False, description="Output hourly average prices (if your energy provider uses hourly prices)"),
    timezone: str = Query(DEFAULT_TIMEZONE, description=f"Timezone for startTs and output timestamps. Default is {DEFAULT_TIMEZONE}")) -> PricesModelShort:
    """
    Get price prediction - short output format with unix timestamp array and price array
    """
    res = await prices_handler.prices(hours, fixed_price, tax_percent, start_ts, region.to_region(), unit, evaluation, hourly, timezone, format=OutputFormat.SHORT)
    assert isinstance(res, PricesModelShort)
    return res


@app.get("/eval_plot", response_class=Response, response_model=None, responses={
        200: {
            "content": {"image/png": {}},
            "description": "PNG plot"
        },
        400: {
            "content": {"application/json": {}}
        }
    })
async def generate_evaluation_plot(
    start_ts: datetime | None = Query(None, description="Plot range start, at most ~1 year in the past. Default today 00:00Z", alias="startTs"),
    end_ts: datetime | None = Query(None, description="Plot range end, Default startTs + 1 week. At most 31 days after startTs and 10 days from now", alias="endTs"),
    region: PriceRegionName = Query(PriceRegionName.DE, description="Region/bidding zone", alias="country"),
    transparent: bool = Query(False, description="Render with transparent background"),
    width: int = Query(2048, description="image width in pixels", ge=300, le=10000),
    height: int = Query(1024, description="image height in pixels", ge=300, le=10000)):
    """
    Trains a model just for you, training with 120 days before the given time range and providing a forecast for the given range.
    - If there is no cached weather or price data for the given time range, this request can take a while. Be patient.
    - This request is rather CPU intensive. Do not batch-call or you will be banned.
    """
    now = datetime.now(timezone.utc)
    start_ts = start_ts or now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = end_ts or start_ts + timedelta(days=7)
    start_ts = start_ts.astimezone(timezone.utc)
    end_ts = end_ts.astimezone(timezone.utc)
    if (end_ts - start_ts).total_seconds() > 31 * 24 * 60 * 60:
        raise HTTPException(status_code=400, detail="At most 4 weeks can be plotted")
    
    if start_ts < now - timedelta(days=365):
        raise HTTPException(status_code=400, detail="Requested range too far in the past")
    
    if end_ts > now + timedelta(days=10):
        raise HTTPException(status_code=400, detail="Requested range too far in the future")
    
    if end_ts <= start_ts:
        raise HTTPException(status_code=400, detail="endTs must be after startTs")

    # reuse the same data stores for a unified cache
    orig_predictor = prices_handler.get_predictor(region.to_region())
    
    predictor = pp.PricePredictor(region.to_region())
    predictor.weatherstore = orig_predictor.weatherstore
    predictor.pricestore = orig_predictor.pricestore
    predictor.auxstore = orig_predictor.auxstore

    learn_start = start_ts - timedelta(days=TRAINING_DAYS)
    learn_end = start_ts
    await predictor.train(learn_start, learn_end)

    predicted = await predictor.predict(start_ts, end_ts, fill_known=False)
    predicted = predicted.rename(columns={"price": "predicted"})

    # make sure we don't refetch future price data with each request - only force-fetch until today.
    # Later price might be returned if the main price API already fetched it.
    latest_price_to_fetch = min(end_ts, now)
    await predictor.pricestore.fetch_missing_data(start_ts, latest_price_to_fetch)
    actual = predictor.pricestore.get_known_data(start_ts, end_ts)
    actual = actual.rename(columns={"price": "actual"})

    merged = pd.concat([predicted, actual])

    img_data = BytesIO()
    plot = merged.plot.line(grid=True)
    assert isinstance(plot.figure, Figure)
    plot.margins(0)
    plot.figure.set_size_inches(width / 100, height / 100)
    plot.figure.savefig(img_data, format="png", transparent=transparent, dpi=100, bbox_inches="tight")
    plt.close(plot.figure)

    img_data.seek(0)
    return Response(content=img_data.read(), media_type="image/png")






