import asyncio
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
from typing import Dict, List, Self
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from predictor.model.priceregion import PriceRegion, PriceRegionName
import predictor.model.pricepredictor as pp


import warnings

# Used internally inside the standard library by asyncio.to_thread.. annoying
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="asyncio"
)



app = FastAPI(title="EPEX day-ahead prediction API", description="""
API can be used free of charge on a fair use premise.
There are no guarantees on availability or correctnes of the data.
This is an open source project, feel free to host it yourself. [Source code and docs](https://github.com/b3nn0/EpexPredictor)

### Attribution
Electricity prices provided under CC-BY-4.0 by [energy-charts.info](https://api.energy-charts.info/) and [ENTSO-E](https://www.entsoe.eu/)

[Weather data by Open-Meteo.com](https://open-meteo.com/)
""")


##### Logging Setup

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now(timezone.utc)

    response = await call_next(request)

    process_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000.0

    client = request.client.host if request.client else "-"
    user_agent = request.headers.get("user-agent", "-")

    log.info(
        '%s "%s %s" %d %.2fms "%s"',
        client,
        request.method,
        request.url,
        response.status_code,
        process_time,
        user_agent,
    )

    return response



logging.getLogger("uvicorn.error").handlers.clear()
logging.getLogger("uvicorn.error").handlers.extend(logging.getLogger().handlers)
logging.getLogger("uvicorn.access").disabled = True # we handle this ourself in middleware above



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

    last_retrain : datetime = datetime(1980, 1, 1, tzinfo=timezone.utc)
    last_weather_update : datetime = datetime(1980, 1, 1, tzinfo=timezone.utc)
    last_known_price : datetime


    cachedprices : pd.DataFrame
    cachedeval : pd.DataFrame

    update_lock: asyncio.Lock

    init_lock: asyncio.Lock
    is_loaded: bool = False

    def __init__(self, region: PriceRegion):
        self.init_lock = asyncio.Lock() # ensures only one aio worker will load persistent data on first access
        self.update_lock = asyncio.Lock() # ensures only one aio worker will trigger model update

        self.cachedprices = pd.DataFrame()
        self.cachedeval = pd.DataFrame()
        self.last_known_price = datetime(1970, 1, 1, tzinfo=timezone.utc)

        self.predictor = pp.PricePredictor(region, storage_dir=EPEXPREDICTOR_DATADIR)

    async def ensure_loaded(self) -> Self:
        async with self.init_lock:
            if self.is_loaded:
                return self
            log.info(f"Loading persistent data for {self.predictor.region.bidding_zone_entsoe}")
            await self.predictor.load_from_persistence()
            self.is_loaded = True
        return self


    def _normalize_start_ts(self, start_ts: datetime | None, tz: ZoneInfo) -> datetime:
        """Normalize start_ts to the target timezone."""
        if start_ts is None:
            return datetime.now(tz=tz)
        if start_ts.tzinfo is None:
            return start_ts.replace(tzinfo=tz)
        return start_ts.astimezone(tz)


    async def prices(self, hours: int = -1, surcharge: float = 0.0, tax_percent: float = 0.0, start_ts: datetime | None = None,
                    unit: PriceUnit = PriceUnit.CT_PER_KWH, evaluation: bool = False, hourly: bool = False,
                    timezone: str = DEFAULT_TIMEZONE, format: OutputFormat = OutputFormat.LONG) -> PricesModel | PricesModelShort:

        await self.update_in_background()

        try:
            tz = ZoneInfo(timezone)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid timezone {timezone}")
        start_ts = self._normalize_start_ts(start_ts, tz)
        end_ts = start_ts + timedelta(hours=hours) if hours >= 0 else datetime(2999, 1, 1, tzinfo=tz)

        prediction = self.cachedeval if evaluation else self.cachedprices
        if hourly:
            prediction = prediction.resample("1h").mean()
        
        prediction = prediction.loc[start_ts:end_ts]

        prices = []

        for dt, price in zip(prediction.index, prediction["price"]): # seems to be much faster than .iterrows()..
            assert isinstance(dt, pd.Timestamp)
            total = (price + surcharge) * (1 + tax_percent / 100.0)
            total = unit.convert(total)
            prices.append(PriceModel(starts_at=dt.to_pydatetime().astimezone(tz), total=round(total, 4)))

        if format == OutputFormat.SHORT:
            return self.format_short(prices)
        return PricesModel(prices=prices, known_until=self.last_known_price.astimezone(tz))

        
    def format_short(self, prices: List[PriceModel]) -> PricesModelShort:
        return PricesModelShort(
            s=[round(p.starts_at.timestamp()) for p in prices],
            t=[round(p.total, 4) for p in prices]
        )


    async def update_in_background(self):
        if self.update_lock.locked() and len(self.cachedprices) > 0:
            return # don't queue up multiple updates if we already have a filled cache

        update_future = self.update_data_if_needed()
        if len(self.cachedprices) == 0: # first call, no prices yet -> wait until first update is done
            await update_future
        else:
            asyncio.create_task(update_future)


    async def update_data_if_needed(self):
        async with self.update_lock:
            currts = datetime.now(timezone.utc)
            train_start = currts - timedelta(days=TRAINING_DAYS)
            train_end = datetime.now(timezone.utc) + timedelta(days=7) # will ensure all weather data is fetched immediately, not partially for training and then partially for prediction

            weather_age = (currts - self.last_weather_update).total_seconds()


            retrain = False

            # since we cache the prediction result, the price store is never queried and never updates until next retrain/weather update..
            # Ensure we retrain (and re-fetch horizon) more often if needed
            if self.predictor.pricestore.needs_horizon_revalidation() or self.predictor.gasstore.needs_horizon_revalidation():
                retrain = True # will be fetched automatically during training

            if weather_age > 60 * 60 * 3:  # update forecasted input data every 3 hours
                start = datetime.now(timezone.utc) - timedelta(days=1)
                end = datetime.now(timezone.utc) + timedelta(days=8)
                await self.predictor.refresh_forecasts(start, end)
                self.last_weather_update = currts
                retrain = True


            if self.predictor.last_data_update() > self.last_retrain or retrain:
                log.info(f"{self.predictor.region.bidding_zone_entsoe}: data has been updated - triggering model retrain")
                self.last_retrain = datetime.now(timezone.utc)

                await self.predictor.train(train_start, train_end)
                newprices, neweval = await self.predictor.predict(train_start, train_end), await self.predictor.predict(train_start, train_end, fill_known=False)
                self.cachedprices = newprices
                self.cachedeval = neweval
                lastknown = self.predictor.pricestore.get_last_known()
                if lastknown is not None:
                    self.last_known_price = lastknown

                self.predictor.cleanup()

 


class Prices:
    region_prices: Dict[PriceRegionName, RegionPriceManager]

    def __init__(self):
        self.region_prices = {}

    async def prices(self, hours: int = -1, surcharge: float = 0.0, tax_percent: float = 0.0, start_ts: datetime | None = None,
                    region: PriceRegionName = PriceRegionName.DE, unit: PriceUnit = PriceUnit.CT_PER_KWH, evaluation: bool = False, hourly: bool = False,
                    timezone: str = DEFAULT_TIMEZONE, format: OutputFormat = OutputFormat.LONG):
        if region not in self.region_prices:
            self.region_prices[region] = RegionPriceManager(region.to_region())
        
        await self.region_prices[region].ensure_loaded()
        return await self.region_prices[region].prices(hours, surcharge, tax_percent, start_ts, unit, evaluation, hourly, timezone, format)
    
    async def get_price_manager(self, region: PriceRegionName):
        if region not in self.region_prices:
            self.region_prices[region] = RegionPriceManager(region.to_region())
        
        await self.region_prices[region].ensure_loaded()
        return self.region_prices[region]


prices_handler = Prices()


@app.get("/prices")
async def get_prices(
    hours: int = Query(-1, description="How many hours to predict"),
    surcharge: float = Query(0.0, description="Add this fixed amount to all prices (ct/kWh)"),
    tax_percent: float = Query(0.0, description="Tax % to add to the final price", alias="taxPercent"),
    start_ts: datetime | None = Query(None, description="Start output from this time. At most ~90 days in the past", alias="startTs"),
    region: PriceRegionName = Query(PriceRegionName.DE, description="Region/bidding zone"),
    evaluation: bool = Query(False, description="Switches to evaluation mode. All values will be generated by the model, instead of only future values. Useful to evaluate model performance."),
    unit: PriceUnit = Query(PriceUnit.CT_PER_KWH, description="Unit of output"),
    hourly: bool = Query(False, description="Output hourly average prices (if your energy provider uses hourly prices)"),
    timezone: str = Query(DEFAULT_TIMEZONE, description=f"Timezone for startTs and output timestamps. Default is {DEFAULT_TIMEZONE}"),

    # Legacy parameters, only here for backwards compatibility
    country: PriceRegionName = Query(None, description="", include_in_schema=False),
    fixed_price: float = Query(None, description="Add this fixed amount to all prices (ct/kWh)", alias="fixedPrice", include_in_schema=False),
    ) -> PricesModel:
    """
    Get price prediction - verbose output format with objects containing full ISO timestamp and price
    """
    if country:
        region = country
    if fixed_price is not None:
        surcharge = fixed_price

    res = await prices_handler.prices(hours, surcharge, tax_percent, start_ts, region, unit, evaluation, hourly, timezone, format=OutputFormat.LONG)
    assert isinstance(res, PricesModel)
    return res


@app.get("/prices_short")
async def get_prices_short(
    hours: int = Query(-1, description="How many hours to predict"),
    surcharge: float = Query(0.0, description="Add this fixed amount to all prices (ct/kWh)"),
    tax_percent: float = Query(0.0, description="Tax % to add to the final price", alias="taxPercent"),
    start_ts: datetime | None = Query(None, description="Start output from this time. At most ~90 days in the past", alias="startTs"),
    region: PriceRegionName = Query(PriceRegionName.DE, description="Region/bidding zone", alias="country"),
    evaluation: bool = Query(False, description="Switches to evaluation mode. All values will be generated by the model, instead of only future values. Useful to evaluate model performance."),
    unit: PriceUnit = Query(PriceUnit.CT_PER_KWH, description="Unit of output"),
    hourly: bool = Query(False, description="Output hourly average prices (if your energy provider uses hourly prices)"),
    timezone: str = Query(DEFAULT_TIMEZONE, description=f"Timezone for startTs and output timestamps. Default is {DEFAULT_TIMEZONE}"),
    
    # Legacy parameters, only here for backwards compatibility
    country: PriceRegionName = Query(None, description="", include_in_schema=False),
    fixed_price: float = Query(None, description="Add this fixed amount to all prices (ct/kWh)", alias="fixedPrice", include_in_schema=False),
    ) -> PricesModelShort:
    """
    Get price prediction - short output format with unix timestamp array and price array
    """
    if country:
        region = country
    if fixed_price is not None:
        surcharge = fixed_price

    res = await prices_handler.prices(hours, surcharge, tax_percent, start_ts, region, unit, evaluation, hourly, timezone, format=OutputFormat.SHORT)
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
    region: PriceRegionName = Query(PriceRegionName.DE, description="Region/bidding zone"),
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
    pricemanager = await prices_handler.get_price_manager(region)
    await pricemanager.update_data_if_needed()
    orig_predictor = pricemanager.predictor
    
    predictor = pp.PricePredictor(region.to_region())
    predictor.use_datastores_from(orig_predictor)

    learn_start = start_ts - timedelta(days=TRAINING_DAYS)
    learn_end = start_ts
    await predictor.train(learn_start, learn_end)

    predicted = await predictor.predict(start_ts, end_ts, fill_known=False)
    predicted = predicted.rename(columns={"price": "predicted"})

    actual = await predictor.pricestore.get_data(start_ts, end_ts)
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

    response = Response(content=img_data.read(), media_type="image/png")
    response.headers.update({
        "Cache-Control": "max-age=60"
    })

    return response






