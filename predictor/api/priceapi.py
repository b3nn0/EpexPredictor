import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
import logging
import os
import random
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
from predictor.api.evalplotcache import EvalPlotCache


import warnings

# Used internally inside the standard library by asyncio.to_thread.. annoying
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="asyncio"
)



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and train the shared base models before serving, then keep everything
    # fresh in the background - requests only read caches.
    await base_models.ensure_loaded()
    await base_models._do_update()
    refresh_task = asyncio.create_task(base_models.refresh_loop())
    try:
        yield
    finally:
        refresh_task.cancel()


app = FastAPI(title="EPEX day-ahead prediction API", lifespan=lifespan, description="""
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
TRAINING_DAYS = 180
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


    
# Regions that only have an ENTSO-E bidding zone need the ENTSO-E API key to fetch prices.
# Without it, we can't train or predict them, so they are skipped gracefully.
_HAS_ENTSOE_API_KEY = os.getenv("EPEXPREDICTOR_ENTSOE_API_KEY", None) is not None and len(os.getenv("EPEXPREDICTOR_ENTSOE_API_KEY", "")) > 0


def is_region_available(name: PriceRegionName) -> bool:
    region = name.to_region()
    if region.bidding_zone_energycharts is not None:
        return True
    return _HAS_ENTSOE_API_KEY


AVAILABLE_REGIONS = [name for name in PriceRegionName if is_region_available(name)]


class BaseModelCoordinator:
    """
    Owns the single shared set of stage-1 (base) models for all available regions,
    plus the shared data-fetch + base-retrain cycle.

    Stage-2 models (one per requested region) are owned by their RegionPriceManager
    and registered here, so a single background refresh loop (started in the app
    lifespan) can run the base cycle and then update every live stage-2 model.
    Requests never trigger refreshes - they read the caches, and only the very
    first request for a region blocks until its initial update finished.

    The loop sleeps until the earliest upstream revalidation deadline reported by
    the data stores (see DataStore.get_next_horizon_revalidation_time), so it wakes
    exactly when new data is expected to be available. Checks before a deadline are
    cheap no-ops because the stores gate their own horizon revalidation.
    """
    base: Dict[PriceRegionName, pp.PricePredictor]
    managers: Dict[PriceRegionName, "RegionPriceManager"]
    update_lock: asyncio.Lock
    base_ready: bool = False
    _last_retrain: datetime
    _last_weather_refresh: datetime

    # Safety net: never sleep longer than this waiting for a store-reported deadline
    MAX_REFRESH_INTERVAL: timedelta = timedelta(hours=1)
    # cadence for refreshing forecasted inputs (weather + entsoe load)
    WEATHER_REFRESH_INTERVAL: timedelta = timedelta(hours=3)

    def __init__(self):
        self.update_lock = asyncio.Lock()
        self.managers = {}
        self.base = {}
        for name in AVAILABLE_REGIONS:
            region = name.to_region()
            p = pp.PricePredictor(region, storage_dir=EPEXPREDICTOR_DATADIR)
            self.base[name] = p
        # Stage-2 models are wired to ALL available base models (incl. their own)
        self.cross_predictors: list[pp.PricePredictor] = [self.base[n] for n in AVAILABLE_REGIONS]
        self._last_retrain = datetime(1970, 1, 1, tzinfo=timezone.utc)
        self._last_weather_refresh = datetime(1970, 1, 1, tzinfo=timezone.utc)

    async def ensure_loaded(self) -> Self:
        async with self.update_lock:
            if self.base_ready:
                return self
            log.info(f"Loading base models for {len(self.base)} regions")
            await asyncio.gather(*(p.load_from_persistence() for p in self.base.values()))
            # Seed the forecast-refresh cadence from the persisted stores, so a restart
            # shortly after a refresh doesn't immediately re-fetch all weather data.
            self._last_weather_refresh = max(
                (p.weatherstore.last_updated for p in self.base.values()),
                default=self._last_weather_refresh,
            )
            self.base_ready = True
        return self

    def get_base(self, name: PriceRegionName) -> pp.PricePredictor | None:
        return self.base.get(name)

    def get_cross_predictors(self) -> list[pp.PricePredictor]:
        return self.cross_predictors

    def register_manager(self, manager: "RegionPriceManager"):
        """Register a live stage-2 manager so the refresh loop keeps it up to date."""
        self.managers[manager.name] = manager

    async def ensure_base_trained(self):
        """
        Run a base update cycle if any base model is not trained yet.
        Only relevant on cold start, before the refresh loop's first cycle finished -
        stage-2 training needs the base models' forecasts as cross-features.
        """
        if all(p.is_trained() for p in self.base.values()):
            return
        async with self.update_lock:
            if any(not p.is_trained() for p in self.base.values()):
                await self._do_update()

    async def refresh_loop(self):
        """
        Background loop driving the shared base cycle + all live stage-2 managers.
        Sleeps until the earliest revalidation deadline reported by the shared data
        stores, so it wakes right when upstream data is expected to change.
        """
        while True:
            try:
                async with self.update_lock:
                    await self._do_update()
                for manager in list(self.managers.values()):
                    await manager.update_data_if_needed()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Refresh cycle failed - retrying at next deadline")

            await asyncio.sleep(self._seconds_until_next_refresh())

    def _seconds_until_next_refresh(self) -> float:
        now = datetime.now(timezone.utc)
        deadlines = [now + self.MAX_REFRESH_INTERVAL]

        # earliest store-reported revalidation deadline across all shared stores
        for p in self.base.values():
            for store in (p.pricestore, p.gasstore, p.weatherstore, p.entsoestore, p.auxstore):
                ts = store.source_horizon_revalitation_ts
                if ts is not None:
                    deadlines.append(ts)

        # forecast (weather/entsoe) refresh cadence
        deadlines.append(self._last_weather_refresh + self.WEATHER_REFRESH_INTERVAL)

        return max(1.0, min((min(deadlines) - now).total_seconds(), self.MAX_REFRESH_INTERVAL.total_seconds()))

    async def _do_update(self):
        currts = datetime.now(timezone.utc)
        train_start = currts - timedelta(days=TRAINING_DAYS)
        train_end = currts + timedelta(days=7)

        # Remember freshness so we can skip the retrain when nothing changed.
        prev = {name: p.last_data_update() for name, p in self.base.items()}

        # 1) fetch fresh data for all base regions (shared stores; cheap no-op when nothing is due)
        await asyncio.gather(*(
            p.pricestore.get_data(currts, train_end) for p in self.base.values()
        ))
        await asyncio.gather(*(
            p.gasstore.get_data(currts, train_end) for p in self.base.values()
        ))

        # 2) refresh forecasted inputs (weather + entsoe load) on a 3h cadence.
        # Sequential with random spacing: WeatherStore already serializes all OpenMeteo
        # calls globally, so gather would only burst-fire them back-to-back - which
        # OpenMeteo dislikes.
        if (currts - self._last_weather_refresh).total_seconds() > self.WEATHER_REFRESH_INTERVAL.total_seconds():
            for p in self.base.values():
                await p.refresh_forecasts(currts - timedelta(days=1), currts + timedelta(days=8))
                await asyncio.sleep(random.uniform(5, 20))
            self._last_weather_refresh = datetime.now(timezone.utc)

        # 3) retrain the base models that aren't trained yet, or whose data changed
        changed = [name for name, p in self.base.items() if not p.is_trained() or p.last_data_update() > prev[name]]
        if not changed:
            return
        await asyncio.gather(*(
            self.base[name].train(train_start, train_end) for name in changed
        ))
        self._last_retrain = datetime.now(timezone.utc)
        log.info(f"Base models retrained for {len(changed)}/{len(self.base)} regions")

    async def train_eval_plot_data(self, name: PriceRegionName, region: PriceRegion, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
        """
        Train the full 2-stage model stack on the historical window [start_ts - TRAINING_DAYS, start_ts]
        and return the merged predicted/actual DataFrame for [start_ts, end_ts] that the eval plot is rendered from.

        Uses fresh stage-1 models (booster only) that share the live base models' data stores, so a
        historical eval plot never clobbers the live base models' training window, and no separate
        data loading/fetching is needed - train/predict fetch whatever is missing from the shared
        stores implicitly via prepare_dataframe.
        """
        learn_start = start_ts - timedelta(days=TRAINING_DAYS)
        learn_end = start_ts

        # Make sure the shared stores are loaded from persistence (no-op if already done)
        await self.ensure_loaded()

        # 1) fresh stage-1 models, one per available region, sharing the live base models' stores
        bases: list[pp.PricePredictor] = []
        for base_name in AVAILABLE_REGIONS:
            p = pp.PricePredictor(base_name.to_region(), storage_dir=EPEXPREDICTOR_DATADIR)
            p.use_datastores_from(self.base[base_name])
            bases.append(p)

        # 2) stage-2 model for the requested region, wired to the fresh stage-1 models
        eval_model = pp.PricePredictor(region, storage_dir=EPEXPREDICTOR_DATADIR, cross_predictors=bases)
        eval_model.use_datastores_from(self.base[name])

        # 3) train stage 1, then stage 2 (stage 2 needs the fresh stage-1 forecasts as cross-features)
        await asyncio.gather(*(p.train(learn_start, learn_end) for p in bases))
        await eval_model.train(learn_start, learn_end)

        # 4) merge prediction and actuals for the plot range
        predicted = await eval_model.predict(start_ts, end_ts, fill_known=False)
        predicted = predicted.rename(columns={"price": "predicted"})

        actual = await eval_model.pricestore.get_data(start_ts, end_ts)
        actual = actual.rename(columns={"price": "actual"})

        return pd.concat([predicted, actual])


base_models = BaseModelCoordinator()
eval_plot_cache = EvalPlotCache(trainer=base_models.train_eval_plot_data)


class RegionPriceManager:
    """
    Owns the stage-2 model for a single region (and its cached prices).

    The stage-2 model is wired once at construction to the shared base (stage-1)
    models via cross_predictors, and shares its data stores with its own base model
    (so the shared base-fetch cycle keeps this region's data fresh for free).
    It is retrained whenever the base models were retrained (fresh cross-features)
    or its own data changed.
    """
    region: PriceRegion
    name: PriceRegionName
    predictor : pp.PricePredictor

    last_retrain : datetime = datetime(1980, 1, 1, tzinfo=timezone.utc)
    last_known_price : datetime


    cachedprices : pd.DataFrame
    cachedeval : pd.DataFrame

    update_lock: asyncio.Lock

    init_lock: asyncio.Lock
    is_loaded: bool = False

    def __init__(self, name: PriceRegionName, region: PriceRegion):
        self.init_lock = asyncio.Lock() # ensures only one aio worker will load persistent data on first access
        self.update_lock = asyncio.Lock() # ensures only one aio worker will trigger model update

        self.name = name
        self.region = region
        self.cachedprices = pd.DataFrame()
        self.cachedeval = pd.DataFrame()
        self.last_known_price = datetime(1970, 1, 1, tzinfo=timezone.utc)

        base = base_models.get_base(name)
        assert base is not None
        # Stage-2 model: same region, wired to all base models, sharing stores with its own base
        self.predictor = pp.PricePredictor(region, storage_dir=EPEXPREDICTOR_DATADIR, cross_predictors=base_models.get_cross_predictors())
        self.predictor.use_datastores_from(base)

    async def ensure_loaded(self) -> Self:
        async with self.init_lock:
            if self.is_loaded:
                return self
            log.info(f"{self.predictor.region.bidding_zone_entsoe}: Loading persistent data")
            await base_models.ensure_loaded()
            self.is_loaded = True
        return self


    def _normalize_start_ts(self, start_ts: datetime | None, tz: ZoneInfo, hourly: bool) -> datetime:
        """Normalize start_ts to the target timezone."""
        if start_ts is None:
            now = datetime.now(tz=tz)
            if hourly:
                return now.replace(second=0, microsecond=0, minute=0, hour=now.hour)
            else:
                return now.replace(second=0, microsecond=0, minute=(now.minute // 15 * 15), hour=now.hour)
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
        start_ts = self._normalize_start_ts(start_ts, tz, hourly)
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
        # Refreshes run in the background refresh loop; requests only need to block
        # on the very first update for this region (cold cache).
        if len(self.cachedprices) == 0:
            await self.update_data_if_needed()


    async def update_data_if_needed(self):
        # Stage-2 training needs trained base models (cross-features). On cold start the
        # refresh loop may not have completed its first cycle yet, so ensure one here.
        # Must happen before acquiring our update_lock: the refresh loop holds the base
        # lock while calling into managers, so acquiring base->manager in the opposite
        # order here would deadlock.
        await base_models.ensure_base_trained()

        async with self.update_lock:
            # This region's data stores are shared with its base model, so the shared
            # base cycle keeps them fresh for free.

            currts = datetime.now(timezone.utc)
            train_start = currts - timedelta(days=TRAINING_DAYS)
            train_end = currts + timedelta(days=7) # will ensure all weather data is fetched immediately, not partially for training and then partially for prediction

            # Retrain the stage-2 model when our own data changed, or when any base model
            # was retrained (which means at least one of our cross-features changed).
            if self.predictor.last_data_update() > self.last_retrain or base_models._last_retrain > self.last_retrain:
                log.info(f"{self.predictor.region.bidding_zone_entsoe}: retraining stage-2 model")
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
        manager = await self.get_price_manager(region)
        return await manager.prices(hours, surcharge, tax_percent, start_ts, unit, evaluation, hourly, timezone, format)
    
    async def get_price_manager(self, region: PriceRegionName):
        if not is_region_available(region):
            raise HTTPException(status_code=404, detail=f"Region {region} is not available (requires an ENTSO-E API key)")
        if region not in self.region_prices:
            self.region_prices[region] = RegionPriceManager(region, region.to_region())
            base_models.register_manager(self.region_prices[region])
        
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
    region: PriceRegionName = Query(PriceRegionName.DE, description="Region/bidding zone"),
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
    if not is_region_available(region):
        raise HTTPException(status_code=404, detail=f"Region {region} is not available (requires an ENTSO-E API key)")

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

    # Training the full 2-stage stack per historical window is expensive, so the trained
    # results are cached for 30 minutes per (region, range); stale entries are served
    # while a background refresh retrains. Rendering to PNG is cheap and per-request.
    merged = await eval_plot_cache.get(region, start_ts, end_ts)

    img_data = BytesIO()
    plot = merged.plot.line(grid=True)
    assert isinstance(plot.figure, Figure)
    plot.margins(0)
    plot.figure.set_size_inches(width / 100, height / 100)
    plot.figure.savefig(img_data, format="png", transparent=transparent, dpi=100, bbox_inches="tight")
    plt.close(plot.figure)

    response = Response(content=img_data.getvalue(), media_type="image/png")
    response.headers.update({
        "Cache-Control": "max-age=60"
    })

    return response






