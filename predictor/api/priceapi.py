import logging
import sys
import os
import asyncio
from typing import Dict
from enum import Enum
import bisect

from pydantic import BaseModel
import pytz
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
import datetime

from predictor.model.pricepredictor import Country

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

import predictor.model.pricepredictor as pp

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

    

class CountryPrices:
    predictor : pp.PricePredictor

    last_weather_update : datetime.datetime = datetime.datetime(1980, 1, 1)
    last_price_update : datetime.datetime = datetime.datetime(1980, 1, 1)

    last_known_price : tuple[datetime.datetime, float] = (datetime.datetime.now(), 0)

    cachedprices : Dict[datetime.datetime, float] = {}
    cachedeval : Dict[datetime.datetime, float] = {}

    updateTask : asyncio.Task | None = None

    def __init__(self, country : Country):
        self.predictor =  pp.PricePredictor(country, testdata=USE_PERSISTENT_TESTDATA)

    async def prices(self, hours : int = -1, fixedPrice : float = 0.0, taxPercent : float = 0.0, startTs : datetime.datetime|None = None,
                    unit : PriceUnit = PriceUnit.CT_PER_KWH, evaluation : bool = False, hourly : bool = False,
                    timezone : str = "Europe/Berlin"):

        await self.update_in_background()

        tz = pytz.timezone(timezone)

        if startTs is None:
            startTs = datetime.datetime.now(tz=tz)
        else:
            if startTs.tzinfo is None:
                startTs = startTs.astimezone(tz)
        
        endTs = datetime.datetime(2999, 1, 1, tzinfo=tz)
        if hours >= 0:
            endTs = startTs + datetime.timedelta(hours=hours)

        prediction = self.cachedprices if evaluation is False else self.cachedeval

        # Calculate hourly averages if hourly mode is enabled
        if hourly:
            hourly_averages = {}
            for dt in sorted(prediction.keys()):
                dt_local = dt.astimezone(tz)
                hour_key = dt_local.replace(minute=0, second=0, microsecond=0)
                
                if hour_key not in hourly_averages:
                    hourly_averages[hour_key] = []
                hourly_averages[hour_key].append(prediction[dt])
            
            # Replace prediction with hourly averages, skipping empty lists to avoid division by zero
            prediction = {hour_dt: sum(prices) / len(prices) for hour_dt, prices in hourly_averages.items() if len(prices) > 0}

        prices : list[PriceModel] = []
        
        dts = list(sorted(prediction.keys()))
        startindex = max(0, bisect.bisect_right(dts, startTs) - 1)
        endindex = min(len(dts)-1, bisect.bisect_right(dts, endTs))

        for dt in dts[startindex:endindex]:
            price = prediction[dt]
            total = (price + fixedPrice) * (1 + taxPercent / 100.0)
            total = unit.convert(total)
            dt = dt.astimezone(tz)

            prices.append(PriceModel(startsAt=dt, total=round(total, 4)))

        return PricesModel(
            prices = prices,
            knownUntil = self.last_known_price[0].astimezone(tz)
        )


    async def update_in_background(self):
        if self.updateTask is None:
            self.updateTask = asyncio.create_task(self.update_data_if_needed())
        
        if len(self.cachedprices) == 0:
            await self.updateTask # sync refresh on first call


    async def update_data_if_needed(self):
        try:
            currts = datetime.datetime.now()

            price_age = currts - self.last_price_update
            weather_age = currts - self.last_weather_update

            self.is_currently_updating = True

            # Update prices every 12 hours. If it's after 13:00 local, and we don't have prices for the next day yet, update every 5 minutes
            latest_price = self.predictor.get_last_known_price()
            price_update_frequency = 12 * 60 * 60
            if latest_price is None or (latest_price[0] - datetime.datetime.now(pytz.UTC)).total_seconds() <= 60 * 60 * 11:
                price_update_frequency = 5 * 60

            retrain = False
            if price_age.total_seconds() > price_update_frequency:
                await self.predictor.refresh_prices()
                self.last_price_update = currts
                retrain = True

            if weather_age.total_seconds() > 60 * 60 * 6: # update weather every 6 hours
                await self.predictor.refresh_forecasts()
                self.last_weather_update = currts
                retrain = True

            if retrain:
                await self.predictor.train()
                newprices, neweval = await self.predictor.predict(), await self.predictor.predict(estimateAll=True)
                self.cachedprices = newprices
                self.cachedeval = neweval
                lastknown = self.predictor.get_last_known_price()
                if lastknown is not None:
                    self.last_known_price = lastknown

        finally:
            self.updateTask = None


class PriceModel(BaseModel):
    startsAt : datetime.datetime
    total: float

class PricesModel(BaseModel):
    prices : list[PriceModel]
    knownUntil: datetime.datetime

class Prices:
    countryPrices : Dict[Country, CountryPrices] = {
        Country.DE: CountryPrices(Country.DE),
        Country.AT: CountryPrices(Country.AT),
        Country.BE: CountryPrices(Country.BE)
    }

    def __init__(self):
        pass

    async def prices(self, hours : int = -1, fixedPrice : float = 0.0, taxPercent : float = 0.0, startTs : datetime.datetime|None = None,
                    country : Country = Country.DE, unit : PriceUnit = PriceUnit.CT_PER_KWH, evaluation : bool = False, hourly : bool = False,
                    timezone : str = "Europe/Berlin"):
        return await self.countryPrices[country].prices(hours,fixedPrice, taxPercent, startTs, unit, evaluation, hourly, timezone)


pricesHandler = Prices()
@app.get("/prices", response_model=PricesModel)
async def get_prices(
    hours : int = Query(-1, description="How many hours to predict"),
    fixedPrice : float = Query(0.0, description="Add this fixed amount to all prices (ct/kWh)"),
    taxPercent : float = Query(0.0, description="Tax % to add to the final price"),
    startTs : datetime.datetime | None = Query(None, description="Start output from this time. At most ~90 days in the past"),
    country : Country = Query(Country.DE, description="Country Code"),
    evaluation : bool = Query(False, description="Switches to evaluation mode. All values will be generated by the model, instead of only future values. Useful to evaluate model performance."),
    unit : PriceUnit = Query(PriceUnit.CT_PER_KWH, description="Unit of output"),
    hourly : bool = Query(False, description="Output hourly average prices (if your energy provider uses hourly prices)"),
    timezone : str = Query("Europe/Berlin", description="Timezone for startTs and output timestamps. Default is Europe/Berlin")):
    """
    Get price prediction
    """
    return await pricesHandler.prices(hours, fixedPrice, taxPercent, startTs, country, unit, evaluation, hourly, timezone)


    
