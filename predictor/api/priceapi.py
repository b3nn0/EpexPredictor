import logging
import sys
import os
import asyncio
from typing import Dict

import pytz
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
import datetime



app = FastAPI(title="EPEX day-ahead prediction API")


rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

# StreamHandler for console
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
stream_handler.setFormatter(log_formatter)
rootLogger.addHandler(stream_handler)

# TODO: isn't there a nicer way...?
logging.getLogger("uvicorn.access").handlers.clear()
logging.getLogger("uvicorn.access").addHandler(stream_handler)

log = logging.getLogger(__name__)



@app.get("/")
def api_docs():
    return RedirectResponse("/docs")


USE_PERSISTENT_TESTDATA = os.getenv("USE_PERSISTENT_TEST_DATA", "false").lower() in ("yes", "true", "t", "1")


import predictor.model.pricepredictor as pp


class Prices:
    predictor : pp.PricePredictor = pp.PricePredictor(testdata=USE_PERSISTENT_TESTDATA)
    
    last_weather_update : datetime.datetime = datetime.datetime(1980, 1, 1)
    last_price_update : datetime.datetime = datetime.datetime(1980, 1, 1)

    cachedprices : Dict[datetime.datetime, float] = {}
    cachedeval : Dict[datetime.datetime, float] = {}

    updateTask : asyncio.Task | None = None


    def __init__(self):
        pass


    async def prices(self, hours : int = -1, fixedPrice : float = 0.0, taxPercent : float = 0.0, startTs : datetime.datetime|None = None, evaluation : bool = False):
        await self.update_in_background()

        tzgerman = pytz.timezone("Europe/Berlin")

        if startTs is None:
            startTs = datetime.datetime.now(tz=tzgerman)
            startTs = startTs.replace(minute=0, second=0, microsecond=0)
        else:
            if startTs.tzinfo is None:
                startTs = startTs.astimezone(tzgerman)
        
        endTs = datetime.datetime(2999, 1, 1, tzinfo=tzgerman)
        if hours >= 0:
            endTs = startTs + datetime.timedelta(hours=hours)

        prices = self.cachedprices if evaluation is False else self.cachedeval

        result = []
        for dt in sorted(prices.keys()):
            if dt < startTs:
                continue
            if dt > endTs:
                continue
            formatted = dt.astimezone(tzgerman).isoformat()
            price = prices[dt] / 10.0 # to ct/kWh
            total = (price + fixedPrice) * (1 + taxPercent / 100.0)
            result.append(
                {
                    "startsAt": formatted,
                    "total": round(total, 2)
                }
            )

        return { "prices": result }

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
            if latest_price is None or (latest_price[0] - datetime.datetime.now(pytz.UTC)).total_seconds() <= 60 * 60 * 10:
                price_update_frequency = 5 * 60

            retrain = False
            if price_age.total_seconds() > price_update_frequency:
                await self.predictor.refresh_prices()
                self.last_price_update = currts
                retrain = True

            if weather_age.total_seconds() > 60 * 60 * 8: # update weather every 8 hours
                await self.predictor.refresh_forecasts()
                self.last_weather_update = currts
                retrain = True

            if retrain:
                await self.predictor.train()
                newprices, neweval = await self.predictor.predict(), await self.predictor.predict(estimateAll=True)
                self.cachedprices = newprices
                self.cachedeval = neweval

        finally:
            self.updateTask = None


        






pricesHandler = Prices()
@app.get("/prices")
async def get_prices(
    hours : int = Query(-1, description="How many hours to predict"),
    fixedPrice : float = Query(0.0, description="Add this fixed amount to all prices (ct/kWh)"),
    taxPercent : float = Query(0.0, description="Tax % to add to the final price"),
    startTs : datetime.datetime|None = Query(None, description="Start output from this time. At most ~60 days"),
    evaluation : bool = Query(False, description="Switches to evaluation mode. All values will be generated by the model, instead of only future values. Useful to evaluate model performance.")):
    """
    Get price prediction
    """
    return await pricesHandler.prices(hours, fixedPrice, taxPercent, startTs, evaluation)


    
