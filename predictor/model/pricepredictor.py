#!/usr/bin/python3

import math
from typing import Dict, List, Tuple, cast
from enum import Enum
import pandas as pd
from datetime import datetime, timedelta, tzinfo, timezone
from astral import sun, LocationInfo
import statistics
import pytz
import aiohttp
import json
import logging
import os
import asyncio
import holidays
import time

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

log = logging.getLogger(__name__)

class Country(str, Enum):
    DE = "DE"
    AT = "AT"
    BE = "BE"
    NL = "NL"

class CountryConfig:
    COUNTRY_CODE : str
    FILTER : str
    BIDDING_ZONE : str
    LATITUDES : list[float]
    LONGITUDES : list[float]

    def __init__ (self, COUNTRY_CODE, TIMEZONE, BIDDING_ZONE, LATITUDES, LONGITUDES):
        self.COUNTRY_CODE = COUNTRY_CODE
        self.BIDDING_ZONE = BIDDING_ZONE
        self.TIMEZONE = TIMEZONE
        self.LATITUDES = LATITUDES
        self.LONGITUDES = LONGITUDES

# We sample these coordinates for solar/wind/temperature
COUNTRY_CONFIG = {
    Country.DE:  CountryConfig(
        COUNTRY_CODE = "DE",
        BIDDING_ZONE = "DE-LU",
        TIMEZONE = "Europe/Berlin",
        LATITUDES =  [48.4, 49.7, 51.3, 52.8, 53.8, 54.1],
        LONGITUDES = [9.3, 11.3, 8.6, 12.0, 8.1, 11.6]
    ),
    Country.AT : CountryConfig(
        COUNTRY_CODE = "AT",
        BIDDING_ZONE = "AT",
        TIMEZONE = "Europe/Berlin",
        LATITUDES = [48.36, 48.27, 47.32, 47.00, 47.11],
        LONGITUDES = [16.31, 13.85, 10.82, 13.54, 15.80],
    ),
    Country.BE : CountryConfig(
        COUNTRY_CODE = "BE",
        BIDDING_ZONE = "BE",
        TIMEZONE = "Europe/Berlin",
        LATITUDES=[51.27, 50.73, 49.99],
        LONGITUDES=[3.07, 4.79, 5.38],
    ),
    Country.NL : CountryConfig(
        COUNTRY_CODE = "NL",
        BIDDING_ZONE = "NL",
        TIMEZONE = "Europe/Amsterdam",
        LATITUDES=[52.69, 52.36, 50.51],
        LONGITUDES=[6.11, 4.90, 5.41],
    ),
}


class PricePredictor:
    config : CountryConfig
    weather : pd.DataFrame | None = None
    prices: pd.DataFrame | None = None

    fulldata : pd.DataFrame | None = None

    testdata : bool = False
    learnDays : int
    forecastDays : int

    predictor : KNeighborsRegressor | None = None

    def __init__(self, country: Country = Country.DE, testdata : bool = False, learnDays=90, forecastDays=7):
        self.config = COUNTRY_CONFIG[country]
        self.testdata = testdata
        self.learnDays = learnDays
        self.forecastDays = forecastDays

    
    async def train(self, subset=None, prepare=True) -> None:
        # To determine the importance of each parameter, we first weight them using linreg, because knn is treating difference in each parameter uniformly
        if prepare:
            self.fulldata = await self.prepare_dataframe()
        if self.fulldata is None:
            return

        if subset is None:
            learnset = self.fulldata.dropna()
        else:
            learnset = subset.dropna()

        params = learnset.drop(columns=["price"])
        output = learnset["price"]
        linreg = LinearRegression().fit(params, output)
        param_scaling_factors = linreg.coef_
        
        # Apply same scaling to learning set and full data
        params *= param_scaling_factors

        to_scale = self.fulldata.drop(columns=["price"])
        to_scale *= param_scaling_factors
        self.fulldata = pd.concat([to_scale, self.fulldata["price"]], axis=1)

        # Since all numeric values (wind/solar/temperature) now have the same scaling/relevance to the output variable, we can now just sum them up
        # Intention: we don't care if we have a lot of production from wind OR from solar
        #windcols = [f"wind_{i}" for i in range(len(self.config.LATITUDES))]
        #irradiancecols = [f"irradiance_{i}" for i in range(len(self.config.LATITUDES))]
        #tempcols = [f"temp_{i}" for i in range(len(self.config.LATITUDES))]
        #weathercols = windcols + irradiancecols + tempcols

        #params["weathersum"] = params[weathercols].sum(axis=1)
        #params.drop(columns=weathercols, inplace=True)
        #self.fulldata["weathersum"] = self.fulldata[weathercols].sum(axis=1)
        #self.fulldata.drop(columns=weathercols, inplace=True)

        self.predictor = KNeighborsRegressor(n_neighbors=3).fit(params, output)

        

    def is_trained(self) -> bool:
        return self.predictor is not None

    async def predict_raw(self, estimateAll : bool = False) -> pd.DataFrame:
        if self.predictor is None:
            await self.train()
        assert self.fulldata is not None
        assert self.predictor is not None

        predictionDf = self.fulldata.copy()
        predictionDf["price"] = self.predictor.predict(predictionDf.drop(columns=["price"]))

        return predictionDf

    async def predict(self, estimateAll : bool = False) -> Dict[datetime, float]:
        """
        if estimateAll is true, you will get an estimation for the full time range, even if the prices are known already (for performance evaluation).
        if false, you will get known data as is, and only estimations for unknown data
        """
        assert self.fulldata is not None

        predictionDf = await self.predict_raw(estimateAll)

        predDict = self._to_price_dict(predictionDf)

        if not estimateAll:
            knownDict = self._to_price_dict(self.fulldata)
            predDict.update(knownDict)

      
        return predDict

    def _to_price_dict(self, df : pd.DataFrame) -> Dict[datetime, float]:
        result = {}
        for time, row in df.iterrows():
            ts = cast(pd.Timestamp, time).to_pydatetime()
            price = row["price"]
            if math.isnan(price):
                continue
            result[ts] = row["price"]
        return result


    def is_timestamp(self, tz : tzinfo, t : datetime, h : int, m : int) -> int:
        local = t.astimezone(tz)
        return 1 if local.hour == h and local.minute == m else 0

    async def prepare_dataframe(self) -> pd.DataFrame | None:
        if self.weather is None:
            await self.refresh_forecasts()
        if self.prices is None:
            await self.refresh_prices()
        assert self.weather is not None
        assert self.prices is not None
        
        df = self.weather.copy().dropna()
        df = pd.concat([df, self.prices], axis=1).reset_index()
        # allow nan only in price column. All others should be filled with valid data
        datacols = list(df.columns.values)
        datacols.remove("price")
        df = df.dropna(subset=datacols).copy()

        # Drop everything that's older than learnDays (useful if we e.g. load 90 days from testData=True, but only really want to evaluate 30 days)
        df = df[df["time"] >= datetime.now(timezone.utc) - timedelta(days=self.learnDays)]


        tzlocal = pytz.timezone(self.config.TIMEZONE)
        holis = holidays.country_holidays(self.config.COUNTRY_CODE)
        df["holiday"] = df["time"].apply(lambda t: 1 if t.astimezone(tzlocal).weekday() == 6 or t.astimezone(tzlocal).date() in holis else 0)
        for i in range(6):
            df[f"day_{i}"] = df["time"].apply(lambda t: 1 if t.astimezone(tzlocal).weekday() == i else 0)
        #df["saturday"] = df["time"].apply(lambda t: 1 if t.weekday() == 5 else 0)

        timecols : List[pd.Series|pd.DataFrame] = []
        for h in range(0, 24):
            for m in range(0, 60, 15):
                col = df["time"].apply(lambda t: self.is_timestamp(tzlocal, t, h, m))
                col.name = f"i_{h}_{m}"
                timecols.append(col)
        
        locinfo = LocationInfo(name=self.config.COUNTRY_CODE, region=self.config.COUNTRY_CODE, timezone=self.config.TIMEZONE, latitude=statistics.mean(self.config.LATITUDES), longitude=statistics.mean(self.config.LONGITUDES))
        sr_influence = df["time"].apply(lambda t: min(180, abs((t - sun.sun(locinfo.observer, date=t)["sunrise"]).total_seconds() / 60)))
        sr_influence.name = "sr_influence"
        timecols.append(sr_influence)

        ss_influence = df["time"].apply(lambda t: min(180, abs((t - sun.sun(locinfo.observer, date=t)["sunset"]).total_seconds() / 60)))
        ss_influence.name = "ss_influence"
        timecols.append(ss_influence)

        timecols.insert(0, df)
        df = pd.concat(timecols, axis=1)

       
        df.set_index("time", inplace=True)
        return df


    async def refresh_prices(self) -> None:
        log.info("Updating prices...")
        try:
            self.prices = await self.fetch_prices()
            last_price = self.get_last_known_price()
            log.info("Price update done. Prices available until " + last_price[0].isoformat() if last_price is not None else "UNEXPECTED NONE")
        except Exception as e:
            log.warning(f"Failed to update prices : {str(e)}")
    
    async def refresh_forecasts(self) -> None:
        log.info("Updating weather forecast...")
        try:
            self.weather = await self.fetch_weather()
            log.info("Weather update done")
        except Exception as e:
            log.warning(f"Failed to update forecast : {str(e)}")
        
    
    async def fetch_weather(self) -> pd.DataFrame | None:
        cacheFn = f"weather_{self.config.COUNTRY_CODE}.json"
        if self.testdata and os.path.exists(cacheFn):
            log.warning("Loading weather from persistent cache!")
            await asyncio.sleep(0) # simulate async http
            weather = pd.read_json(cacheFn)
            weather.index = weather.index.tz_localize("UTC") # type: ignore
            weather.index.set_names("time", inplace=True)
            return weather

        lats = ",".join(map(str, self.config.LATITUDES))
        lons = ",".join(map(str, self.config.LONGITUDES))
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&azimuth=0&tilt=0&past_days={self.learnDays}&forecast_days={self.forecastDays}&minutely_15=wind_speed_80m,temperature_2m,global_tilted_irradiance&timezone=UTC"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.text()

                data = json.loads(data)
                frames = []
                for i, fc in enumerate(data):
                    df = pd.DataFrame(columns=["time", f"wind_{i}", f"temp_{i}"])
                    times = fc["minutely_15"]["time"]
                    winds = fc["minutely_15"]["wind_speed_80m"]
                    temps = fc["minutely_15"]["temperature_2m"]
                    irradiance = fc["minutely_15"]["global_tilted_irradiance"]
                    df["time"] = times
                    df[f"irradiance_{i}"] = irradiance
                    df[f"wind_{i}"] = winds
                    df[f"temp_{i}"] = temps
                    df.set_index("time", inplace=True)
                    df.dropna(inplace=True)
                    frames.append(df)

                df = pd.concat(frames, axis=1).reset_index()
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df.set_index("time", inplace=True)

                if self.testdata:
                    df.to_json(cacheFn)
                
                return df

    async def fetch_prices(self) -> pd.DataFrame | None:
        cacheFn = f"prices_{self.config.COUNTRY_CODE}.json"
        if self.testdata and os.path.exists(cacheFn):
            log.warning("Loading prices from persistent cache!")
            await asyncio.sleep(0) # simulate async http
            prices = pd.read_json(cacheFn)
            prices.index = prices.index.tz_localize("UTC") # type: ignore
            prices.index.set_names("time", inplace=True)
            return prices

        startTs = (datetime.now(timezone.utc) - timedelta(days=self.learnDays)).timestamp()
        endTs = (datetime.now() + timedelta(days=5)).timestamp()
        url = f"https://api.energy-charts.info/price?bzn={self.config.BIDDING_ZONE}&start={int(startTs)}&end={int(endTs)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"accept": "application/json"}) as resp:
                data = await resp.json()
                timestamps = data["unix_seconds"]
                prices = data["price"]
                data = pd.DataFrame.from_dict(dict(zip(timestamps, prices)), orient="index", columns=["price"])
                data.index = pd.to_datetime(data.index, unit="s", utc=True)
                data.index.name = "time"
                data["price"] = data["price"] / 10

                if self.testdata:
                    data.to_json(cacheFn)
                return data


    def get_last_known_price(self) -> Tuple[datetime, float] | None:
        if self.prices is None:
            return None
        lastrow = self.prices.dropna().reset_index().iloc[-1]
        return lastrow["time"].to_pydatetime(), float(lastrow["price"])




async def main():
    import sys
    pd.set_option("display.max_rows", None)
    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO
    )

    pred = PricePredictor(testdata=True)
    await pred.train()

    actual = await pred.predict()
    predicted = await pred.predict(estimateAll=True)

    #xdt : List[datetime] = list(actual.keys())
    #x = map(str, range(0, len(actual)))
    actuals = map(lambda p: str(round(p, 1)), actual.values())
    preds = map(lambda p: str(round(p, 1)), predicted.values())

    start = 500
    end = start+14*24*4

    #x = list(x)[start:end]
    actuals = list(actuals)[start:end]
    preds = list(preds)[start:end]

    print (
        f"""
---
config:
    xyChart:
        width: 1700
        height: 900
        plotReservedSpacePercent: 80
        xAxis:
            showLabel: false
---
xychart-beta
    title "Performance comparison"
    line [{",".join(actuals)}]
    line [{",".join(preds)}]
    """)
    

    
    """prices = pred.predict()
    prices = {
        k.isoformat(): v for k, v in prices.items()
    }
    print(json.dumps(prices))"""

if __name__ == "__main__":
    asyncio.run(main())
