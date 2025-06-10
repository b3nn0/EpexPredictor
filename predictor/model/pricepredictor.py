#!/usr/bin/python3

import math
from typing import Dict, List, Tuple, cast
from enum import Enum
import pandas as pd
import datetime
import aiohttp
import json
import logging
import os
import asyncio
import holidays
import time
import pytz

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

log = logging.getLogger(__name__)

class Country(str, Enum):
    DE = 'DE'
    AT = 'AT'

class CountryConfig:
    COUNTRY_CODE : str
    FILTER : str
    LATITUDES : list[float]
    LONGITUDES : list[float]

    def __init__ (self, COUNTRY_CODE, FILTER, LATITUDES, LONGITUDES):
        self.COUNTRY_CODE = COUNTRY_CODE
        self.FILTER = FILTER
        self.LATITUDES = LATITUDES
        self.LONGITUDES = LONGITUDES

# We sample these coordinates for solar/wind/temperature
COUNTRY_CONFIG = {
        Country.DE:  CountryConfig(
                COUNTRY_CODE = 'DE',
                FILTER = '4169',
                LATITUDES =  [
                    48.4,
                    49.7,
                    51.3,
                    52.8,
                    53.8,
                    54.1
                ],
                LONGITUDES = [
                    9.3,
                    11.3,
                    8.6,
                    12.0,
                    8.1,
                    11.6
                ]
               ),
        Country.AT : CountryConfig(
                COUNTRY_CODE = 'AT',
                FILTER = '4170',
                LATITUDES = [
                    48.36,
                    48.27,
                    47.32,
                    47.00,
                    47.11
                ],
                LONGITUDES = [
                    16.31,
                    13.85,
                    10.82,
                    13.54,
                    15.80
                ],
               ),
        }


class PricePredictor:
    config : CountryConfig
    weather : pd.DataFrame | None = None
    prices: pd.DataFrame | None = None

    fulldata : pd.DataFrame | None = None

    testdata : bool = False
    learnDays : int = 30
    forecastDays : int

    predictor : KNeighborsRegressor | None = None

    def __init__(self, country: Country = Country.DE, testdata : bool = False, learnDays=30, forecastDays=7):
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
        orig_price = self.fulldata["price"]
        self.fulldata.drop(columns=["price"], inplace=True)
        self.fulldata *= param_scaling_factors
        self.fulldata["price"] = orig_price

        # Since all numeric values (wind/solar/temperature) now have the same scaling/relevance to the output variable, we can now just sum them up
        # Intention: we don't care if we have a lot of production from wind OR from solar
        windcols = [f"wind_{i}" for i in range(len(self.config.LATITUDES))]
        irradiancecols = [f"irradiance_{i}" for i in range(len(self.config.LATITUDES))]
        tempcols = [f"temp_{i}" for i in range(len(self.config.LATITUDES))]
        weathercols = windcols + irradiancecols + tempcols

        params["weathersum"] = params[weathercols].sum(axis=1)
        params.drop(columns=weathercols, inplace=True)
        self.fulldata["weathersum"] = self.fulldata[weathercols].sum(axis=1)
        self.fulldata.drop(columns=weathercols, inplace=True)

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

    async def predict(self, estimateAll : bool = False) -> Dict[datetime.datetime, float]:
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

    def _to_price_dict(self, df : pd.DataFrame) -> Dict[datetime.datetime, float]:
        result = {}
        for time, row in df.iterrows():
            ts = cast(pd.Timestamp, time).to_pydatetime()
            price = row["price"]
            if math.isnan(price):
                continue
            result[ts] = row["price"]
        return result


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

        tzlocal = pytz.timezone("Europe/Berlin")
        holis = holidays.country_holidays(self.config.COUNTRY_CODE)
        df["holiday"] = df["time"].apply(lambda t: 1 if t.astimezone(tzlocal).weekday() == 6 or t.astimezone(tzlocal).date() in holis else 0)
        for i in range(6):
            df[f"day_{i}"] = df["time"].apply(lambda t: 1 if t.astimezone(tzlocal).weekday() == i else 0)
        #df["saturday"] = df["time"].apply(lambda t: 1 if t.weekday() == 5 else 0)
        for h in range(0, 24):
            df[f"h_{h}"] = df["time"].apply(lambda t: 1 if t.astimezone(tzlocal).hour == h else 0)
        
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
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&azimuth=0&tilt=0&past_days={self.learnDays}&forecast_days={self.forecastDays}&hourly=wind_speed_80m,temperature_2m,global_tilted_irradiance&timezone=UTC"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.text()

                data = json.loads(data)
                frames = []
                for i, fc in enumerate(data):
                    df = pd.DataFrame(columns=["time", f"wind_{i}", f"temp_{i}"]) # type: ignore
                    times = fc["hourly"]["time"]
                    winds = fc["hourly"]["wind_speed_80m"]
                    temps = fc["hourly"]["temperature_2m"]
                    irradiance = fc["hourly"]["global_tilted_irradiance"]
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

        filter = self.config.FILTER # marktpreis
        region = self.config.COUNTRY_CODE 
        filterCopy = filter
        regionCopy = region
        resolution = "hour"

        # Get available timestamps
        url = f"https://www.smard.de/app/chart_data/{filter}/{region}/index_{resolution}.json"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.text()
                timestamps : List[int] = json.loads(data)["timestamps"]
                timestamps.sort()

            startTs = 1000 * (int(time.time()) - self.learnDays * 24 * 60 * 60)
        
            startIndex = len(timestamps) - 1
            for i, timestamp in enumerate(timestamps):
                if timestamp > startTs:
                    startIndex = i - 1
                    break
            timestamps = timestamps[startIndex:]

            pricesDict = {}

            for timestamp in timestamps:
                url = f"https://www.smard.de/app/chart_data/{filter}/{region}/{filterCopy}_{regionCopy}_{resolution}_{timestamp}.json"
                async with session.get(url) as resp:
                    data = await resp.text()
                    series : List = json.loads(data)["series"]
                    for entry in series:
                        price = entry[1]
                        if price is None:
                            continue
                        dt = datetime.datetime.fromtimestamp(entry[0] / 1000, tz=pytz.timezone("Europe/Berlin"))
                        dt = dt.astimezone(pytz.UTC)
                        pricesDict[dt] = price
            
            data = pd.DataFrame.from_dict(pricesDict, orient="index", columns=["price"]).reset_index() # type: ignore
            data.rename(columns={"index": "time"}, inplace=True)
            data["time"] = pd.to_datetime(data["time"], utc=True)
            data["price"] = data["price"] / 10
            data.set_index("time", inplace=True)

            if self.testdata:
                data.to_json(cacheFn)

            return data

    def get_last_known_price(self) -> Tuple[datetime.datetime, float] | None:
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

    pred = PricePredictor(testdata=False)
    await pred.train()

    actual = await pred.predict()
    predicted = await pred.predict(estimateAll=True)

    #xdt : List[datetime.datetime] = list(actual.keys())
    #x = map(str, range(0, len(actual)))
    actuals = map(lambda p: str(round(p, 1)), actual.values())
    preds = map(lambda p: str(round(p, 1)), predicted.values())

    start = 100
    end = start+14*24

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
