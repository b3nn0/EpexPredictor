#!/usr/bin/python3

import math
from typing import Dict, List, Tuple, cast
from enum import Enum
import pandas as pd
from datetime import datetime, date, timedelta, tzinfo, timezone
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

from sklearn.linear_model import Lasso
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
    HOLIDAYS : list[holidays.HolidayBase] # one entry for each regional holiday set, e.g. one for BW, one for BY, ...
    

    def __init__ (self, COUNTRY_CODE, TIMEZONE, BIDDING_ZONE, LATITUDES, LONGITUDES):
        self.COUNTRY_CODE = COUNTRY_CODE
        self.BIDDING_ZONE = BIDDING_ZONE
        self.TIMEZONE = TIMEZONE
        self.LATITUDES = LATITUDES
        self.LONGITUDES = LONGITUDES

        self.HOLIDAYS = []
        country_holidays = holidays.country_holidays(self.COUNTRY_CODE)
        if country_holidays.subdivisions is None or len(country_holidays.subdivisions) == 0:
            self.HOLIDAYS.append(country_holidays)
        else:
            for subdiv in country_holidays.subdivisions:
                self.HOLIDAYS.append(holidays.country_holidays(country=self.COUNTRY_CODE, subdiv=subdiv))


# We sample these coordinates for solar/wind/temperature
COUNTRY_CONFIG = {
    Country.DE:  CountryConfig(
        COUNTRY_CODE = "DE",
        BIDDING_ZONE = "DE-LU",
        TIMEZONE = "Europe/Berlin",
        # 64 locations: Aggressive distribution - large states get 5-6, medium get 4, small get 3
        # BW (5): Stuttgart, Freiburg, Karlsruhe, Ulm, Mannheim
        LATITUDES =  [48.78, 47.99, 49.01, 48.40, 49.49,
        # BY (6): Munich, Nuremberg, Regensburg, Augsburg, Würzburg, Ingolstadt
                      48.14, 49.45, 49.02, 48.37, 49.79, 48.76,
        # NI (5): Hannover, Oldenburg, Braunschweig, Osnabrück, Göttingen
                      52.37, 53.14, 52.27, 52.28, 51.54,
        # NW (6): Cologne, Dortmund, Düsseldorf, Münster, Essen, Bielefeld
                      50.94, 51.51, 51.23, 51.96, 51.46, 52.02,
        # BB (4): Potsdam, Cottbus, Frankfurt/Oder, Brandenburg
                      52.40, 51.76, 52.34, 52.41,
        # SH (4): Kiel, Lübeck, Flensburg, Husum (offshore wind)
                      54.32, 53.87, 54.78, 54.48,
        # MV (4): Rostock, Schwerin, Greifswald, Stralsund
                      54.09, 53.63, 54.10, 54.31,
        # SN (4): Dresden, Leipzig, Chemnitz, Zwickau
                      51.05, 51.34, 50.83, 50.72,
        # ST (4): Magdeburg, Halle, Dessau, Quedlinburg
                      51.95, 51.48, 51.84, 51.79,
        # HE (4): Frankfurt, Kassel, Wiesbaden, Darmstadt
                      50.11, 51.32, 50.08, 49.87,
        # RP (4): Mainz, Trier, Koblenz, Ludwigshafen
                      49.99, 49.76, 50.36, 49.48,
        # TH (4): Erfurt, Jena, Gera, Weimar
                      50.98, 50.93, 50.88, 50.98,
        # BE (3): Berlin-Mitte, Berlin-Spandau, Berlin-Köpenick
                      52.52, 52.53, 52.45,
        # HB (3): Bremen, Bremerhaven, Bremen-Nord
                      53.08, 53.55, 53.17,
        # HH (3): Hamburg-Center, Hamburg-Harburg, Hamburg-Bergedorf
                      53.55, 53.46, 53.49,
        # SL (3): Saarbrücken, Neunkirchen, Saarlouis
                      49.24, 49.35, 49.31],
        LONGITUDES = [9.18, 7.85, 8.40, 9.99, 8.47,
                      11.58, 11.08, 12.10, 10.90, 9.94, 11.42,
                      9.74, 8.21, 10.52, 7.97, 9.93,
                      6.96, 7.47, 6.77, 7.63, 7.01, 8.53,
                      13.07, 14.33, 14.55, 12.56,
                      10.14, 10.69, 9.54, 9.05,
                      12.10, 11.42, 13.38, 13.09,
                      13.74, 12.38, 12.92, 12.48,
                      11.93, 11.97, 12.24, 11.14,
                      8.68, 9.48, 8.24, 8.65,
                      8.27, 6.64, 7.60, 8.44,
                      11.03, 11.59, 12.08, 11.33,
                      13.40, 13.20, 13.57,
                      8.80, 8.58, 8.65,
                      10.00, 9.97, 10.21,
                      6.99, 7.21, 6.75],
    ),
    Country.AT : CountryConfig(
        COUNTRY_CODE = "AT",
        BIDDING_ZONE = "AT",
        TIMEZONE = "Europe/Berlin",
        # 24 locations: 2-3 per state covering Vienna, Lower/Upper Austria, Styria, Tyrol, Carinthia, Salzburg, Vorarlberg, Burgenland
        # Vienna (2), Lower Austria (3), Upper Austria (3), Styria (3), Tyrol (3), Carinthia (2), Salzburg (3), Vorarlberg (2), Burgenland (3)
        LATITUDES = [48.21, 48.27, 48.31, 48.09, 47.68, 48.31, 48.24, 48.03, 47.07, 47.27, 47.56, 47.26, 47.07, 46.62, 47.80, 47.81, 47.48, 47.30, 47.08, 46.77, 47.62, 47.08, 47.85, 47.31],
        LONGITUDES = [16.37, 16.17, 15.63, 16.25, 15.44, 14.29, 14.51, 13.93, 15.44, 15.04, 14.29, 13.09, 11.40, 14.31, 13.04, 12.88, 13.38, 10.90, 12.68, 13.37, 14.66, 9.67, 16.53, 16.38],
    ),
    Country.BE : CountryConfig(
        COUNTRY_CODE = "BE",
        BIDDING_ZONE = "BE",
        TIMEZONE = "Europe/Berlin",
        # 12 locations: 4 per region (Flanders, Wallonia, Brussels) covering major cities and geographic spread
        # Flanders (4): Antwerp, Ghent, Bruges, Leuven
        # Wallonia (4): Liège, Charleroi, Namur, Mons
        # Brussels + extras (4): Brussels, Mechelen, Hasselt, Arlon
        LATITUDES = [51.22, 51.05, 51.21, 50.88, 50.63, 50.41, 50.47, 50.45, 50.85, 51.03, 50.93, 49.68],
        LONGITUDES = [4.40, 3.72, 3.22, 4.70, 5.57, 4.44, 4.87, 3.95, 4.35, 4.48, 5.33, 5.82],
    ),
    Country.NL : CountryConfig(
        COUNTRY_CODE = "NL",
        BIDDING_ZONE = "NL",
        TIMEZONE = "Europe/Amsterdam",
        # 18 locations: 1-2 per province covering coastal, central, and southern regions
        # North: Groningen, Friesland, Drenthe | West: North/South Holland, Utrecht | Central: Gelderland, Flevoland, Overijssel
        # South: North Brabant, Limburg, Zeeland
        LATITUDES = [53.22, 53.20, 52.99, 52.37, 52.16, 52.09, 51.99, 52.51, 52.52, 52.42, 51.69, 51.44, 51.56, 51.81, 51.99, 51.44, 51.35, 50.85],
        LONGITUDES = [6.57, 5.79, 6.56, 4.90, 4.50, 5.12, 5.89, 6.08, 5.47, 4.62, 4.78, 5.47, 5.09, 5.84, 4.14, 3.61, 6.17, 5.69],
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
        # To determine the importance of each parameter, we first weight them using Lasso regression
        # Lasso (L1 regularization) helps by zeroing out less important features
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

        # Use Lasso regression for feature weighting (alpha=0.1 provides good regularization)
        reg = Lasso(alpha=0.1).fit(params, output)
        param_scaling_factors = reg.coef_

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

        self.predictor = KNeighborsRegressor(n_neighbors=7).fit(params, output)

        

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
    
    def is_holiday(self, t : datetime) -> float:
        if t.weekday() == 6:
            return 1
        
        date = t.date()
       
        cnt_holiday = 0
        for h in self.config.HOLIDAYS:
            if date in h:
                cnt_holiday += 1
        # Average regional holidays. E.g. if it's a holiday in half of the regions -> 0.5
        result = cnt_holiday / len(self.config.HOLIDAYS)
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

        # Drop everything that's older than learnDays (useful if we e.g. load 90 days from testData=True, but only really want to evaluate 30 days)
        df = df[df["time"] >= datetime.now(timezone.utc) - timedelta(days=self.learnDays)]


        tzlocal = pytz.timezone(self.config.TIMEZONE)
        df["holiday"] = df["time"].apply(lambda t: self.is_holiday(t.astimezone(tzlocal)))
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
