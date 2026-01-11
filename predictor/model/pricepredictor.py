#!/usr/bin/python3

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Dict, List, Tuple, cast

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from .auxdatastore import AuxDataStore
from .priceregion import *
from .pricestore import PriceStore
from .weatherstore import WeatherStore

log = logging.getLogger(__name__)


class PricePredictor:
    region: PriceRegion
    weatherstore: WeatherStore
    pricestore: PriceStore
    auxstore : AuxDataStore

    traindata: pd.DataFrame | None = None
    param_scaling_factors = None

    predictor: KNeighborsRegressor | None = None

    def __init__(self, region: PriceRegion, storage_dir: str|None = None):
        self.region = region
        self.weatherstore = WeatherStore(region, storage_dir)
        self.pricestore = PriceStore(region, storage_dir)
        self.auxstore = AuxDataStore(region)

    def is_trained(self) -> bool:
        return self.predictor is not None

    
    async def train(self, start: datetime, end: datetime):
        # To determine the importance of each parameter, we first weight them using Lasso regression
        # Lasso (L1 regularization) helps by zeroing out less important features

        self.traindata = await self.prepare_dataframe(start, end, True)
        if self.traindata is None:
            return
        self.traindata.dropna(inplace=True)

        params = self.traindata.drop(columns=["price"])
        output = self.traindata["price"]

        # Use Lasso regression for feature weighting (alpha=0.1 provides good regularization)
        #reg = Lasso(alpha=0.1).fit(params, output)
        reg = LinearRegression().fit(params, output)
        self.param_scaling_factors = reg.coef_

        # Apply same scaling to learning set and full data
        params *= self.param_scaling_factors

        self.predictor = KNeighborsRegressor(n_neighbors=7).fit(params, output)



    async def predict(self, start: datetime, end: datetime, fill_known=True) -> pd.DataFrame:
        assert self.is_trained() and self.predictor is not None

        df = await self.prepare_dataframe(start, end, False)
        assert df is not None

        prices_known = df["price"]

        params = df.drop(columns=["price"])
        params *= self.param_scaling_factors

        resultdf = pd.DataFrame(index=params.index)
        resultdf["price"] = self.predictor.predict(params)
        
        if fill_known:
            resultdf.update(prices_known)

        return resultdf

    def to_price_dict(self, df : pd.DataFrame) -> Dict[datetime, float]:
        result = {}
        for time, row in df.iterrows():
            ts = cast(pd.Timestamp, time).to_pydatetime()
            price = row["price"]
            if math.isnan(price):
                continue
            result[ts] = row["price"]
        return result



    async def prepare_dataframe(self, start: datetime, end: datetime, refresh_prices) -> pd.DataFrame | None:
        weather = await self.weatherstore.get_data(start, end)
        if refresh_prices:
            prices = await self.pricestore.get_data(start, end)
        else:
            prices = self.pricestore.get_known_data(start, end)
        auxdata = await self.auxstore.get_data(start, end)
        
        df = pd.concat([weather, auxdata], axis=1).dropna()
        df = pd.concat([df, prices], axis=1)
        return df

    def get_last_known_price(self) -> Tuple[datetime, float] | None:
        prices = self.pricestore.data
        if len(prices) == 0:
            return None
        lastrow = prices.dropna().reset_index().iloc[-1]
        return lastrow["time"].to_pydatetime(), float(lastrow["price"])


    async def refresh_weather(self, start : datetime, end: datetime):
        """
            Will re-fetch everything starting from yesterday during next training
            Not sure when past data becomes "stable", so better be sure and fetch a bit more
            TODO: might want to make this more robust to keep old weather data in case OpenMeteo is not reachable
        """
        await self.weatherstore.refresh_range(start, end)


    async def refresh_prices(self) -> bool:
        """
        true if actual new prices are available
        """
        lastknown = self.get_last_known_price()
        if lastknown is None:
            return True
        lastdt, _ = lastknown

        updated = await self.pricestore.fetch_missing_data(lastdt, datetime.now(timezone.utc) + timedelta(days=3))
        if not updated:
            return False
        
        lastafter = self.get_last_known_price()
        if lastafter is not None and lastafter[0] != lastdt:
            return True
        return False
    
    def cleanup(self):
        """
        Delete data older than 1 year
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        self.weatherstore.drop_before(cutoff)
        self.pricestore.drop_before(cutoff)
        self.auxstore.drop_before(cutoff)




