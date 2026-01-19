#!/usr/bin/python3

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, cast

import pandas as pd
import lightgbm as lgb

from .auxdatastore import AuxDataStore
from .priceregion import PriceRegion
from .pricestore import PriceStore
from .weatherstore import WeatherStore

log = logging.getLogger(__name__)


class PricePredictor:
    region: PriceRegion
    weatherstore: WeatherStore
    pricestore: PriceStore
    auxstore: AuxDataStore

    traindata: pd.DataFrame | None = None

    predictor: lgb.LGBMRegressor | None = None

    def __init__(self, region: PriceRegion, storage_dir: str | None = None):
        self.region = region
        self.weatherstore = WeatherStore(region, storage_dir)
        self.pricestore = PriceStore(region, storage_dir)
        self.auxstore = AuxDataStore(region, storage_dir)

    def is_trained(self) -> bool:
        return self.predictor is not None


    async def train(self, start: datetime, end: datetime):
        self.traindata = await self.prepare_dataframe(start, end, True)
        if self.traindata is None:
            return
        self.traindata.dropna(inplace=True)

        params = self.traindata.drop(columns=["price"])
        output = self.traindata["price"]

        self.predictor = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            force_col_wise=True,
            verbosity=-1
        )
        self.predictor.fit(params, output)



    async def predict(self, start: datetime, end: datetime, fill_known=True) -> pd.DataFrame:
        assert self.is_trained() and self.predictor is not None

        df = await self.prepare_dataframe(start, end, False)
        assert df is not None

        prices_known = df["price"]

        params = df.drop(columns=["price"])

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



    async def prepare_dataframe(self, start: datetime, end: datetime, refresh_prices: bool = True) -> pd.DataFrame | None:
        weather = await self.weatherstore.get_data(start, end)
        if refresh_prices:
            prices = await self.pricestore.get_data(start, end)
        else:
            prices = self.pricestore.get_known_data(start, end)
        auxdata = await self.auxstore.get_data(start, end)

        df = pd.concat([weather, auxdata], axis=1).dropna()
        df = pd.concat([df, prices], axis=1)

        return df

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
        lastknown = self.pricestore.get_last_known()
        if lastknown is None:
            return True

        updated = await self.pricestore.fetch_missing_data(lastknown, datetime.now(timezone.utc) + timedelta(days=3))
        if not updated:
            return False

        lastafter = self.pricestore.get_last_known()
        return lastafter is not None and lastafter != lastknown
    
    def cleanup(self):
        """
        Delete data older than 1 year
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        self.weatherstore.drop_before(cutoff)
        self.pricestore.drop_before(cutoff)
        self.auxstore.drop_before(cutoff)




