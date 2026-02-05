#!/usr/bin/python3

import asyncio
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
from .entsoedatastore import EntsoeDataStore
from .gaspricestore import GasPriceStore

log = logging.getLogger(__name__)


class PricePredictor:
    region: PriceRegion
    weatherstore: WeatherStore
    pricestore: PriceStore
    entsoestore: EntsoeDataStore
    auxstore: AuxDataStore
    gasstore: GasPriceStore

    traindata: pd.DataFrame | None = None

    predictor: lgb.Booster | None = None

    def __init__(self, region: PriceRegion, storage_dir: str | None = None):
        self.region = region
        self.weatherstore = WeatherStore(region, storage_dir)
        self.pricestore = PriceStore(region, storage_dir)
        self.auxstore = AuxDataStore(region, storage_dir)
        self.entsoestore = EntsoeDataStore(region, storage_dir)
        self.gasstore = GasPriceStore(region, storage_dir)

    async def load_from_persistence(self):
        await asyncio.gather(
            self.weatherstore.load(),
            self.pricestore.load(),
            self.auxstore.load(),
            self.entsoestore.load(),
            self.gasstore.load()
        )
        return self
    
    def last_data_update(self) -> datetime:
        return max(self.weatherstore.last_updated, self.pricestore.last_updated, self.entsoestore.last_updated, self.gasstore.last_updated)

    def use_datastores_from(self, other: "PricePredictor"):
        assert self.region.bidding_zone_entsoe == other.region.bidding_zone_entsoe
        self.weatherstore = other.weatherstore
        self.pricestore = other.pricestore
        self.auxstore = other.auxstore
        self.entsoestore = other.entsoestore
        self.gasstore = other.gasstore

    def is_trained(self) -> bool:
        return self.predictor is not None


    async def train(self, start: datetime, end: datetime):
        self.traindata = await self.prepare_dataframe(start, end)
        if self.traindata is None:
            return
        self.traindata.dropna(inplace=True)

        params = self.traindata.drop(columns=["price"])
        output = self.traindata["price"]

        lgb_dataset = lgb.Dataset(params, label=output)
        params_lgb = {
            "force_col_wise": True,
            "verbosity": -1,
        }

        self.predictor = await asyncio.to_thread(lgb.train, params=params_lgb, train_set=lgb_dataset)



    async def predict(self, start: datetime, end: datetime, fill_known=True) -> pd.DataFrame:
        assert self.is_trained() and self.predictor is not None

        df = await self.prepare_dataframe(start, end)
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



    async def prepare_dataframe(self, start: datetime, end: datetime) -> pd.DataFrame | None:
        weather, prices, auxdata = await asyncio.gather(
            self.weatherstore.get_data(start, end),
            self.pricestore.get_data(start, end),
            self.auxstore.get_data(start, end)
        )

        df = pd.concat([weather, auxdata], axis=1, sort=True)

        
        if self.region.use_entsoe_load_forecast:
            entsoedata = await self.entsoestore.get_data(start, end)
            if len(entsoedata) > 0:
                df = pd.concat([df, entsoedata], axis=1, sort=True)

        if self.region.use_de_nat_gas_price:
            gasprices = await self.gasstore.get_data(start, end)
            gasprices = gasprices.reindex(weather.index).ffill()
            df = pd.concat([df, gasprices], axis=1, sort=True)

        df = pd.concat([df, prices], axis=1, sort=True)
        return df

    async def refresh_forecasts(self, start : datetime, end: datetime):
        """
            Will re-fetch everything starting from yesterday during next training
            Not sure when past data becomes "stable", so better be sure and fetch a bit more
        """
        await self.weatherstore.refresh_range(start, end)
        await self.entsoestore.refresh_range(start, end)


    def cleanup(self):
        """
        Delete data older than 1 year
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        self.weatherstore.drop_before(cutoff)
        self.pricestore.drop_before(cutoff)
        self.auxstore.drop_before(cutoff)
        self.entsoestore.drop_before(cutoff)
        self.gasstore.drop_before(cutoff)




