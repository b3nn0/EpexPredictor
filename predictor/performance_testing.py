#!/usr/bin/python3

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import cast

from sklearn.metrics import mean_absolute_error, mean_squared_error

import model.pricepredictor as pred
from model.auxdatastore import AuxDataStore
from model.priceregion import PriceRegion
from model.pricestore import PriceStore
from model.weatherstore import WeatherStore
from model.datastore import DataStore

START: datetime = datetime.fromisoformat("2025-01-17T00:00:00Z")
END: datetime = datetime.fromisoformat("2026-01-17T00:00:00Z")
REGION : PriceRegion = PriceRegion.DE
LEARN_DAYS : int = 120

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)


async def load_data(store_class, cache_dir) -> DataStore:
    store = store_class(REGION, cache_dir)
    await store.fetch_missing_data(START - timedelta(days=LEARN_DAYS), END)
    return store


async def main():
    learn_start = START - timedelta(days=LEARN_DAYS)
    learn_end = START

    # MAE/RMSE values for 1 to 3 day predictions
    d1_mae = []
    d1_mse = []
    d2_mae = []
    d2_mse = []
    d3_mae = []
    d3_mse = []

    predictor = pred.PricePredictor(REGION)
    # preload data for whole time range to reduce individual http requests
    predictor.weatherstore = cast(WeatherStore, await load_data(WeatherStore, "."))
    predictor.pricestore = cast(PriceStore, await load_data(PriceStore, "."))
    predictor.auxstore = cast(AuxDataStore, await load_data(AuxDataStore, None))
    iterations = 0

    while learn_end < END - timedelta(days=3):
        
        # intervals to predict and check. Could be done nicer but w/e
        d0 = learn_end
        d1 = learn_end + timedelta(days=1)
        d2 = learn_end + timedelta(days=2)
        d3 = learn_end + timedelta(days=3)

        # Make sure training/prediction doesn't "cheat" with data that is known during performance testing, but not for actual forecasts
        predictor.pricestore.horizon_cutoff = learn_end

        await predictor.train(learn_start, learn_end - timedelta(minutes=15)) # exclusive last
        prediction = await predictor.predict(d0, d3, False)

        predictor.pricestore.horizon_cutoff = None
        actual = await predictor.pricestore.get_data(d0, d3)


        d1_mae.append(mean_absolute_error(actual.loc[d0:d1]["price"], prediction.loc[d0:d1]["price"]))
        d2_mae.append(mean_absolute_error(actual.loc[d1:d2]["price"], prediction.loc[d1:d2]["price"]))
        d3_mae.append(mean_absolute_error(actual.loc[d2:d3]["price"], prediction.loc[d2:d3]["price"]))

        d1_mse.append(mean_squared_error(actual.loc[d0:d1]["price"], prediction.loc[d0:d1]["price"]))
        d2_mse.append(mean_squared_error(actual.loc[d1:d2]["price"], prediction.loc[d1:d2]["price"]))
        d3_mse.append(mean_squared_error(actual.loc[d2:d3]["price"], prediction.loc[d2:d3]["price"]))
        
        learn_start += timedelta(days=1)
        learn_end += timedelta(days=1)
        iterations += 1
        print('.', end='')

    print()


    print(f"iterations tested: {iterations}")
    print(f"1d: RMSE={round(math.sqrt(sum(d1_mse)/len(d1_mse)), 2)}, MAE={round(sum(d1_mae)/len(d1_mae), 2)}")
    print(f"2d: RMSE={round(math.sqrt(sum(d2_mse)/len(d2_mse)), 2)}, MAE={round(sum(d2_mae)/len(d2_mae), 2)}")
    print(f"3d: RMSE={round(math.sqrt(sum(d3_mse)/len(d3_mse)), 2)}, MAE={round(sum(d3_mae)/len(d3_mae), 2)}")




asyncio.run(main())
