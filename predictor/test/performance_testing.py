#!/usr/bin/python3

import asyncio
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from predictor.model.auxdatastore import AuxDataStore
import predictor.model.pricepredictor as pred
from predictor.model.priceregion import *
from predictor.model.pricestore import PriceStore
from sklearn.metrics import mean_absolute_error, mean_squared_error
from predictor.model.weatherstore import WeatherStore

START: datetime = datetime.fromisoformat("2025-01-01T00:00:00Z")
END: datetime = datetime.fromisoformat("2026-01-10T00:00:00Z")
REGION : PriceRegion = PriceRegion.DE
LEARN_DAYS : int = 90

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)



async def load_weather() -> WeatherStore:
    weatherstore = WeatherStore(REGION, ".")
    await weatherstore.fetch_missing_data(START, END)
    return weatherstore

async def load_prices() -> PriceStore:
    pricestore = PriceStore(REGION, ".")
    await pricestore.fetch_missing_data(START, END)
    return pricestore

async def load_aux() -> AuxDataStore:
    aux = AuxDataStore(REGION)
    await aux.fetch_missing_data(START, END)
    return aux

async def main():
    #pd.set_option("display.max_rows", None)
    weather = await load_weather()
    prices = await load_prices()
    aux = await load_aux()

    learn_start = START
    learn_end = learn_start + timedelta(days=LEARN_DAYS)

    # MAE/MSE values for 1 to 3 day predictions
    d1_mae = []
    d1_mse = []
    d2_mae = []
    d2_mse = []
    d3_mae = []
    d3_mse = []

    predictor = pred.PricePredictor(REGION)
    predictor.weatherstore = weather
    predictor.pricestore = prices
    predictor.auxstore = aux
    iterations = 0

    while learn_end < END - timedelta(days=3):
        # intervals to predict and check. Could be done nicer but w/e
        d0 = learn_end
        d1 = learn_end + timedelta(days=1)
        d2 = learn_end + timedelta(days=2)
        d3 = learn_end + timedelta(days=3)

        await predictor.train(learn_start, learn_end)


        actual1 = await prices.get_data(d0, d1)
        actual2 = await prices.get_data(d1, d2)
        actual3 = await prices.get_data(d2, d3)
        pred1 = await predictor.predict(d0, d1, False)
        pred2 = await predictor.predict(d1, d2, False)
        pred3 = await predictor.predict(d2, d3, False)

        # Some odd stuff happens during daylight sa

        d1_mae.append(mean_absolute_error(actual1["price"], pred1["price"]))
        d2_mae.append(mean_absolute_error(actual2["price"], pred2["price"]))
        d3_mae.append(mean_absolute_error(actual3["price"], pred3["price"]))
        
        d1_mse.append(mean_squared_error(actual1["price"], pred1["price"]))
        d2_mse.append(mean_squared_error(actual2["price"], pred2["price"]))
        d3_mse.append(mean_squared_error(actual3["price"], pred3["price"]))


        learn_start += timedelta(days=1)
        learn_end += timedelta(days=1)
        iterations += 1


    print(f"iterations tested: {iterations}")
    print(f"1d: mean squared error={round(sum(d1_mse)/len(d1_mse), 2)}, mean absolute error = {round(sum(d1_mae)/len(d1_mae), 2)}")
    print(f"2d: mean squared error={round(sum(d2_mse)/len(d2_mse), 2)}, mean absolute error = {round(sum(d2_mae)/len(d2_mae), 2)}")
    print(f"3d: mean squared error={round(sum(d3_mse)/len(d3_mse), 2)}, mean absolute error = {round(sum(d3_mae)/len(d3_mae), 2)}")




asyncio.run(main())