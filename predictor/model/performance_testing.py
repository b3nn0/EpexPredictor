#!/usr/bin/python3

import asyncio
from sklearn.metrics import mean_squared_error
import pricepredictor as pred
import logging

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)


async def main():
    pp = pred.PricePredictor(testdata=True, country=pred.Country.DE)
    # pp = pred.PricePredictor(testdata=True, country=pred.Country.AT)
    fulldata = await pp.prepare_dataframe()
    assert fulldata is not None
    fulldata.dropna(inplace=True)

    n = 100
    errorsum = 0
    for i in range(n):
        pp.fulldata = fulldata.copy()
        train = fulldata.sample(frac=0.9) # train only on a random subset of data
        await pp.train(subset=train, prepare=False)

        test = fulldata.drop(train.index) # but use the remaining data for actual prediction
        prediction = await pp.predict_raw(estimateAll=True)
        prediction = prediction.drop(train.index)

        test.set_index("time", inplace=True)
        prediction.set_index("time", inplace=True)

        errorsum += mean_squared_error(test[['price']], prediction[['price']])

    print(f"error={errorsum/n}")





        











asyncio.run(main())