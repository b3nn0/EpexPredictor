#!/usr/bin/python3

import asyncio
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pricepredictor as pred
import logging

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)


async def main():
    pp = pred.PricePredictor(testdata=True, country=pred.Country.DE)
    #pp = pred.PricePredictor(testdata=False, country=pred.Country.AT)
    fulldata = await pp.prepare_dataframe()
    #fulldata.to_csv("/tmp/data.csv")
    #print(fulldata)
    assert fulldata is not None
    fulldata.dropna(inplace=True)

    n = 500
    sqerror = 0
    abserror = 0
    for i in range(n):
        pp.fulldata = fulldata.copy()
        train = fulldata.sample(frac=0.9) # train only on a random subset of data
        #train = fulldata[0:int(0.9*len(fulldata))]
        await pp.train(subset=train, prepare=False)

        test = fulldata.drop(train.index) # but use the remaining data for actual prediction
        prediction = await pp.predict_raw(estimateAll=True)
        prediction = prediction.drop(train.index)

        sqerror += mean_squared_error(test[['price']], prediction[['price']])
        abserror += mean_absolute_error(test[['price']], prediction[['price']])

    print(f"mean squared error={sqerror/n}, mean absolute error = {abserror/n}")





        











asyncio.run(main())