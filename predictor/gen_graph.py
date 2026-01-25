#!/usr/bin/python3

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
from model.pricepredictor import PricePredictor
from model.priceregion import PriceRegionName

START = datetime.fromisoformat("2026-01-01T00:00:00Z")
END = datetime.fromisoformat("2026-01-21T00:00:00Z")
LEARN_DAYS = 120
REGION = PriceRegionName.DE.to_region()

async def main():
    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO
    )

    pred = await PricePredictor(REGION, ".").load_from_persistence()

    learn_start = START - timedelta(days=LEARN_DAYS)
    learn_end = START
    await pred.train(learn_start, learn_end)

    predicted = await pred.predict(START, END, fill_known=False)
    pred_vals = map(float, pred.to_price_dict(predicted).values())
    predicted = predicted.rename(columns={"price": "predicted"})
    actual = await pred.pricestore.get_data(START, END)
    actual_vals = map(float, pred.to_price_dict(actual).values())
    actual = actual.rename(columns={"price": "actual"})

    merged = pd.concat([predicted, actual])
    merged.plot.line(grid=True)
    plt.show()

    pred_vals = [str(round(v, 1)) for v in pred_vals]
    actual_vals = [str(round(v, 1)) for v in actual_vals]

    print(f"""
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
    line [{",".join(actual_vals)}]
    line [{",".join(pred_vals)}]
    """)



if __name__ == "__main__":
    asyncio.run(main())