#!/usr/bin/python3

import asyncio
import logging
from datetime import datetime, timedelta

from model.pricepredictor import PricePredictor
from model.priceregion import PriceRegion


async def main():
    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO
    )

    pred = PricePredictor(PriceRegion.DE, ".")
    start = datetime.fromisoformat("2025-10-05T00:00:00Z")
    end = start + timedelta(days=90)
    await pred.train(start, end)

    pred_end = end + timedelta(days=6)
    predicted = await pred.predict(end, pred_end, fill_known=False)
    actual = await pred.pricestore.get_data(end, pred_end)

    actuals = [str(round(p, 1)) for p in actual["price"]]
    preds = [str(round(p, 1)) for p in predicted["price"]]

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


if __name__ == "__main__":
    asyncio.run(main())