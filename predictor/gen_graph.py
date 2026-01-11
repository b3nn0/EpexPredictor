#!/usr/bin/python3

import asyncio
import logging
from datetime import datetime, timedelta

from model.pricepredictor import PricePredictor
from model.priceregion import PriceRegion


async def main():
    import sys

    #pd.set_option("display.max_rows", None)
    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO
    )

    pred = PricePredictor(PriceRegion.DE, ".")
    start = datetime.fromisoformat("2025-03-01T00:00:00Z")
    end = start + timedelta(days=90)
    await pred.train(start, end)

    pred_end = end + timedelta(days=6)
    predicted = await pred.predict(end, pred_end, fill_known=False)
    actual = await pred.pricestore.get_data(end, pred_end)

    #xdt : List[datetime] = list(actual.keys())
    #x = map(str, range(0, len(actual)))
    actuals = map(lambda p: str(round(p, 1)), actual["price"])
    preds = map(lambda p: str(round(p, 1)), predicted["price"])

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