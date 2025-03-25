from typing import Dict

import pytz
from flask import Flask, request
import datetime
import dateutil.parser
import json

app = Flask("pricepredictor")


@app.route("/")
def api_help():
    return """
        <html>
        <head>
            <title>Price Predictor API</title>
            <style>
                body {
                    font-family: 'Open Sans', sans-serif;
                    line-height: 1.5;
                }
                table {
                    border-collapse: collapse;
                }
                th, td {
                    border: 1px solid black;
                    padding: 8px;
                }
               
            </style>
        </head>
        <body>
        <h1>Endpoint /prices</h1>
        <h2>Parameters</h2>
        <table>
            <thead>
                <tr>
                    <th>Param</th>
                    <th>Description</th>
                    <th>Values</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>hours</td>
                    <td>How many hours to predict</td>
                    <td>integer, Default: as many as possible</td>
                </tr>
                <tr>
                    <td>fixedPrice</td>
                    <td>Add this fixed amount to all prices (ct/kWh)</td>
                    <td>float, Default 0.0</td>
                </tr>
                <tr>
                    <td>taxPercent</td>
                    <td>Tax % to add to the final price</td>
                    <td>float, Default 0.0</td>
                </tr>
                <tr>
                    <td>startTs</td>
                    <td>Start output from this time. At most ~60 days</td>
                    <td>iso date, Defaults to now()</td>
                </tr>
                <tr>
                    <td>evaluation</td>
                    <td>Switches to evaluation mode. All values will be generated by the model, instead of only future values.<br>Useful to evaluate model performance.</td>
                    <td>bool, default false</td>
                </tr>
            </tbody>
        </table>
        </body>
        </html>
    """


import predictor.model.pricepredictor as pp

class Prices:
    predictor : pp.PricePredictor = pp.PricePredictor(testdata=False)
    last_weather_update : datetime.datetime = datetime.datetime(1980, 1, 1)
    last_price_update : datetime.datetime = datetime.datetime(1980, 1, 1)

    cachedprices : Dict[datetime.datetime, float]
    cachedeval : Dict[datetime.datetime, float]

    def __init__(self):
        pass

    def prices(self):
        hours = int(request.args.get("hours", "-1"))
        fixedPirce = float(request.args.get("fixedPrice", "0.0"))
        taxPercent = float(request.args.get("taxPercent", "0.0"))
        evaluation = request.args.get("evaluation", "false") == "true"
        startTs = request.args.get("startTs", None)
        tzgerman = pytz.timezone("Europe/Berlin")

        if startTs is None:
            startTs = datetime.datetime.now(tz=tzgerman)
            startTs = startTs.replace(minute=0, second=0, microsecond=0)
        else:
            startTs = dateutil.parser.isoparse(startTs)
            if startTs.tzinfo is None:
                startTs = startTs.astimezone(tzgerman)
        
        endTs = datetime.datetime(2999, 1, 1, tzinfo=tzgerman)
        if hours >= 0:
            endTs = startTs + datetime.timedelta(hours=hours)


        currts = datetime.datetime.now()
        weather_age = currts - self.last_weather_update
        price_age = currts - self.last_price_update
        retrain = False
        if weather_age.seconds > 60 * 60 * 8: # update weather every 4 hours
            self.predictor.refresh_forecasts()
            self.last_weather_update = currts
            retrain = True
        if price_age.seconds > 60 * 15: # update prices every 15 mins
            self.predictor.refresh_prices()
            self.last_price_update = currts
            retrain = True
        if retrain:
            self.predictor.train()
            self.cachedprices = self.predictor.predict()
            self.cachedeval = self.predictor.predict(estimateAll=True)
        
        prices = self.cachedprices if evaluation is False else self.cachedeval

        result = []
        for dt in sorted(prices.keys()):
            if dt < startTs:
                continue
            if dt > endTs:
                continue
            formatted = dt.astimezone(tzgerman).isoformat()
            price = prices[dt] / 10.0 # to ct/kWh
            total = (price + fixedPirce) * (1 + taxPercent / 100.0)
            result.append(
                {
                    "startsAt": formatted,
                    "total": round(total, 2)
                }
            )

        return json.dumps(
            {
                "prices": result
            }
        )


prices : Prices = Prices()
app.add_url_rule("/prices", view_func=prices.prices)

    
