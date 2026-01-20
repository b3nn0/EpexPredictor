# EPEX day-ahead price prediction

This is a simple statistical model to predict EPEX day-ahead prices based on various parameters.
It works to a reasonably good degree. Better than many of the commercial solutions.
This repository includes
- The self-training prediction model itself
- A simple FastAPI app to get a REST API up
- A Docker compose file to have it running wherever

Supported Countries:
- Germany (default)
- Austria
- Belgium
- Netherlands
- Others can be added relatively easily, if there is interest


## Lookout
- Maybe package it directly as a Home Assistant Add-on

## The Model
We sample multiple locations distributed across each country. We fetch [Weather data from Open-Meteo.com](https://open-meteo.com/) for those locations for the past n days (default n=120).
This serves as the main data source.
Price data is provided under CC BY 4.0 by smartd.de, retrieved via https://api.energy-charts.info/.

### Features

Weather features (per sample location):
- Wind speed at 80m
- Temperature at 2m
- Global tilted irradiance (solar)
- Air pressure at mean sea level
- Relative humidity

Time features:
- Azimuth of the sun as indicator of time of day
- Elevation of the sun
- Day of the week (Monday to Saturday)
- Holiday/Sunday indicator (regional holidays weighted by fraction of regions, e.g. 0.5 if half the regions have the holiday)
- Sunrise influence: how many minutes between sunrise and the current time slot
- Sunset influence: how many minutes between sunset and the current time slot
- Morning influence: how many minutes between now and 6AM today
- Evening influence: how many minutes between now and 6PM today

Output:
- Electricity price

## How it works
The model uses **LightGBM gradient boosting** to predict electricity prices. LightGBM automatically learns non-linear relationships and feature interactions, making it well-suited for electricity price prediction where factors like low wind+solar can cause price spikes due to merit order pricing.

## Model performance
For performance testing, see `predictor/performance_testing.py`.

Remarks:
- Tests were run in early 2026, with data from 2025-01-01 to 2026-01-11. The model is tuned for 15 minute pricing. Since data before 2025-10-01 were using hourly pricing, actual performance might be slightly better
- The model uses a 120-day rolling training window
- Tests were done with historical weather data. If the weather forecast is wrong, performance might be slightly worse in practice

Results (1-day ahead prediction):
| Country | MAE (ct/kWh) | RMSE (ct/kWh) |
|---------|--------------|---------------|
| DE | 1.70 | 2.65 |
| AT | 1.90 | 2.96 |
| BE | 1.78 | 2.61 |
| NL | 1.79 | 2.72 |

Some observations:
- At night, predictions are typically within 0.5 ct/kWh
- Morning/Evening peaks are typically within 1-1.5 ct/kWh
- Extreme peaks due to "Dunkelflaute" are correctly detected, but estimation of the exact price is a challenge (e.g. the model might predict 75ct while reality is 60ct or vice versa)
- High PV noons are usually correctly detected with good accuracy


### Current forecast
![image](https://epexpredictor.batzill.com/eval_plot?country=DE&transparent=false&width=1024&height=512)


Feel free to generate your own performance evaluation graphs for any time range [here](https://epexpredictor.batzill.com/docs#/default/generate_evaluation_plot_eval_plot_get).


# Public API
You can find a freely accessible installment of this software [here](https://epexpredictor.batzill.com/).
Get a glimpse of the current prediction [here](https://epexpredictor.batzill.com/prices).

There are no guarantees given whatsoever - it might work for you or not.
I might stop or block this service at any time. Fair use is expected!

# Home Assistant integration
At some point, I might create a HA addon to run everything locally.
For now, you have to either use my server, or run it yourself.

Note: Home Assistant only supports a limited amount of data in state attributes. Therefore, we use the "short format" output, and limit the time to 120 hours.
If you need more, you will have to be more creative.
Personally, I provide the data as a HA "service" (now "action") using pyscript, and then call this service to work with the data.



### Configuration:
```yaml
# Make sure you change the parameters fixedPrice and taxPercent according to your electricity plan
sensor:
  - platform: rest
    resource: "https://epexpredictor.batzill.com/prices_short?fixedPrice=13.70084&taxPercent=19&unit=EUR_PER_KWH&hours=120"
    method: GET
    unique_id: epex_price_prediction
    name: "EPEX Price Prediction"
    unit_of_measurement: €/kWh
    value_template: "{{ value_json.t[0] }}"
    json_attributes:
      - s
      - t

  # If you want to evaluate performance in real time, you can add another sensor like this
  # and plot it in the same diagram as the actual prediction sensor

  #- platform: rest
  #  resource: "https://epexpredictor.batzill.com/prices_short?fixedPrice=13.70084&taxPercent=19&evaluation=true&unit=EUR_PER_KWH&hours=120"
  #  method: GET
  #  unique_id: epex_price_prediction_evaluation
  #  name: "EPEX Price Prediction Evaluation"
  #  unit_of_measurement: €/kWh
  #  value_template: "{{ value_json.t[0] }}"
  #  json_attributes:
  #    - s
  #    - t
```

### Display, e.g. via Plotly Graph Card:
```yaml
type: custom:plotly-graph
time_offset: 26h
layout:
  yaxis9:
    fixedrange: true
    visible: false
    minallowed: 0
    maxallowed: 1
entities:
  - entity: sensor.epex_price_prediction
    name: EPEX Price Prediction
    unit_of_measurement: ct/kWh
    texttemplate: "%{y:.0f}"
    mode: lines+text
    textposition: top right
    filters:
      - fn: |-
          ({xs, ys, meta}) => {
            return {
              xs: xs.concat(meta.s.map(s => s*1000)),
              ys: ys.concat(meta.t).map(t => +t*100)
            }
          }
  - entity: ""
    name: Now
    yaxis: y9
    showlegend: false
    line:
      width: 1
      dash: dot
      color: orange
    x: $ex [Date.now(), Date.now()]
    "y":
      - 0
      - 1
hours_to_show: 30
refresh_interval: 10
```

# evcc integration

[evcc](https://evcc.io/) is an open-source EV charging controller that can optimize charging based on electricity prices. This EPEX predictor integrates seamlessly with evcc to enable smart charging based on predicted electricity prices.

### Configuration

Add the following to your evcc configuration file (`evcc.yaml`):

```yaml
# Make sure you change the parameters fixedPrice and taxPercent according to your electricity plan
tariffs:
  currency: EUR
  grid:
    type: custom
    forecast:
      source: http
      uri: https://epexpredictor.batzill.com/prices?country=DE&fixedPrice=13.15&taxPercent=19&unit=EUR_PER_KWH&timezone=UTC
      jq: '[.prices[] | { start: .startsAt, "end": (.startsAt | strptime("%Y-%m-%dT%H:%M:%SZ") | mktime + 900 | strftime("%Y-%m-%dT%H:%M:%SZ")), "value": .total}] | tostring'
```

