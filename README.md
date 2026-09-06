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
- Sweden (SE1-SE4)
- Denmark (DK1-DK2)
- Spain
- Portugal
- Others can be added relatively easily, if there is interest


## Lookout
- Maybe package it directly as a Home Assistant Add-on

## The Model
We sample multiple locations distributed across each region. We fetch [Weather data from Open-Meteo.com](https://open-meteo.com/) for those locations for the past n days (default n=180).
This serves as the main data source.

Price data is provided under CC BY 4.0 by smartd.de, retrieved via [api.energy-charts.info](https://api.energy-charts.info/) and [ENTSO-E transparency platform](https://transparency.entsoe.eu/).

Grid load data is provided by [ENTSO-E transparency platform](https://transparency.entsoe.eu/).

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

Other:
- Entso-E load forecast (optional, but highly recommended, especially for DE and AT)
- Natural gas day-ahead-price, forward filled (select regions only)
- Stage-1 price forecasts of all regions (stage-2 models only, see below)

Output:
- Electricity price

## How it works
The model uses **LightGBM gradient boosting** to predict electricity prices. LightGBM automatically learns non-linear relationships and feature interactions, making it well-suited for electricity price prediction where factors like low wind+solar can cause price spikes due to merit order pricing.

The prediction runs in **two stages**:
1. **Stage 1**: one base model per region, trained on that region's own features (weather, prices, load, gas).
2. **Stage 2**: one model per region, trained on the same features plus the **stage-1 forecasts of all regions** as extra input features.

The cross-region forecasts let the model pick up on price coupling between neighboring markets (e.g. a wind lull in one region pushing up prices in connected ones), which a single-region model cannot see.

## Model performance
For performance testing, see `predictor/performance_testing.py`. It runs a rolling backtest of the full 2-stage stack: stage-1 models for all regions, then stage-2 models using the stage-1 forecasts of all regions as cross-features (set `TWO_STAGE = False` to fall back to the original single-stage evaluation).

Remarks:
- Tests were run in 2026, with data from 2025-09-01 to 2026-09-01. The model is tuned for 15 minute pricing.
- The model uses a 180-day rolling training window
- Tests were done with historical weather data. If the weather forecast is wrong, performance might be slightly worse in practice

Results (1/2/3-day ahead prediction):
| Region | 1d RMSE | 1d MAE | 2d RMSE | 2d MAE | 3d RMSE | 3d MAE |
|--------|---------|--------|---------|--------|---------|--------|
| DE     | 2.93    | 1.73   | 3.15    | 1.88   | 3.17    | 1.91   |
| AT     | 3.11    | 1.98   | 3.35    | 2.17   | 3.42    | 2.24   |
| BE     | 3.15    | 1.84   | 3.39    | 2.0    | 3.43    | 2.02   |
| NL     | 3.04    | 1.75   | 3.21    | 1.87   | 3.25    | 1.91   |
| SE1    | 2.7     | 1.68   | 3.01    | 1.91   | 3.11    | 1.97   |
| SE2    | 2.7     | 1.65   | 3.07    | 1.91   | 3.16    | 1.97   |
| SE3    | 2.87    | 2.02   | 3.11    | 2.23   | 3.12    | 2.26   |
| SE4    | 3.31    | 2.34   | 3.52    | 2.55   | 3.55    | 2.58   |
| DK1    | 2.79    | 1.79   | 2.99    | 1.92   | 3.06    | 1.97   |
| DK2    | 3.06    | 1.94   | 3.25    | 2.11   | 3.27    | 2.13   |
| ES     | 2.35    | 1.7    | 2.66    | 1.96   | 2.79    | 2.06   |
| PT     | 2.47    | 1.8    | 2.82    | 2.12   | 2.94    | 2.22   |

Some observations:
- At night, predictions are typically within 0.5 ct/kWh
- Morning/Evening peaks are typically within 1-1.5 ct/kWh
- Extreme peaks due to "Dunkelflaute" are correctly detected, but estimation of the exact price is a challenge (e.g. the model might predict 75ct while reality is 60ct or vice versa)
- High PV noons are usually correctly detected with good accuracy


### Current forecast (DE)
![image](https://epexpredictor.batzill.com/eval_plot?region=DE&transparent=false&width=1024&height=512)


Feel free to generate your own plot for other time ranges or regions [here](https://epexpredictor.batzill.com/docs#/default/generate_evaluation_plot_eval_plot_get).

Note that the eval plot trains the full 2-stage model stack (all regions) for the requested historical window, which is CPU-intensive. Results are therefore cached for 30 minutes per (region, range) - after that, the last result is served while a refresh runs in the background.


# Public API
You can find a freely accessible installment of this software [here](https://epexpredictor.batzill.com/).
Get a glimpse of the current prediction [here](https://epexpredictor.batzill.com/prices).

There are no guarantees given whatsoever - it might work for you or not.
I might stop or block this service at any time. Fair use is expected!

# Self Hosting
You can easily self-host this software. For easy deployment, check out the docker compose file.
You will probably want to register with Entso-E and request an API key.
Without Entso-E API access
- some parameters are missing and the model will perform significantly worse, especially for DE and AT
- Some regions will not be available (e.g. SE1-4)

# Home Assistant integration
At some point, I might create a HA addon to run everything locally.
For now, you have to either use my server, or run it yourself.

Note: Home Assistant only supports a limited amount of data in state attributes. Therefore, we use the "short format" output, and limit the time to 120 hours.
If you need more, you will have to be more creative.
Personally, I provide the data as a HA "service" (now "action") using pyscript, and then call this service to work with the data.



### Configuration:
```yaml
# Make sure you change the parameters region, surcharge and taxPercent according to your electricity plan
sensor:
  - platform: rest
    resource: "https://epexpredictor.batzill.com/prices_short?region=DE&surcharge=13.70084&taxPercent=19&unit=EUR_PER_KWH&hours=120"
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
  #  resource: "https://epexpredictor.batzill.com/prices_short?region=DE&surcharge=13.70084&taxPercent=19&evaluation=true&unit=EUR_PER_KWH&hours=120"
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

[evcc](https://evcc.io/) is an open-source EV charging controller that can optimize charging based on electricity prices.
It now has native support for EpexPredictor, see [docs](https://docs.evcc.io/docs/tariffs#epex-predictor-predicted-epex-spot-prices)