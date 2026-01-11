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
We sample multiple locations distributed across each country, including offshore North Sea locations for coastal countries (DE, NL, BE) to capture offshore wind farm production. We fetch hourly [Weather data from Open-Meteo.com](https://open-meteo.com/) for those locations for the past n days (default n=90).
This serves as the main data source.
Price data is provided under CC BY 4.0 by smartd.de, retrieved via https://api.energy-charts.info/.



Parameters:

- Wind for each location
- Temperature for each location
- Expected solar irradiance for each location
- Hour of day
- Day of the week from monday to saturday
- Whether it is a Holiday/Sunday (regional holidays are considered by the number of regions they apply to, e.g. 0.5 if a holiday applies to half the regions of a country)
- A measure of sunrise influence - how many minutes between sunrise and now, capped at 3 hours: $\min(180, |t_{now} - t_{sunrise}|)$ and vice versa for
- A measure of sunset influence, same formula

Output:
- Electricity price

## How it works
- First, we use linear regression to determine the importance of each training parameter.
- This alone is not enough, because electricity prices are not linear.
E.g. low wind&solar leads to gas power plants being turned on, and due to merit order pricing, electricity prices explode.
- Therefore, we then multiply each parameter with its weight (linreg coefficients) to get a "normalized" data set.
- In the next step, we use a KNN (k=7) approach to find hours in the past with similar properties and use that to determine the final price.

## Model performance
For performance testing, see `predictor/performance_testing.py`.
Remarks:
- Tests were run in early 2026, with data from 2025-01-01 to 2026-01-11. The model is tuned for 15 minute pricing. Since data before 2025-10-01 were using hourly pricing, actual performance might be slightly better
- Tests were done with historical weather data. If the weather forecast is wrong, performance might be slightly worse in practice

Results:\
DE: Mean squared error ~8.37 ct/kWh, mean absolute error ~1.85 ct/kWh\
AT: Mean squared error ~10.45 ct/kWh, mean absolute error ~2.11 ct/kWh\
BE: Mean squared error ~9.43 ct/kWh, mean absolute error ~2.04 ct/kWh\
NL: Mean squared error ~9.53 ct/kWh, mean absolute error ~2.0 ct/kWh

Some observations:
- At night, predictions are typically within 0.5ct/kWh
- Morning/Evening peaks are typically within 1-1.5ct/kWh
- Extreme peaks due to "Dunkelflaute" are correctly detected, but estimation of the exact price is a challenge. E.g.
the model might predict 75ct, while in reality it's only 60ct or vice versa
- High PV noons are usually correctly detected with good accuracy.
- Offshore wind data significantly improves prediction accuracy for coastal countries (BE, NL, DE)

This graph compares the actual prices to the ones returned by the model for a random two week time period in early 2025.

Note that this was created for a time range in the past with historic weather data, rather than forecasted weather data,
so actual performance might be a bit worse if the weather forecast is not correct.

```mermaid
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
    line [3.3,3.3,3.3,3.3,3.4,3.4,3.4,3.4,3.7,3.7,3.7,3.7,5.5,5.5,5.5,5.5,7.0,7.0,7.0,7.0,6.8,6.8,6.8,6.8,5.3,5.3,5.3,5.3,2.9,2.9,2.9,2.9,0.2,0.2,0.2,0.2,-0.0,-0.0,-0.0,-0.0,-0.1,-0.1,-0.1,-0.1,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.1,-0.1,-0.1,-0.1,-0.0,-0.0,-0.0,-0.0,0.6,0.6,0.6,0.6,7.4,7.4,7.4,7.4,11.2,11.2,11.2,11.2,13.7,13.7,13.7,13.7,14.6,14.6,14.6,14.6,13.0,13.0,13.0,13.0,10.8,10.8,10.8,10.8,10.7,10.7,10.7,10.7,10.0,10.0,10.0,10.0,9.9,9.9,9.9,9.9,9.9,9.9,9.9,9.9,10.0,10.0,10.0,10.0,9.8,9.8,9.8,9.8,9.5,9.5,9.5,9.5,8.5,8.5,8.5,8.5,5.5,5.5,5.5,5.5,0.1,0.1,0.1,0.1,-0.0,-0.0,-0.0,-0.0,-0.1,-0.1,-0.1,-0.1,-0.3,-0.3,-0.3,-0.3,-1.3,-1.3,-1.3,-1.3,-0.8,-0.8,-0.8,-0.8,-0.3,-0.3,-0.3,-0.3,-0.0,-0.0,-0.0,-0.0,4.1,4.1,4.1,4.1,8.9,8.9,8.9,8.9,13.1,13.1,13.1,13.1,19.2,19.2,19.2,19.2,19.2,19.2,19.2,19.2,13.3,13.3,13.3,13.3,10.2,10.2,10.2,10.2,9.3,9.3,9.3,9.3,8.5,8.5,8.5,8.5,7.7,7.7,7.7,7.7,7.4,7.4,7.4,7.4,7.3,7.3,7.3,7.3,6.6,6.6,6.6,6.6,3.6,3.6,3.6,3.6,1.3,1.3,1.3,1.3,0.2,0.2,0.2,0.2,-0.0,-0.0,-0.0,-0.0,-0.1,-0.1,-0.1,-0.1,-0.2,-0.2,-0.2,-0.2,-0.5,-0.5,-0.5,-0.5,-1.3,-1.3,-1.3,-1.3,-2.0,-2.0,-2.0,-2.0,-1.6,-1.6,-1.6,-1.6,-0.6,-0.6,-0.6,-0.6,-0.1,-0.1,-0.1,-0.1,1.9,1.9,1.9,1.9,7.9,7.9,7.9,7.9,9.4,9.4,9.4,9.4,10.7,10.7,10.7,10.7,10.6,10.6,10.6,10.6,9.0,9.0,9.0,9.0,8.2,8.2,8.2,8.2,7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.8,7.8,7.8,7.8,8.3,8.3,8.3,8.3,9.2,9.2,9.2,9.2,10.8,10.8,10.8,10.8,11.8,11.8,11.8,11.8,10.2,10.2,10.2,10.2,8.6,8.6,8.6,8.6,5.7,5.7,5.7,5.7,3.3,3.3,3.3,3.3,2.0,2.0,2.0,2.0,1.3,1.3,1.3,1.3,1.1,1.1,1.1,1.1,2.0,2.0,2.0,2.0,5.6,5.6,5.6,5.6,8.3,8.3,8.3,8.3,9.5,9.5,9.5,9.5,10.9,10.9,10.9,10.9,19.6,19.6,19.6,19.6,24.5,24.5,24.5,24.5,14.2,14.2,14.2,14.2,11.0,11.0,11.0,11.0,11.1,11.1,11.1,11.1,11.0,11.0,11.0,11.0,10.2,10.2,10.2,10.2,9.9,9.9,9.9,9.9,9.5,9.5,9.5,9.5,10.8,10.8,10.8,10.8,13.1,13.1,13.1,13.1,11.7,11.7,11.7,11.7,8.6,8.6,8.6,8.6,5.9,5.9,5.9,5.9,0.2,0.2,0.2,0.2,-0.0,-0.0,-0.0,-0.0,-0.1,-0.1,-0.1,-0.1,-0.3,-0.3,-0.3,-0.3,-0.4,-0.4,-0.4,-0.4,-0.1,-0.1,-0.1,-0.1,0.0,0.0,0.0,0.0,6.6,6.6,6.6,6.6,8.8,8.8,8.8,8.8,10.9,10.9,10.9,10.9,13.3,13.3,13.3,13.3,11.2,11.2,11.2,11.2,10.6,10.6,10.6,10.6,9.5,9.5,9.5,9.5,7.5,7.5,7.5,7.5,6.7,6.7,6.7,6.7,6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.8,6.8,6.8,6.8,8.0,8.0,8.0,8.0,10.0,10.0,10.0,10.0,11.4,11.4,11.4,11.4,11.2,11.2,11.2,11.2,7.9,7.9,7.9,7.9,7.2,7.2,7.2,7.2,4.8,4.8,4.8,4.8,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,1.8,1.8,1.8,1.8,4.3,4.3,4.3,4.3,7.2,7.2,7.2,7.2,9.1,9.1,9.1,9.1,10.9,10.9,10.9,10.9,14.6,14.6,14.6,14.6,19.3,19.3,19.3,19.3,13.9,13.9,13.9,13.9,11.2,11.2,11.2,11.2,9.4,9.4,9.4,9.4,9.8,9.8,9.8,9.8,9.3,9.3,9.3,9.3,8.7]
    line [5.8,6.6,6.2,6.2,6.0,6.0,6.0,6.0,7.5,7.5,7.5,7.5,7.3,6.6,6.0,6.0,6.4,6.4,6.4,6.4,6.4,7.2,8.0,9.5,5.4,6.2,10.0,10.5,2.0,0.9,4.2,6.1,0.1,0.1,0.0,0.6,0.0,-0.0,0.1,0.1,-0.0,-0.0,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.3,-0.3,-0.2,-0.2,-0.3,-0.2,0.3,-0.1,-0.1,-0.1,5.8,4.8,2.6,1.8,10.5,10.5,10.5,10.5,13.6,13.6,13.6,12.1,14.1,14.1,13.0,13.0,11.5,11.5,11.5,11.5,11.5,11.5,11.5,11.5,10.7,11.0,10.5,9.5,10.0,10.0,10.0,9.7,9.0,9.0,9.0,8.7,8.6,8.6,8.6,8.6,8.5,8.6,8.5,8.5,8.6,8.6,8.7,8.7,9.0,9.3,9.6,9.6,9.0,9.0,8.0,7.8,8.1,6.4,6.7,8.7,5.2,6.6,8.4,9.1,1.9,1.9,1.5,3.9,-1.4,-2.0,-2.0,-1.3,-2.6,-2.6,-2.0,-1.1,-2.0,-2.0,-0.8,-0.3,-0.3,-0.2,-0.3,-0.2,-0.2,-0.3,-0.3,-0.3,-0.5,-0.4,-0.5,-0.5,0.4,0.5,0.8,0.8,6.6,6.6,5.7,4.7,10.2,10.4,10.4,10.4,13.4,13.2,13.6,13.6,17.2,17.2,17.2,15.7,11.9,11.9,11.4,11.4,10.1,10.2,10.3,10.4,9.3,9.3,9.3,9.5,9.8,9.5,8.4,8.3,7.7,7.9,7.9,7.9,7.5,7.5,7.5,7.5,7.2,7.2,7.2,6.8,6.8,6.8,6.8,6.8,3.5,2.5,2.1,2.1,0.6,0.6,-0.3,-0.4,-0.4,0.1,2.3,4.9,0.8,2.3,3.2,4.6,0.3,0.7,1.0,0.9,-0.7,-2.4,-4.6,-3.0,-5.2,-5.0,-4.2,-10.3,-17.1,-7.8,-4.4,-2.2,-11.8,-11.8,-3.2,-0.8,-4.1,-1.0,-4.0,-1.0,-0.7,-0.9,-1.0,-2.4,-2.8,-2.4,-1.3,-1.5,3.2,0.6,0.6,0.1,7.9,7.9,7.7,7.7,10.9,10.9,10.9,10.9,11.9,11.9,11.9,11.1,10.8,10.8,10.8,10.8,10.7,10.7,10.7,10.7,8.8,9.5,9.5,9.5,9.6,9.6,9.6,9.6,9.0,9.0,9.0,9.0,8.7,8.3,8.3,8.3,8.2,8.2,8.2,8.2,9.0,8.5,8.5,8.5,9.5,9.5,9.5,9.5,11.4,11.4,11.4,11.4,11.9,13.3,12.8,13.0,12.9,11.3,11.2,11.2,8.0,8.2,7.2,8.8,3.1,3.5,2.9,3.8,2.2,3.1,3.1,3.1,2.9,3.1,3.1,3.8,3.8,4.7,4.6,4.4,3.5,4.4,3.5,3.7,4.4,3.7,4.7,3.8,7.3,7.0,6.7,7.0,6.4,6.2,6.9,7.4,10.4,10.4,10.5,10.2,13.8,13.8,13.8,13.8,16.8,16.8,16.8,16.8,13.8,15.1,15.5,15.1,14.3,13.3,12.9,12.9,10.5,10.7,10.5,10.2,9.6,9.6,9.6,9.6,9.1,9.1,9.4,9.4,8.9,8.9,8.9,9.2,9.3,8.8,8.4,8.4,8.7,8.7,9.0,9.0,9.6,9.6,9.6,9.6,12.3,12.3,12.3,12.0,13.1,11.8,11.2,11.2,8.6,9.7,10.8,11.1,7.1,7.5,6.1,6.6,4.3,3.3,3.1,2.0,0.1,0.1,0.3,0.2,-0.6,-1.0,-0.8,0.0,0.0,0.0,-0.7,-1.0,-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,7.2,6.1,4.8,3.8,10.5,10.5,10.5,9.4,13.7,13.7,13.7,13.7,16.5,16.5,16.5,16.3,13.8,13.8,14.1,14.1,12.1,12.1,12.0,12.0,10.6,10.6,10.6,10.3,9.7,9.4,9.4,9.4,8.7,8.7,8.7,8.7,8.4,8.4,8.4,8.4,8.3,8.3,8.3,8.3,8.4,8.4,8.4,8.4,9.1,9.1,9.1,9.1,10.1,10.0,10.0,8.5,9.2,9.1,10.5,11.0,6.9,8.9,9.7,10.3,5.3,5.5,6.3,8.2,7.4,7.4,3.6,4.4,5.4,7.4,7.3,7.0,5.4,5.3,5.1,4.7,4.7,3.9,4.5,3.1,4.6,2.3,0.7,3.7,4.2,3.9,2.8,1.5,1.6,0.9,1.3,4.1,6.4,6.4,6.4,7.5,10.2,10.2,10.4,10.4,13.1,12.7,12.7,12.5,15.1,14.5,13.8,14.0,13.0,13.0,13.0,13.0,12.3,12.3,12.2,11.9,10.8,10.6,10.6,10.5,10.7,10.1,9.4,9.5,8.9,9.1,9.0,9.0,8.9]
```


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

