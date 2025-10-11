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


## Lookout
- Maybe package it directly as a Home Assistant Add-on

## The Model
We sample a few sample points all over the country and fetch hourly [Weather data from Open-Meteo.com](https://open-meteo.com/) for those for the past n days (default n=90).
This serves as the main data source.



Parameters:

- Wind for each location
- Temperature for each location
- Expected solar irradiance for each location
- Hour of day
- Day of the week from monday to saturday
- Whether it is a Holiday/Sunday

Output:
- Electricity price

## How it works
- First, we create a simple multi-linear-regression to get an idea how important each of
the training parameters is.
- This alone is not enough, because electricity prices are not linear.
E.g. low wind&solar leads to gas power plants being turned on, and due to merit order pricing, electricity prices explode.
- Therefore, we then multiply each parameter with its weight (LinReg factor) to get a "normalized" data set.
- Since we don't care if prices are low due to high wind or high solar (or higher temperature, thus less heating), we now simply sum up all those normalized numeric values
- In the next step, we use a KNN (k=3) approach to find hours in the past with similar properties and use that to determine the final price.

## Model performance
For performance testing, we used historical weather data with a 90%/10% split for a training/testing data set. See `predictor/model/performance_testing.py`.

Results:\
DE: Mean squared error ~4.02 ct/kWh, mean absolute error ~1.42 ct/kWh\
AT: Mean squared error ~5.20 ct/kWh, mean absolute error ~1.66 ct/kWh

Some observations:
- At night, predictions are usually within 1-2ct/kWh
- Morning/Evening peaks are usually within 3-4ct/kWh
- Extreme peaks due to "Dunkelflaute" are correctly detected, but estimation of the exact price is a challange. E.g.
the model might predict 75ct, while in reality it's only 60ct or vice versa
- High PV noons are usually correctly detected. Sometimes it will return 3ct instead of -1ct, but the ballpark is usually correct.

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
    line [9.6,13.9,15.1,10.8,7.9,5.0,0.4,0.0,0.0,1.3,7.1,11.7,14.9,16.5,14.9,12.6,10.7,10.1,9.1,9.0,9.0,9.1,9.2,9.9,11.5,13.6,14.8,12.2,8.3,6.5,3.2,1.3,0.8,3.7,9.0,12.7,15.0,21.7,17.9,14.9,13.2,11.9,11.2,12.5,11.4,11.7,11.7,11.8,12.1,12.8,12.1,9.8,7.3,1.8,0.2,-0.0,-0.0,0.0,5.6,10.1,14.6,17.4,14.0,11.1,10.0,9.9,9.8,9.5,9.5,9.4,9.7,9.9,10.2,10.0,9.3,6.5,2.9,1.0,0.0,-0.2,-0.2,0.0,1.7,9.1,12.6,13.5,13.7,13.0,12.3,11.3,10.7,9.7,9.5,9.7,9.7,9.5,10.6,14.8,16.0,13.4,9.9,8.4,7.8,7.6,7.7,8.4,9.7,12.7,14.8,17.5,16.0,14.2,12.7,10.7,9.8,10.4,10.2,10.0,10.0,10.3,11.1,14.8,15.3,14.6,12.9,10.2,9.7,9.6,9.8,10.2,11.3,13.3,17.2,20.5,18.1,14.9,13.3,12.1,10.7,10.2,10.0,9.8,9.7,10.2,11.0,14.5,16.2,14.8,12.5,10.3,10.0,9.8,10.0,10.2,11.8,14.2,15.7,20.0,17.9,14.8,13.2,12.0,11.0,10.9,10.2,9.9,9.7,9.8,10.6,13.3,16.0,15.8,14.8,12.4,11.1,11.0,10.8,10.8,11.0,12.1,14.3,16.6,17.3,14.9,13.6,12.3,11.2,10.9,10.6,10.6,10.4,10.5,11.4,14.3,15.5,15.5,13.2,11.2,10.6,10.4,10.5,10.5,10.9,12.5,14.6,15.5,15.1,13.0,11.1,11.4,10.5,11.2,10.3,9.6,9.0,8.7,9.1,9.7,9.9,9.9,8.6,7.2,6.7,5.7,4.8,5.5,7.1,8.4,12.2,13.0,13.3,12.1,10.8,10.5,9.5,9.0,8.6,8.0,7.8,7.9,8.1,8.5,8.2,7.8,5.7,2.4,0.3,0.1,0.0,0.3,3.1,7.1,11.7,13.1,14.0,12.5,10.8,10.1,8.0,7.8,7.9,7.9,8.1,8.6,10.3,13.5,15.3,13.5,10.1,6.8,5.1,3.1,2.8,5.0,7.0,9.9,14.3,17.7,17.0,14.2,11.9,10.8,9.4,9.2,9.0,9.2,8.9,9.5,11.1,13.9,13.7,10.7,7.2,2.5,0.1,-0.1,-0.0,0.6,5.0,9.6,14.1,17.1,16.7,14.2,12.3,11.6,10.3,10.7,10.2,10.1,10.2,10.7,12.3,15.8,14.8,10.8,8.0,2.7,-0.0,-0.1,-0.0,0.1,6.7,11.0,14.8,24.8,22.0,17.7,13.8,12.4,11.1,11.7,10.3,10.0,10.0,10.3]
    line [10.6,13.8,14.7,12.7,8.4,5.7,0.2,-0.0,0.0,1.0,7.5,11.0,14.4,17.7,16.4,15.0,12.6,11.1,10.5,9.4,8.4,8.6,9.1,10.0,12.1,14.0,15.0,12.8,9.9,7.3,3.8,3.3,2.6,2.4,7.4,11.2,14.2,19.4,20.1,14.1,13.3,12.3,11.1,11.5,10.6,10.4,10.3,10.8,10.2,11.9,10.5,9.9,7.8,1.5,1.2,2.2,0.0,0.7,5.2,10.1,14.1,15.0,14.0,10.9,9.7,10.8,9.1,7.7,6.5,9.1,6.0,9.0,9.2,9.4,10.0,7.7,1.6,0.8,0.2,0.3,-0.5,-0.8,3.3,8.1,12.3,13.9,14.1,13.2,12.4,12.0,11.1,10.5,10.1,9.9,9.7,9.9,10.8,14.8,15.8,14.4,13.6,9.8,7.4,5.3,4.4,7.1,7.9,11.4,15.1,17.9,16.9,15.5,12.7,11.9,11.2,10.3,10.0,9.8,9.8,9.9,10.7,14.9,16.0,16.2,14.7,10.7,9.2,10.0,8.7,8.3,9.4,12.2,15.1,19.0,20.2,14.6,12.3,12.2,11.0,10.6,10.5,9.8,10.0,10.1,10.9,15.1,15.4,14.5,13.2,11.7,10.6,10.0,10.2,10.5,11.2,12.9,14.9,17.7,18.3,15.1,13.1,11.9,11.0,11.2,10.8,9.8,10.1,10.0,11.0,14.0,15.2,15.5,12.7,10.9,10.6,10.3,8.9,10.6,11.2,12.9,14.9,17.7,16.4,14.3,13.1,12.1,10.5,10.1,10.8,10.1,10.3,10.4,11.5,14.3,15.9,15.4,12.6,10.2,10.6,9.9,9.4,10.6,11.2,12.9,14.9,19.8,16.2,14.4,11.4,11.1,9.7,10.6,9.5,9.8,9.6,9.7,10.2,9.7,9.9,10.0,7.8,7.3,5.0,5.3,1.9,7.7,8.6,10.4,14.1,15.0,14.2,13.0,11.5,10.7,9.0,7.7,7.0,9.1,8.9,9.0,9.2,9.4,9.3,8.2,3.5,3.1,0.9,1.4,0.3,0.6,3.3,8.1,12.3,13.5,14.1,10.7,11.9,11.1,10.0,9.0,7.7,7.6,7.6,7.9,9.2,13.1,13.7,13.6,9.1,3.9,1.7,1.0,1.0,2.1,6.3,8.8,13.1,17.9,16.9,14.2,12.7,11.3,10.6,10.5,9.8,9.5,9.9,9.9,11.0,13.9,14.9,12.7,8.1,4.8,0.0,1.7,0.4,2.3,6.0,10.5,14.3,17.1,16.7,14.5,12.3,11.5,10.4,9.7,9.4,9.5,9.7,9.0,11.3,14.4,14.6,10.2,6.6,2.5,0.2,-0.0,0.0,1.0,7.2,10.8,14.3,18.9,18.3,16.0,12.5,11.5,11.1,10.4,9.4,10.3,10.1,10.0]
```


# Public API
You can find a freely accessible installment of this software [here](https://epexpredictor.batzill.com/).
Get a glimpse of the current prediction [here](https://epexpredictor.batzill.com/prices).

There are no guarantees given whatsoever - it might work for you or not.
I might stop or block this service at any time. Fair use is expected!

# Home Assistant integration
At some point, I might create a HA addon to run everything locally.
For now, you have to either use my server, or run it yourself.

Note: Home Assistant only supports a limited amount of data in state attributes. Therefore, this graph will be limited to 72 hours.
If you need more, you will have to be more creative.
Personally, I provide the data as a HA "service" (now "action") using pyscript, and then call this service to work with the data.



### Configuration:
```yaml
# Make sure you change the parameters fixedPrice and taxPercent according to your electricity plan
sensor:
  - platform: rest
    resource: "https://epexpredictor.batzill.com/prices?country=DE&fixedPrice=13.15&taxPercent=19&hours=72"
    method: GET
    unique_id: epex_price_prediction
    name: "EPEX Price Prediction"
    scan_interval: 500
    unit_of_measurement: ct/kWh
    value_template: "{{ value_json.prices[0].total }}"
    json_attributes:
      - prices

  # If you want to evaluate performance in real time, you can add another sensor like this
  # and plot it in the same diagram as the actual prediction sensor

  #- platform: rest
  #  resource: "https://epexpredictor.batzill.com/prices?country=DE&fixedPrice=13.15&taxPercent=19&#evaluation=true&hours=72"
  #  method: GET
  #  unique_id: epex_price_prediction_evaluation
  #  scan_interval: 3600
  #  name: "EPEX Price Prediction Evaluation"
  #  unit_of_measurement: ct/kWh
  #  value_template: "{{ value_json.prices[0].total }}"
  #  json_attributes:
  #    - prices
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
    name: EPEX Price History
    unit_of_measurement: ct/kWh
    texttemplate: "%{y:.0f}"
    mode: lines+text
    textposition: top right
  - entity: sensor.epex_price_prediction
    attribute: dummy
    name: EPEX Price Prediction
    unit_of_measurement: ct/kWh
    texttemplate: "%{y:.0f}"
    mode: lines+text
    textposition: top right
    filters:
      - fn: |-
          ({meta}) => ({
              xs: meta.prices.map(p => new Date(p.startsAt)),
              ys: meta.prices.map(p => p.total)
          })
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

