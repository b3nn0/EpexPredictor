# EpexPredictor integration

Adds electricity price forecasts from an EpexPredictor server to Home Assistant.
Connect to the local App, the public server, or another self-hosted EpexPredictor server.

## Installation

### HACS

Open **HACS → Custom repositories**, add this repository's GitHub
URL with type **Integration**, and download **EpexPredictor**. Restart Home
Assistant, then follow the setup steps below.

### Manual installation

Copy the `custom_components/epexpredictor` directory from a repository version
containing the integration into `<HA configuration directory>/custom_components/`.
Restart Home Assistant, then follow the setup steps below.

## Setup

After installing the custom integration and restarting Home Assistant, open
**Settings → Devices & services → Add integration → EpexPredictor**.

Choose a server:

- **Local App:** use an installed, running EpexPredictor App.
- **Public EpexPredictor server:** use the public service.
- **Custom server:** enter the address of another EpexPredictor server, such as
  `http://hostname:8000`.

Then choose the **region**, **fixed surcharge** and **tax percentage**. The
surcharge is in **ct/kWh before tax**. The tax percentage is applied after the
surcharge is added, and sensor prices are shown in **EUR/kWh**. Avoid adding costs
that are already included elsewhere in your price calculations.

Use **Configure** on the integration entry to change these settings later.
To follow another region, add another EpexPredictor integration entry.

## Entities

- **Price prediction:** the current period's price, using published prices where
  available and predictions for future periods. Also includes upcoming prices
  for displaying in a forecast graph.
- **Forecast fetched:** when Home Assistant last successfully retrieved the forecast.
- **Price prediction evaluation:** the model's predicted prices, including periods
  with published prices, so you can compare predictions with actual market prices.
  Disabled by default; enable it from the device's entity list when needed.

Forecasts and the current price update automatically. If the server cannot be
reached, the sensors show **Unavailable** until the connection recovers.
