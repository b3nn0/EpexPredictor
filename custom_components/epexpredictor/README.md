# EpexPredictor integration

Adds electricity price forecasts from an EpexPredictor server to Home Assistant.
Connect to the local App, the public server, or another self-hosted EpexPredictor server.

## Installation

### Marketplace

[![Open in Marketplace](https://my.home-assistant.io/badges/marketplace_repository.svg)](https://my.home-assistant.io/redirect/marketplace_repository/?owner=b3nn0&repository=EpexPredictor&category=integration)

Open **Settings → Marketplace → Add from link**, add
`https://github.com/b3nn0/EpexPredictor`, and install **EpexPredictor**. Restart
Home Assistant, then follow the setup steps below. Marketplace requires
Home Assistant 2026.11 or later; manual installation is also available below.

### Manual installation

Copy the `custom_components/epexpredictor` directory from a repository version
containing the integration into `<HA configuration directory>/custom_components/`.
Restart Home Assistant, then follow the setup steps below.

## Setup

For **Local App** setup, [install and start the App](../../README.md#home-assistant-app-and-integration)
first. The integration can also use a public or custom server without the App.

[![Add integration](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=epexpredictor)

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

Forecasts refresh every 15 minutes from the public server and every 5 minutes
from other servers. The current price advances automatically each quarter-hour.
Forecast arrays are available to live graphs but are excluded from Recorder;
the sensor's price history is still recorded. If the server cannot be
reached, the sensors show **Unavailable** until the connection recovers.

## Forecast graph

See the [Plotly Graph Card example](../../README.md#display-eg-via-plotly-graph-card)
in the main README. Replace its entity ID with your **Price prediction** entity.
