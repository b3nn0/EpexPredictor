# EpexPredictor

Runs the [EpexPredictor](https://github.com/b3nn0/EpexPredictor) electricity-price
forecast API on Home Assistant OS. Forecasts are calculated locally from external
market and weather data. An internet connection is required.

## Installation

1. Open **Settings → Apps → App store**.
2. Open the menu, select **Repositories**, and add
   `https://github.com/Elmardus/EpexPredictor`.
3. Install **EpexPredictor**, review its configuration, and start it.
4. Check the App logs while initial downloads and training complete.

Currently supported architecture: **amd64**. Installation builds the container
on the host and may take some time.

## Configuration

| Option | Default | Description |
| --- | --- | --- |
| `entsoe_api_key` | Empty | Optional ENTSO-E Transparency Platform key. Adds price and load data; required for some regions. See upstream documentation for regional requirements. |
| `use_nordpool` | `false` | Enables Nord Pool as an additional upstream price source. Enable only when the intended use complies with its licence and terms. |

The data directory is managed by the App. Training duration, background refresh
schedule and base-model regions are controlled by upstream code, not App settings.
Forecast region, horizon and price adjustments are API request parameters. This
allows different consumers to request different forecasts from the same server.

## Network and API access

No host port is published by default. Home Assistant can reach the API over the
internal App network without publishing a port. An empty or disabled host-port
field in **Configuration → Network** is therefore expected.

The API listens internally on TCP port **8000**. Use the Supervisor-generated App
hostname: the App slug with underscores replaced by hyphens, for example
`<repository-id>-epexpredictor`. A manually installed local App uses
`local-epexpredictor`.

- `/prices`: prices with ISO timestamps and the published-price cutoff.
- `/prices_short`: parallel `s` (Unix seconds) and `t` (prices) arrays.
- `/docs`: interactive upstream API documentation, including evaluation endpoints.

Example request with no tax or surcharge:

```text
http://<app-hostname>:8000/prices_short?region=DE&unit=EUR_PER_KWH
```

| Request parameter | Meaning |
| --- | --- |
| `region` | Supported bidding zone. |
| `hours` | Requested output horizon, subject to available forecast inputs. |
| `unit` | `CT_PER_KWH`, `EUR_PER_KWH` or `EUR_PER_MWH`. |
| `surcharge` | Fixed addition in **ct/kWh**, before percentage tax and unit conversion. |
| `taxPercent` | Percentage applied after adding the surcharge. |
| `timezone` | IANA timezone for output and interpreting timestamps without an offset. |
| `startTs` | Optional starting timestamp. |
| `hourly` | Return hourly averages. |
| `evaluation` | Predict known periods as well, for model evaluation. |

Consult `/docs` for current defaults and the complete API. Home Assistant's
built-in REST integration can consume the compact format. Choose an appropriate
horizon for HA's entity attribute-size constraints. Account for adjustments
already applied by consumers to avoid adding them twice.

For browser or other LAN-client access, publish a host port in App network
settings, then open `http://<ha-host>:<published-port>/docs`. The API has no
authentication; do not publish it directly to the internet.

## Operation and resources

First startup downloads historical inputs and trains shared models for all
available base regions. Requesting one region does not restrict that initial
work to that region. Later requests use cached forecasts; background tasks update
inputs and retrain models when needed.

The process keeps libraries, models and data in memory between updates. Resource
requirements depend on available regions, stored history and requested operations;
training temporarily increases CPU and memory use. This is a long-running server,
not a scheduled process that exits after producing each forecast.

Data caches live under `/data`, survive restarts and are included in backups
that include this App. Models are retrained at startup using cached data.
No privileged mode, HA API access or HA configuration-directory mount is needed.

## Troubleshooting

- **Slow startup:** inspect logs for download/training progress. Repeated restarts
  interrupt that work.
- **Source errors:** check internet access and upstream data availability. An
  ENTSO-E key provides an additional source where supported.
- **Health check:** the container checks API responsiveness after a startup grace
  period. This does not verify forecast freshness or quality. No Supervisor HTTP
  watchdog is enabled by default.

## Upstream and data sources

Prediction and training are provided by EpexPredictor. This App adds HA packaging
and a small transport patch that spaces Energy-Charts requests, allows slower
responses and checks HTTP status before parsing. See the App changelog for changes.

Upstream code uses the BSD-3-Clause licence. Market and weather data retain their
providers' licences and attribution requirements; see the upstream project and
API documentation for data-source credits.
