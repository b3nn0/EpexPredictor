# Using EpexPredictor

EpexPredictor runs locally and provides electricity price forecasts for Home
Assistant and other applications. It needs an internet connection to download
market and weather data.

## Start the App

1. Open the App's **Configuration** tab and review the settings below.
2. Select **Start** on the **Info** tab.
3. Open **Logs** to follow progress. The first start can take a while as historical
   data is downloaded and the forecasting model is prepared.

Once startup finishes, the App keeps its forecasts up to date automatically.

## Settings

| Setting | What to enter |
| --- | --- |
| **ENTSO-E API key** | An optional ENTSO-E Transparency Platform key for additional price and load data. Leave empty if you do not have a key. Some regions require one. |
| **Enable Nord Pool data source** | Enable to use Nord Pool as an additional price source, provided your use complies with its licence and terms. Disabled by default. |

Choose your region, surcharge and tax percentage when setting up the Home
Assistant integration or another application that uses these forecasts.

## Use the forecasts in Home Assistant

Install the EpexPredictor integration separately from the App, then:

1. Open **Settings → Devices & services**.
2. Select **Configure** on the discovered EpexPredictor card. Alternatively, select
   **Add integration → EpexPredictor → Local App**.
3. Choose your region and enter your surcharge and tax percentage.

The integration adds forecast sensors that can be used in dashboards. See the
[integration guide](../../custom_components/epexpredictor/README.md) for installation,
settings and entity descriptions.

## Open the API in a browser

This is optional; the Home Assistant integration does not need a published port.
An empty or disabled port under **Configuration → Network** is normal.

To use the interactive API documentation from another device on your network:

1. Set a host port for the API under **Configuration → Network** and save.
2. Restart the App if prompted.
3. Open `http://<ha-host>:<host-port>/docs` in your browser.

The API has no authentication. Keep access on your local network and do not expose
the port directly to the internet.

## If something is not working

- **The App is still starting:** check **Logs** for progress and let it finish.
  Restarting repeatedly interrupts downloads and model preparation.
- **The integration cannot find the App:** check that the App is running and up
  to date, then try **Add integration → EpexPredictor → Local App** again.
- **Forecasts are unavailable:** check **Logs** for connection or data-source
  errors. Confirm internet access and check the API key if one is configured.
