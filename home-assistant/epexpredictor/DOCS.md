# EpexPredictor App

Runs the unchanged upstream FastAPI server. Home Assistant configuration is the
runtime entry point. A small build-time compatibility patch spaces Energy-Charts
requests at least 31 seconds apart, allows 60-second responses, and checks HTTP
status before parsing. Prediction and background updates remain upstream-owned.

## Install the first development build

Copy this `epexpredictor` directory to `/addons/epexpredictor` on Home Assistant OS,
then check for updates in the App store and install it from **Local apps**.
The initial version targets amd64; other architectures require a verified build.
The build downloads upstream revision
`cb3f98ec222f9fddd13308c304c54467ad3bccb6` directly. It does not build uncommitted
changes elsewhere in this checkout. Update `UPSTREAM_REVISION` deliberately when
updating the bundled server. Upstream dependencies are currently unpinned, so the
source revision alone does not guarantee byte-for-byte reproducible builds.

## Options

- **entsoe_api_key:** optional ENTSO-E transparency API key. Recommended upstream
  for additional price/load data; some regions require it. NL can run without it.
- **use_nordpool:** optional extra upstream data source, disabled by default.
  Enable only when your use complies with its data licence/terms.

Region, horizon, VAT and surcharge are API request parameters, not server options.
Your existing HA tariff calculation can therefore remain unchanged.

## Startup and storage

First startup downloads historical inputs and trains shared models for all
available regions. It may take a long time and use substantial CPU/memory.
Selecting NL in a request does not limit initial training to NL.
Inspect App logs for progress. The built-in container health check waits up to
one hour for startup; becoming unhealthy does not itself restart the container.
No Supervisor HTTP watchdog is configured to interrupt slow first-time training.
The check verifies API responsiveness, not forecast freshness or quality.

Upstream uses `/data` for its persistent data caches. Normal App backups include
this storage. The server still needs internet access for market/weather inputs.
No HA API access, host networking, privileged mode or HA configuration mount is needed.

## Connect

For a locally installed App, HA Core can use:

```text
http://local-epexpredictor:8000/prices_short?region=NL&hours=120&unit=EUR_PER_KWH&taxPercent=21&surcharge=0&timezone=Europe%2FAmsterdam
```

For a repository install, use the Supervisor-generated App hostname instead of
`local-epexpredictor`. Keep published NordPool prices preferred in the consumer.
The returned price includes VAT but no fixed costs with the example above; keep
the existing HA cost addition to avoid applying VAT twice.

LAN port publishing is off by default. To open `/docs` in your browser, map port
8000 in the App network settings. The API has no authentication; do not expose it
to the internet. Internal HA access does not require a published port.

Before switching live sensors, verify a nonempty `/prices_short` response with
equal-length `s` and `t` arrays, 15-minute timestamps and the requested horizon.
Then verify cache persistence after restart and monitor resource use. Leave the
public feed configured until this local validation passes.
