# EpexPredictor integration

A Home Assistant integration for the EpexPredictor API. Install it separately
from the optional server App; it can also connect to a public or self-hosted
EpexPredictor server. Prediction and training remain on the server.

## Installation

### HACS

HACS requires a published repository version containing `custom_components/epexpredictor`.
An integration available only on a development branch must be installed manually
until that version is published; the App repository alone does not install it.

Once available, open **HACS → Custom repositories**, add this repository's GitHub
URL with type **Integration**, and download **EpexPredictor**. Restart Home
Assistant, then follow the setup steps below.

### Manual installation

Copy the `custom_components/epexpredictor` directory from a repository version
containing the integration into `<HA configuration directory>/custom_components/`.
Restart Home Assistant, then follow the setup steps below.

## Setup

After installing the custom integration and restarting Home Assistant, open
**Settings → Devices & services → Add integration → EpexPredictor**.

Choose **Local App**, **Public EpexPredictor server**, or **Custom server**. The
custom field accepts a base URL, without API endpoint paths or query parameters.
Local App discovery requires a running App version that supports Supervisor
discovery. Otherwise, its internal hostname can be entered as a custom URL.
The selected server is used explicitly; there is no automatic remote fallback.

Choose a region, fixed surcharge and percentage tax. The surcharge is in
**ct/kWh before tax**; sensor values are in **EUR/kWh**. Do not add the same costs
again in downstream consumers. Each configuration entry represents one server
and region. Add another entry to follow another region. Settings remain editable
through the entry's **Configure** action.

## Entities

- **Price prediction:** current interval's price, with compact `s` timestamp and
  `t` price arrays in its attributes for plotting or other consumers.
- **Forecast fetched:** time of the last successful forecast fetch. It is not a
  timestamp of model training or of upstream source publication.
- **Price prediction evaluation:** disabled by default. Enable it from the device's
  entities to fetch predictions for already-known prices as well. Evaluation is
  not requested while the entity is disabled. A failed evaluation request does
  not make the regular forecast unavailable.

Forecasts refresh periodically. Price entity states also advance at interval
boundaries without an additional HTTP request. An unavailable server makes its
entities unavailable; Home Assistant retries rather than changing servers.

## Discovery

The App announces its internal API URL after upstream startup completes. If the
integration is installed and loaded, Home Assistant offers a discovered setup
flow; region and pricing settings still require confirmation. Discovery does
not install the custom integration, publish a host port, or configure tariffs.

## Replacing YAML REST sensors

Validate the new entities first. Then remove the old REST configuration and
reload REST, updating consumers to the new entity IDs as needed. The integration
does not automatically remove YAML sensors or take over their entity IDs/history.
